"""
prismaticvla_train.py

Training script for Vision-Language-Action (VLA) Policies, built on top of pretrained VLMs, trained using mixtures of
the Open-X Embodiment dataset, with added Question-Answer pairs. Performs training in native PyTorch, using Fully-Sharded 
Data Parallel (FSDP) to run distributed across GPUs (and nodes). By default, assumes that CUDA toolkit is >= 11.0 (to 
support BF16 mixed precision).

Run with:
    - [Single Node One-GPU (Debug)] : torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/prismaticvla_train.py
    - [Single Node Multi-GPU (= $K)]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/prismaticvla_train.py
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union

import draccus
import torch
import torch.distributed as dist
import yaml
from peft import LoraConfig

from prismatic.conf import DatasetConfig, DatasetRegistry, ModelConfig, ModelRegistry
from prismatic.models import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform, get_vlm
from prismatic.overwatch import initialize_overwatch
from prismatic.training import PrismaticVLAMetrics, get_train_strategy
from prismatic.util import set_global_seed
from prismatic.vla import get_prismatic_vla_dataset_and_collator, get_prismatic_vla_json_dataset_and_collator
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['HF_TOKEN'] = "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"

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


@dataclass
class TrainConfig:
    # fmt: off

    # ModelConfig (`prismatic/conf/models.py`); override with --model.type `ModelRegistry.<MODEL>.model_id`
    model: ModelConfig = field(
        default_factory=ModelConfig.get_choice_class(ModelRegistry.PRISM_DINOSIGLIP_7B.model_id)
    )

    # DatasetConfig (`prismatic/conf/datasets.py`); override with --dataset.type `DatasetRegistry.<DATASET>.dataset_id`
    # Specify this, if you want to train on a dataset in JSON format, with images in an associated folder
    json_dataset: Optional[DatasetConfig] = field(
        default_factory=DatasetConfig.get_choice_class(DatasetRegistry.RLDS_OXE_QNA_MIX1.dataset_id)
    )

    # Additional Parameters (should pass them if using JSON dataset -> above)
    train_val_split_3qna: Optional[float] = 0.01                    # Training-validation split for conversations in RLDS Open-X datasets with 3 QnAs
    train_val_split_multi_qna: Optional[float] = 0.9                # Training-validation split for conversations in RLDS Open-X datasets with more than 3 QnAs
    randomize_qnas: bool = True                                     # Whether to randomize the order of question-answers for an image

    # RLDS Directory Paths
    # Specify this, if you want to train on a dataset in RLDS format, with added QnA pairs
    rlds_data_root_dir: Optional[Path] = Path("/home/ubuntu/tensorflow_datasets")  # Path to Open-X dataset directory
    rlds_data_mix: Optional[str] = "taco+mutex"                     # Name of dataset mixture
    
    run_root_dir: Path = Path("/home/ubuntu/prismatic_vlas/runs")   # Path to directory to store logs & checkpoints

    # Stage
    stage: str = "lora-finetune"                                    # Training Stage
    
    # Resume Run Parameters
    pretrained_checkpoint: Optional[Path] = None                    # Absolute Path to Checkpoint
    is_resume: bool = True                                          # Whether we are continuing a prior training run (only applicable given pretrained checkpoint)
    resume_step: Optional[int] = None                               # Global Step to Resume (should match checkpoint)
    resume_epoch: Optional[int] = None                              # Epoch to Resume (should match checkpoint)

    # Run Arguments
    run_id: Optional[str] = None                                    # Run ID for logging, Weights & Biases
    run_id_note: str = "vla"                                        # Extra note for logging, Weights & Biases
    save_interval: int = 500                                        # Interval for saving checkpoints (in steps)
    overwrite: bool = True                                          # Whether to overwrite checkpoints
    image_aug: bool = False                                         # Whether to enable image augmentations
    seed: int = 7                                                   # Random seed (for reproducibility)
    shuffle_buffer_size: int = 100000                               # Size of Shuffle Buffer

    # HF Hub Credentials (for any gated models)
    hf_token: Union[str, Path] = "HF_TOKEN"                         # Environment variable or Path to HF Token

    # Tracking Parameters
    trackers: Tuple[str, ...] = ("jsonl", )                         # Trackers to initialize (if W&B, add config!)
    wandb_project: str = "openvla"                                  # Name of W&B project to log to (use default!)
    wandb_entity: str = "stanford-voltron"                          # Name of entity to log under

    # LoRA Arguments
    use_lora: bool = True                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 64                                             # Rank of LoRA weight matrix
    lora_alpha: int = 64                                            # LoRA alpha
    lora_dropout: float = 0.05                                      # Dropout applied to LoRA weights
    lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]  # LoRA target modules

    # Action Head Arguments
    action_head_specifier: str = "gelu"                             # Action Head Specifier (can be 'linear', 'relu', 'gelu', or 'diffusion')
    use_map: bool = True                                            # Whether to use Multi-head attention pooling, or mean pooling across sequence dimension
    num_map_heads: int = 8                                          # Number of attention heads (if using MAP)

    # Weights for losses
    lm_loss_weight: float = 0.5                                     # Weight for language modeling loss
    l2_loss_weight: float = 0.5                                     # Weight for L2 loss

    # Additional params
    use_layer_output_pooler: bool = True                            # If True, we process the outputs of all hidden layers by pooling them before sending to the action head.
    lop_mlp_type: str = "linear"                                    # MLP type in the Layer output pooler (can be 'linear', 'relu', or 'gelu')
    lop_num_map_heads: int = 4                                      # Number of attention heads in the Layer output pooler

    hidden_layer_aggregation: Optional[str] = "average"             # Whether to average all hidden layer outputs or just take the last hidden layer output. Can be 'average' or 'last' and should be specified if not using LOP.

    def __post_init__(self) -> None:
        """Set optimization parameters based on `stage` in {"align", "finetune"}."""
        if self.stage == "align":
            self.epochs = self.model.align_epochs
            self.max_steps = self.model.align_max_steps
            self.global_batch_size = self.model.align_global_batch_size
            self.per_device_batch_size = self.model.align_per_device_batch_size

            self.learning_rate = self.model.align_learning_rate
            self.weight_decay = self.model.align_weight_decay
            self.max_grad_norm = self.model.align_max_grad_norm
            self.lr_scheduler_type = self.model.align_lr_scheduler_type
            self.warmup_ratio = self.model.align_warmup_ratio

            self.train_strategy = self.model.align_train_strategy

        elif self.stage.endswith("finetune"):
            self.epochs = self.model.finetune_epochs
            self.max_steps = self.model.finetune_max_steps
            self.global_batch_size = self.model.finetune_global_batch_size
            self.per_device_batch_size = self.model.finetune_per_device_batch_size

            self.learning_rate = self.model.finetune_learning_rate
            self.weight_decay = self.model.finetune_weight_decay
            self.max_grad_norm = self.model.finetune_max_grad_norm
            self.lr_scheduler_type = self.model.finetune_lr_scheduler_type
            self.warmup_ratio = self.model.finetune_warmup_ratio

            self.train_strategy = self.model.finetune_train_strategy

        else:
            raise ValueError(f"Stage `{self.stage}` is not supported!")

    # fmt: on


@draccus.wrap()
def train(cfg: TrainConfig) -> None:
    overwatch.info("PrismaticVLA Training :: Warming Up")

    # Note => Under `torchrun` initializing `overwatch` will automatically set up `torch.distributed`
    torch.cuda.set_device(device_id := overwatch.local_rank())
    torch.cuda.empty_cache()

    # Check datasets
    # Note => The JSON dataset takes precedence
    assert cfg.json_dataset is not None or cfg.rlds_data_root_dir is not None, "Specify either a JSON Dataset config, or a RLDS data directory!"
    use_json_dataset = True if cfg.json_dataset is not None else False

    # Create Unique Run Name & Save Directory
    model_id = cfg.model.model_id
    data_id = cfg.json_dataset.dataset_id if use_json_dataset else cfg.rlds_data_mix

    if cfg.randomize_qnas:
        data_id += "+r_qnas"
    if cfg.use_layer_output_pooler:
        model_id += "+lop"

    cfg.run_id = f"{data_id}+{model_id}+{cfg.action_head_specifier}+stage-{cfg.stage}+x{cfg.seed}+e{cfg.epochs}" if cfg.run_id is None else cfg.run_id
    cfg.run_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        cfg.run_id += "--image_aug"

    # Start =>> Build Directories and Set Randomness
    overwatch.info('"Do or do not; there is no try."', ctx_level=1)
    hf_token = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else os.environ[cfg.hf_token]
    worker_init_fn = set_global_seed(cfg.seed, get_worker_init_fn=True)
    os.makedirs(run_dir := (cfg.run_root_dir / cfg.run_id), exist_ok=True)
    os.makedirs(cfg.run_root_dir / cfg.run_id / "checkpoints", exist_ok=True)

    # Save Configuration =>> additionally save a JSON version for later HF Integration
    if overwatch.is_rank_zero():
        draccus.dump(cfg, open(run_dir / "config.yaml", "w"))
        with open(run_dir / "config.yaml", "r") as f_yaml, open(run_dir / "config.json", "w") as f_json:
            yaml_cfg = yaml.safe_load(f_yaml)
            json.dump(yaml_cfg, f_json, indent=2)

    # Load Vision Backbone --> on CPU, in Full Precision (initializing model, image_transform via TIMM)
    overwatch.info(f"Loading Vision Backbone [bold]{cfg.model.vision_backbone_id}[/] via TIMM ")
    vision_backbone, image_transform = get_vision_backbone_and_transform(
        cfg.model.vision_backbone_id, image_resize_strategy=cfg.model.image_resize_strategy
    )

    # Load LLM Backbone --> on CPU, in Full Precision (initializing Tokenizer + handling special tokens if necessary)
    overwatch.info(f"Loading Pretrained LLM [bold]{cfg.model.llm_backbone_id}[/] via HF Transformers")
    # Wrap LLM with PEFT (LoraConfig), if applicable
    lora_config = None
    if cfg.use_lora:
        assert "lora" in cfg.stage, "Should use LoRA fine-tuning stage if `use_lora` is set to True!"
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            target_modules=cfg.lora_target_modules,
            task_type="CAUSAL_LM",
        )

    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        cfg.model.llm_backbone_id, 
        llm_max_length=cfg.model.llm_max_length, 
        hf_token=hf_token,
        use_lora=cfg.use_lora,
        lora_config=lora_config,
    )

    # Create VLM => wraps `vision_backbone` and `llm`
    overwatch.info(f"Instantiating PrismaticVLM `{model_id}` for Training Stage = `{cfg.stage}`")
    action_head_configs = {
        "action_head_specifier": cfg.action_head_specifier,
        "use_map": cfg.use_map,
        "num_map_heads": cfg.num_map_heads,
    }

    layer_output_pooler_configs = None
    if cfg.use_layer_output_pooler:
        layer_output_pooler_configs = {
            "lop_mlp_type": cfg.lop_mlp_type,
            "lop_num_map_heads": cfg.lop_num_map_heads,
        }


    vlm = get_vlm(
        model_id,
        cfg.model.arch_specifier,
        vision_backbone,
        llm_backbone,
        enable_mixed_precision_training=cfg.model.enable_mixed_precision_training,
        use_layer_output_pooler=cfg.use_layer_output_pooler,
        layer_output_pooler_configs=layer_output_pooler_configs,
        hidden_layer_aggregation=cfg.hidden_layer_aggregation,
        use_action_head=True,
        action_head_configs=action_head_configs,
        seed=cfg.seed,
    )

    # [Validate] Model should be in Full Precision!
    for param in vlm.parameters():
        assert param.dtype == torch.float32, f"Loaded VLM parameter not in full precision: {param}"

    # [Explicit] Call to `freeze_backbones` here for clarity => will log exactly what is frozen / what's not!
    overwatch.info(f"Invoking `VLM.freeze_backbones()` for `{model_id}` => Training Stage: `{cfg.stage}`")
    vlm.freeze_backbones(cfg.stage)

    # Load Weights from Checkpoint (depends on stage, config)
    overwatch.info(f"Invoking `VLM.load_checkpoint()` for `{model_id}` => Training Stage: `{cfg.stage}`")
    vlm.load_from_checkpoint(cfg.stage, run_dir, pretrained_checkpoint=cfg.pretrained_checkpoint)

    # Print number of total/trainable model parameters
    num_params = sum(p.numel() for p in vlm.parameters())
    num_trainable_params = sum(p.numel() for p in vlm.parameters() if p.requires_grad)
    overwatch.info(
        f"# Parameters (in millions): {num_params / 10**6:.3f} Total, {num_trainable_params / 10**6:.3f} Trainable"
    )

    # Get PrismaticVLA Dataset & Collator
    if use_json_dataset:
        overwatch.info(f"Creating Dataset `{cfg.json_dataset.dataset_id}` => Stage: `{cfg.stage}`")
        prismatic_vla_json_dataset, collator = get_prismatic_vla_json_dataset_and_collator(
            cfg.stage,
            cfg.json_dataset,
            image_transform,
            tokenizer,
            prompt_builder_fn=llm_backbone.prompt_builder_fn,
            padding_side=tokenizer.padding_side,
            train_val_split_3qna=cfg.train_val_split_3qna,
            train_val_split_multi_qna=cfg.train_val_split_multi_qna,
            randomize_qnas=cfg.randomize_qnas,
        )
        n_train_examples = len(prismatic_vla_json_dataset)

    else:
        overwatch.info(f"Creating PrismaticVLA Open-X Dataset with Mixture `{cfg.rlds_data_mix}` => Stage: `{cfg.stage}`")
        prismatic_vla_dataset, collator = get_prismatic_vla_dataset_and_collator(
            cfg.rlds_data_root_dir,
            cfg.rlds_data_mix,
            image_transform=image_transform,
            tokenizer=tokenizer,
            prompt_builder_fn=vlm.llm_backbone.prompt_builder_fn,
            default_image_resolution=vision_backbone.default_image_resolution,
            shuffle_buffer_size=cfg.shuffle_buffer_size,
            image_aug=cfg.image_aug,
        )
        n_train_examples = len(prismatic_vla_dataset)

        # Save dataset statistics for de-normalization at inference time
        if overwatch.is_rank_zero():
            save_dataset_statistics(prismatic_vla_dataset.dataset_statistics, run_dir)

    # Create Train Strategy
    overwatch.info(f"Initializing Train Strategy `{cfg.train_strategy}`")
    train_strategy = get_train_strategy(
        train_strategy=cfg.train_strategy,
        vlm=vlm,
        device_id=device_id,
        stage=cfg.stage,
        epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        global_batch_size=cfg.global_batch_size,
        per_device_batch_size=cfg.per_device_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        enable_gradient_checkpointing=cfg.model.enable_gradient_checkpointing,
        enable_mixed_precision_training=cfg.model.enable_mixed_precision_training,
        reduce_in_full_precision=cfg.model.reduce_in_full_precision,
        worker_init_fn=worker_init_fn,
    )
    train_strategy.run_setup(run_dir=run_dir, n_train_examples=n_train_examples)

    # Create Metrics =>> Handles on the fly tracking, logging to specified trackers (e.g., JSONL, Weights & Biases)
    overwatch.info(f"Creating Metrics with Active Trackers => `{cfg.trackers}`")
    metrics = PrismaticVLAMetrics(
        cfg.trackers,
        cfg.run_id,
        run_dir,
        draccus.encode(cfg),
        wandb_project=cfg.wandb_project,
        wandb_entity=cfg.wandb_entity,
        resume_step=cfg.resume_step,
        resume_epoch=cfg.resume_epoch,
    )

    # Run PrismaticVLA Training
    overwatch.info("Starting PrismaticVLA Training Loop")
    if use_json_dataset:
        train_strategy.run_prismatic_vla_json_training(
            prismatic_vla_json_dataset,
            collator,
            metrics,
            stage=cfg.stage,
            seed=cfg.seed,
            save_interval=cfg.save_interval,
            overwrite=cfg.overwrite,
            lm_loss_weight=cfg.lm_loss_weight,
            l2_loss_weight=cfg.l2_loss_weight,
        )
        
    else:
        train_strategy.run_prismatic_vla_training(
            prismatic_vla_dataset,
            collator,
            metrics,
            save_interval=cfg.save_interval,
            overwrite=cfg.overwrite,
            lm_loss_weight=cfg.lm_loss_weight,
            l2_loss_weight=cfg.l2_loss_weight,
        )

    # Finalize
    overwatch.info("Done with Training =>> Finalizing Metrics")
    metrics.finalize()

    # And... we're done!
    overwatch.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    train()