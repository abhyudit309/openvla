"""
base_strategy.py

Abstract class definition of a (distributed) training strategy, with full annotations of class methods, utility
functions, and initialization logic.

Training Strategies (DDP, FSDP-Grad, FSDP-Full) tend to have a lot of repeated components; this class does a lot of
heavy lifting.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset
from tqdm import tqdm
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.models.vlms import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.training.metrics import Metrics, VLAMetrics, PrismaticVLAMetrics
from prismatic.util import check_bloat16_supported
from prismatic.util.batching_utils import SplitModalitySampler
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.util.data_utils import (
    PaddedCollatorForActionPrediction, 
    PaddedCollatorForLanguageModeling,
    PaddedCollatorForLanguageModelingAndActionPrediction
)

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# === Abstract Base Class for an arbitrary Training Strategy ===
class TrainingStrategy(ABC):
    def __init__(
        self,
        vlm: PrismaticVLM,
        device_id: int,
        stage: str,
        epochs: int,
        max_steps: Optional[int],
        global_batch_size: int,
        per_device_batch_size: int,
        learning_rate: float,
        weight_decay: float,
        max_grad_norm: float,
        lr_scheduler_type: str,
        warmup_ratio: float,
        enable_gradient_checkpointing: bool = True,
        enable_mixed_precision_training: bool = True,
        reduce_in_full_precision: bool = False,
        mixed_precision_dtype: torch.dtype = torch.bfloat16,
        worker_init_fn: Optional[Callable[[int], None]] = None,
        **_: str,
    ) -> None:
        self.vlm, self.device_id, self.stage = vlm, device_id, stage

        # Get relevant VLM instance parameters before they get (potentially) wrapped
        self.all_module_keys, self.trainable_module_keys = self.vlm.all_module_keys, self.vlm.trainable_module_keys
        self.llm_transformer_layer_cls = self.vlm.llm_backbone.transformer_layer_cls

        # Optimization Parameters
        self.epochs, self.max_steps = epochs, max_steps
        self.global_batch_size, self.per_device_batch_size = global_batch_size, per_device_batch_size

        self.learning_rate, self.weight_decay, self.max_grad_norm = learning_rate, weight_decay, max_grad_norm
        self.lr_scheduler_type, self.warmup_ratio = lr_scheduler_type, warmup_ratio

        # Generic Strategy Parameters
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.enable_mixed_precision_training = enable_mixed_precision_training
        self.reduce_in_full_precision = reduce_in_full_precision
        self.mixed_precision_dtype = mixed_precision_dtype

        # DataLoader Parameters
        self.worker_init_fn = worker_init_fn

        # Optimizers & Scheduler (initialized in `run_setup`)
        self.optimizer, self.lr_scheduler = None, None

        # Lightweight Validation
        assert (
            self.global_batch_size % self.per_device_batch_size == 0
        ), "Per-device batch size must evenly divide global batch size!"
        self.grad_accumulation_steps = self.global_batch_size // self.per_device_batch_size // overwatch.world_size()
        if self.enable_mixed_precision_training:
            assert self.mixed_precision_dtype == torch.bfloat16, "Only BF16 mixed precision training is supported!"
            assert check_bloat16_supported(), "BFloat16 is not supported on this hardware; unset `mixed_precision`"

    @abstractmethod
    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
        only_trainable: bool = True,
        overwrite: bool = False,
    ) -> None: ...

    @abstractmethod
    def run_setup(self, run_dir: Path, n_train_examples: int) -> None: ...

    @abstractmethod
    def clip_grad_norm(self) -> None: ...

    def run_training(
        self,
        dataset: Dataset,
        collator: PaddedCollatorForLanguageModeling,
        metrics: Metrics,
        stage: str = "finetune",
        batch_construction_strategy: str = "split-modality",
        seed: int = 7,
        save_interval: int = 2500,
        overwrite: bool = False,
    ) -> None:
        """Run the training loop for the given `dataset` and `collator`; log losses, results to `metrics`"""
        if "finetune" in stage and batch_construction_strategy == "split-modality":
            # Instantiate the split-modality sampler; if you want to extend with other batch construction schemes,
            #   (e.g., grouping by length) =>> can easily add them here!
            modality_lengths = dataset.get_modality_lengths()
            sampler = SplitModalitySampler(
                dataset,
                modality_lengths,
                global_batch_size=self.global_batch_size,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                seed=seed,
                drop_last=False,
            )

        else:
            sampler = DistributedSampler(
                dataset,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                shuffle=True,
                seed=seed,
                drop_last=False,
            )

        # Create a DataLoader with the initialized sampler, per-device-bsz, and collator
        dataloader = DataLoader(
            dataset,
            batch_size=self.per_device_batch_size,
            sampler=sampler,
            collate_fn=collator,
            num_workers=2,
            worker_init_fn=self.worker_init_fn,
        )

        # Max Steps vs. Epochs Computation
        steps_per_epoch = len(dataloader) // self.grad_accumulation_steps
        if self.max_steps is not None and steps_per_epoch < self.max_steps:
            # Just set `epochs` to some large number --> we'll short-circuit based on steps anyway
            self.epochs = 100

        # === Train ===
        status = metrics.get_status()
        with tqdm(
            total=(
                (self.epochs * (len(dataloader) // self.grad_accumulation_steps))
                if self.max_steps is None
                else self.max_steps
            ),
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            for epoch in range(self.epochs):
                self.vlm.train()
                sampler.set_epoch(epoch)

                # Zero-Gradients (just in case)
                self.optimizer.zero_grad()

                # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
                for train_idx, batch in enumerate(dataloader):
                    # [Contract] self.vlm.forward() must automatically compute `loss` and return!
                    with torch.autocast(
                        "cuda",
                        dtype=self.mixed_precision_dtype,
                        enabled=self.enable_mixed_precision_training,
                    ):
                        output: CausalLMOutputWithPast = self.vlm(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            pixel_values=batch["pixel_values"],
                            labels=batch["labels"],
                            multimodal_indices=batch["multimodal_indices"],
                        )
                        loss = output.loss

                    # Commit Loss (Prior to Gradient Accumulation Normalization)
                    metrics.commit(loss=loss)

                    # Normalize Loss to account for Gradient Accumulation --> Backward!
                    # [IMPORTANT] Technically speaking, doing gradient accumulation in this way is "incorrect"; this is
                    #             because in general, each batch has a *different number of masked out tokens* (because
                    #             we're instruct-tuning). Taking the mean over two unbalanced means != the right thing!
                    #
                    #             HOWEVER -- at least at the 7B scale, the "naive" approach is just as performant as
                    #             the "correct" implementation, without adding extra complexity.
                    #
                    # That being said =>> at the 13B scale, *no matter what we tried, ANY gradient accumulation is just
                    #   really bad for downstream performance. Initial investigation shows that BF16 accumulation
                    #   just really tanks in precision... and don't have a good/clean way to fix this. Would love for
                    #   someone to PR and fix this (and I'd greatly appreciate it!!!)
                    normalized_loss = loss / self.grad_accumulation_steps
                    normalized_loss.backward()

                    # Step =>> Only if Done w/ Gradient Accumulation
                    if (train_idx + 1) % self.grad_accumulation_steps == 0:
                        metrics.commit(update_step_time=True)

                        # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality-assumptions
                        self.clip_grad_norm()

                        # Optimizer & LR Scheduler Step
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()

                        # Push Metrics
                        metrics.commit(global_step=metrics.global_step + 1, lr=self.lr_scheduler.get_last_lr()[0])
                        status = metrics.push()

                        # Check for Save Interval or Max Steps & Save Checkpoint
                        if (terminate := (self.max_steps is not None and metrics.global_step >= self.max_steps)) or (
                            (metrics.global_step % save_interval) == 0):
                            self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item(), overwrite=overwrite)
                            dist.barrier()

                            if terminate:
                                return

                        # Update Progress Bar
                        progress.update()
                        progress.set_description(status)

            # Save checkpoint at the end (if `self.max_steps` is None)
            if (self.max_steps is None) and ((metrics.global_step % save_interval) != 0):
                self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item(), overwrite=overwrite)
                dist.barrier()

    # === VLA Training ===

    def run_vla_training(
        self,
        vla_dataset: IterableDataset,
        collator: PaddedCollatorForActionPrediction,
        action_tokenizer: ActionTokenizer,
        metrics: VLAMetrics,
        save_interval: int = 2500,
        save_full_model: bool = True,
    ) -> None:
        """Run the VLA training loop for the given `dataset` and `collator`; log losses, action metrics to `metrics`."""
        assert isinstance(vla_dataset, IterableDataset), "VLA training expects an IterableDataset!"
        assert self.grad_accumulation_steps == 1, "VLA training does not support gradient accumulation!"

        # Create a DataLoader =>> Set `num_workers` to 0; RLDS loader handles parallelism!
        dataloader = DataLoader(
            vla_dataset,
            batch_size=self.per_device_batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,
            worker_init_fn=self.worker_init_fn,
        )

        # === Train ===
        status = metrics.get_status()
        with tqdm(
            total=(self.epochs * len(dataloader)) if self.max_steps is None else self.max_steps,
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            self.vlm.train()

            # Zero Gradients (just in case)
            self.optimizer.zero_grad()

            # [Contract] DataLoader wraps RLDS Loader (`.as_numpy_iterator() =>> implicit `.repeat()`)
            #   => This means looping over the DataLoader is basically "infinite" (so no outer loop over epochs).
            #      Slightly breaks default PyTorch semantics, which is why we adaptively compute `epoch` below.
            for batch in dataloader:
                # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
                with torch.autocast(
                    "cuda", dtype=self.mixed_precision_dtype, enabled=self.enable_mixed_precision_training
                ):
                    # [Contract] self.vlm.forward() must automatically compute `loss` and return!
                    output: CausalLMOutputWithPast = self.vlm(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        pixel_values=batch["pixel_values"],
                        labels=batch["labels"],
                    )
                    loss = output.loss

                # Commit Loss =>> Backward!
                metrics.commit(loss=loss)
                loss.backward()

                # === Compute Action Token Accuracy & L1 Loss ===

                # To compute action token accuracy, we need to identify the locations of the action tokens
                # in both `output.logits` and `batch["labels"]`. We know that when "right" padding, we
                # insert `self.vlm.vision_backbone.num_patches` at index 1.
                #
                # Computing `action_prediction_accuracy` is then pretty straightforward:
                #   1) Extract "aligned" predictions & labels
                #   2) Compute boolean "mask" where "labels > 2" (where 2 is ID for `EOS_TOKEN`)
                #           => If masking out EOS, then it's just "labels != -100 (IGNORE_INDEX)
                #   3) Compute masked accuracy as `(preds == logits) & mask` --> sum/divide by # unmasked!
                action_preds = output.logits[:, self.vlm.vision_backbone.num_patches : -1].argmax(dim=2)
                action_gt = batch["labels"][:, 1:].to(action_preds.device)
                mask = action_gt > action_tokenizer.action_token_begin_idx

                # Compute Accuracy
                correct_preds = (action_preds == action_gt) & mask
                action_accuracy = correct_preds.sum().float() / mask.sum().float()

                # Compute L1 Loss on Predicted (Continuous) Actions
                continuous_actions_pred = torch.tensor(
                    action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
                )
                continuous_actions_gt = torch.tensor(
                    action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
                )
                action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

                # Commit Metrics
                metrics.commit(action_accuracy=action_accuracy, l1_loss=action_l1_loss, update_step_time=True)

                # Compute metrics per dataset --> only on rank_zero since we don't log them on other workers anyways
                if overwatch.is_rank_zero():
                    datasets = set(batch["dataset_names"])
                    if len(datasets) > 1:
                        for ds in datasets:
                            ds_mask = torch.tensor([elem == ds for elem in batch["dataset_names"]])
                            action_accuracy_ds = correct_preds[ds_mask].sum().float() / mask[ds_mask].sum().float()
                            continuous_actions_pred_ds = torch.tensor(
                                action_tokenizer.decode_token_ids_to_actions(
                                    action_preds[ds_mask][mask[ds_mask]].cpu().numpy()
                                )
                            )
                            continuous_actions_gt_ds = torch.tensor(
                                action_tokenizer.decode_token_ids_to_actions(
                                    action_gt[ds_mask][mask[ds_mask]].cpu().numpy()
                                )
                            )
                            action_l1_loss_ds = torch.nn.functional.l1_loss(
                                continuous_actions_pred_ds, continuous_actions_gt_ds
                            )
                            metrics.commit_for_dataset(
                                dataset_name=ds.decode(), action_accuracy=action_accuracy_ds, l1_loss=action_l1_loss_ds
                            )

                # === Gradient Step ===

                # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality assumptions
                self.clip_grad_norm()

                # Optimizer & LR Scheduler Step
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                # Compute epoch value using number of completed gradient steps
                epoch = (metrics.global_step + 1) // (len(vla_dataset) // self.global_batch_size)

                # Push Metrics
                metrics.commit(global_step=metrics.global_step + 1, epoch=epoch, lr=self.lr_scheduler.get_last_lr()[0])
                status = metrics.push()

                # Check for Save Interval or Max Steps & Save Checkpoint
                if (terminate := (self.max_steps is not None and metrics.global_step >= self.max_steps)) or (
                    (metrics.global_step % save_interval) == 0
                ):
                    self.save_checkpoint(
                        metrics.run_dir, metrics.global_step, epoch, loss.item(), only_trainable=not save_full_model
                    )
                    dist.barrier()

                    if terminate:
                        return

                # Update Progress Bar
                progress.update()
                progress.set_description(status)

    # === Prismatic VLA Training ===

    def run_prismatic_vla_training(
        self,
        prismatic_vla_dataset: IterableDataset,
        collator: PaddedCollatorForLanguageModelingAndActionPrediction,
        metrics: PrismaticVLAMetrics,
        save_interval: int = 2500,
        overwrite: bool = False,
        save_full_model: bool = True,
        lm_loss_weight: float = 0.5,
        l2_loss_weight: float = 0.5,
    ) -> None:
        """
        Run the Prismatic VLA training loop for the given `dataset` and `collator`; 
        log losses, action metrics to `metrics`.
        """
        assert isinstance(prismatic_vla_dataset, IterableDataset), "PrismaticVLA training expects an IterableDataset!"
        assert self.grad_accumulation_steps == 1, "PrismaticVLA training does not support gradient accumulation!"
        assert self.vlm.use_action_head, "PrismaticVLA should have an Action Head!"

        assert lm_loss_weight >= 0 and l2_loss_weight >= 0, "Weights for losses must be non-negative (>= 0)!"
        assert lm_loss_weight + l2_loss_weight == 1.0, "The weights for losses should sum to 1!"
        
        using_diffusion_action_head = self.vlm.action_head_configs["action_head_specifier"] == "diffusion"

        # Create a DataLoader =>> Set `num_workers` to 0; RLDS loader handles parallelism!
        dataloader = DataLoader(
            prismatic_vla_dataset,
            batch_size=self.per_device_batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,
            worker_init_fn=self.worker_init_fn,
        )

        # === Train ===
        status = metrics.get_status()
        with tqdm(
            total=(self.epochs * len(dataloader)) if self.max_steps is None else self.max_steps,
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            self.vlm.train()

            # Zero-Gradients (just in case)
            self.optimizer.zero_grad()

            # [Contract] DataLoader wraps RLDS Loader (`.as_numpy_iterator() =>> implicit `.repeat()`)
            #   => This means looping over the DataLoader is basically "infinite" (so no outer loop over epochs).
            #      Slightly breaks default PyTorch semantics, which is why we adaptively compute `epoch` below.
            for batch in dataloader:
                # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
                with torch.autocast(
                    "cuda", dtype=self.mixed_precision_dtype, enabled=self.enable_mixed_precision_training
                ):
                    # [Contract] self.vlm.forward() must automatically compute language modeling loss
                    output: CausalLMOutputWithPast = self.vlm(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        pixel_values=batch["pixel_values"],
                        labels=batch["labels"],
                        output_hidden_states=True,
                    )

                    if self.vlm.use_layer_output_pooler:
                        all_hidden_layer_outputs = torch.stack(output.hidden_states, dim=2)
                        pooled_hidden_states = self.vlm.layer_output_pooler(all_hidden_layer_outputs)
                    else:
                        # We just consider the last layer output
                        pooled_hidden_states = output.hidden_states[-1]

                    gt_actions = batch["actions"].to(pooled_hidden_states.device)

                    if using_diffusion_action_head:
                        batch_size = pooled_hidden_states.shape[0]

                        # Generate random time and noise
                        time = torch.randint(
                            0,
                            self.vlm.action_head.diffusion_steps,
                            (self.vlm.action_head.n_diffusion_samples, batch_size, 1),
                            generator=self.vlm.action_head.rng,
                            device=pooled_hidden_states.device,
                        )

                        noise = torch.randn(
                            (self.vlm.action_head.n_diffusion_samples, batch_size, 7), 
                            generator=self.vlm.action_head.rng,
                            device=pooled_hidden_states.device,
                        )

                        # Add noise to the action according to the schedule
                        scale = torch.sqrt(self.vlm.action_head.alpha_hats[time])
                        std = torch.sqrt(1 - self.vlm.action_head.alpha_hats[time])
                        noisy_actions = scale * gt_actions.unsqueeze(0) + std * noise

                        pred_eps = self.vlm.action_head(pooled_hidden_states, time=time, noisy_actions=noisy_actions)
                        l1_loss = torch.nn.functional.l1_loss(pred_eps, noise)
                        l2_loss = torch.nn.functional.mse_loss(pred_eps, noise)

                    else:
                        pred_actions = self.vlm.action_head(pooled_hidden_states)
                        l1_loss = torch.nn.functional.l1_loss(pred_actions, gt_actions)
                        l2_loss = torch.nn.functional.mse_loss(pred_actions, gt_actions)

                    # Calculate total loss =>> Weighed sum of language modeling 
                    # loss and action L2 loss 
                    lm_loss = output.loss
                    total_loss = lm_loss_weight * lm_loss + l2_loss_weight * l2_loss

                # Commit Loss =>> Backward!
                metrics.commit(total_loss=total_loss)
                total_loss.backward()

                # === Commit Metrics === #
                metrics.commit(
                    lm_loss=lm_loss,
                    l1_loss=l1_loss, 
                    l2_loss=l2_loss, 
                    update_step_time=True,
                )

                # Compute metrics per dataset --> only on rank_zero since we don't log them on other workers anyways
                if overwatch.is_rank_zero():
                    datasets = set(batch["dataset_names"])
                    if len(datasets) > 1:
                        for ds in datasets:
                            ds_mask = torch.tensor([elem == ds for elem in batch["dataset_names"]])
                            if using_diffusion_action_head:
                                ds_mask = ds_mask.expand((self.vlm.action_head.n_diffusion_samples, ) + ds_mask.shape)
                                l1_loss_ds = torch.nn.functional.l1_loss(pred_eps[ds_mask], noise[ds_mask])
                                l2_loss_ds = torch.nn.functional.mse_loss(pred_eps[ds_mask], noise[ds_mask])
                            else:
                                l1_loss_ds = torch.nn.functional.l1_loss(pred_actions[ds_mask], gt_actions[ds_mask])
                                l2_loss_ds = torch.nn.functional.mse_loss(pred_actions[ds_mask], gt_actions[ds_mask])

                            metrics.commit_for_dataset(
                                dataset_name=ds.decode(), 
                                l1_loss=l1_loss_ds, 
                                l2_loss=l2_loss_ds,
                            )

                # === Gradient Step === #

                # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality assumptions
                self.clip_grad_norm()

                # Optimizer & LR Scheduler Step
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                # Compute epoch value using number of completed gradient steps
                epoch = (metrics.global_step + 1) // (len(prismatic_vla_dataset) // self.global_batch_size)

                # Push Metrics
                metrics.commit(global_step=metrics.global_step + 1, epoch=epoch, lr=self.lr_scheduler.get_last_lr()[0])
                status = metrics.push()

                # Check for Save Interval or Max Steps & Save Checkpoint
                if (terminate := (self.max_steps is not None and metrics.global_step >= self.max_steps)) or (
                    (metrics.global_step % save_interval) == 0):
                    self.save_checkpoint(
                        metrics.run_dir, 
                        metrics.global_step, 
                        epoch, 
                        total_loss.item(),
                        only_trainable=not save_full_model, 
                        overwrite=overwrite, 
                    )
                    dist.barrier()

                    if terminate:
                        return

                # Update Progress Bar
                progress.update()
                progress.set_description(status)

            # Save checkpoint at the end (if `self.max_steps` is None)
            if (self.max_steps is None) and ((metrics.global_step % save_interval) != 0):
                self.save_checkpoint(
                    metrics.run_dir, 
                    metrics.global_step, 
                    epoch, 
                    total_loss.item(),
                    only_trainable=not save_full_model, 
                    overwrite=overwrite,
                )
                dist.barrier()

    # === Prismatic VLA Training with JSON dataset ===
    
    def run_prismatic_vla_json_training(
        self,
        prismatic_vla_json_dataset: Dataset,
        collator: PaddedCollatorForLanguageModelingAndActionPrediction,
        metrics: PrismaticVLAMetrics,
        stage: str = "lora-finetune",
        batch_construction_strategy: str = "split-modality",
        seed: int = 7,
        save_interval: int = 2500,
        overwrite: bool = False,
        save_full_model: bool = True,
        lm_loss_weight: float = 0.5,
        l2_loss_weight: float = 0.5,
    ) -> None:
        """
        Run the training loop for the given `dataset` and `collator`; log losses, results to `metrics`
        """
        assert isinstance(prismatic_vla_json_dataset, Dataset), "PrismaticVLA training expects a Dataset!"
        assert stage == "lora-finetune", f"Stage `{stage}` is not supported!"
        assert batch_construction_strategy == "split-modality"

        assert self.vlm.use_action_head, "PrismaticVLA should have an Action Head!"

        assert lm_loss_weight >= 0 and l2_loss_weight >= 0, "Weights for losses must be non-negative (>= 0)!"
        assert lm_loss_weight + l2_loss_weight == 1.0, "The weights for losses should sum to 1!"

        using_diffusion_action_head = self.vlm.action_head_configs["action_head_specifier"] == "diffusion"

        # Instantiate the split-modality sampler
        modality_lengths = prismatic_vla_json_dataset.get_modality_lengths()
        sampler = SplitModalitySampler(
            prismatic_vla_json_dataset,
            modality_lengths,
            global_batch_size=self.global_batch_size,
            num_replicas=overwatch.world_size(),
            rank=overwatch.rank(),
            seed=seed,
            drop_last=False,
        )

        # Create a DataLoader with the initialized sampler, per-device-bsz, and collator
        dataloader = DataLoader(
            prismatic_vla_json_dataset,
            batch_size=self.per_device_batch_size,
            sampler=sampler,
            collate_fn=collator,
            num_workers=2,
            worker_init_fn=self.worker_init_fn,
        )

        # Max Steps vs. Epochs Computation
        steps_per_epoch = len(dataloader) // self.grad_accumulation_steps
        if self.max_steps is not None and steps_per_epoch < self.max_steps:
            # Just set `epochs` to some large number --> we'll short-circuit based on steps anyway
            self.epochs = 100

        # === Train ===
        status = metrics.get_status()
        with tqdm(
            total=(
                (self.epochs * (len(dataloader) // self.grad_accumulation_steps))
                if self.max_steps is None
                else self.max_steps
            ),
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            for epoch in range(self.epochs):
                self.vlm.train()
                sampler.set_epoch(epoch)

                # Zero-Gradients (just in case)
                self.optimizer.zero_grad()

                # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
                for train_idx, batch in enumerate(dataloader):
                    # [Contract] self.vlm.forward() must automatically compute language modeling loss
                    with torch.autocast(
                        "cuda", dtype=self.mixed_precision_dtype, enabled=self.enable_mixed_precision_training,
                    ):
                        # [Contract] self.vlm.forward() must automatically compute language modeling loss
                        output: CausalLMOutputWithPast = self.vlm(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            pixel_values=batch["pixel_values"],
                            labels=batch["labels"],
                            output_hidden_states=True,
                        )

                        if self.vlm.use_layer_output_pooler:
                            all_hidden_layer_outputs = torch.stack(output.hidden_states, dim=2)
                            pooled_hidden_states = self.vlm.layer_output_pooler(all_hidden_layer_outputs)
                        else:
                            # We just consider the last layer output
                            pooled_hidden_states = output.hidden_states[-1]

                        gt_actions = batch["actions"].to(pooled_hidden_states.device)

                        if using_diffusion_action_head:
                            batch_size = pooled_hidden_states.shape[0]

                            # Generate random time and noise
                            time = torch.randint(
                                0,
                                self.vlm.action_head.diffusion_steps,
                                (self.vlm.action_head.n_diffusion_samples, batch_size, 1),
                                generator=self.vlm.action_head.rng,
                                device=pooled_hidden_states.device,
                            )

                            noise = torch.randn(
                                (self.vlm.action_head.n_diffusion_samples, batch_size, 7), 
                                generator=self.vlm.action_head.rng,
                                device=pooled_hidden_states.device,
                            )

                            # Add noise to the action according to the schedule
                            scale = torch.sqrt(self.vlm.action_head.alpha_hats[time])
                            std = torch.sqrt(1 - self.vlm.action_head.alpha_hats[time])
                            noisy_actions = scale * gt_actions.unsqueeze(0) + std * noise

                            pred_eps = self.vlm.action_head(pooled_hidden_states, time=time, noisy_actions=noisy_actions)
                            l1_loss = torch.nn.functional.l1_loss(pred_eps, noise)
                            l2_loss = torch.nn.functional.mse_loss(pred_eps, noise)

                        else:
                            pred_actions = self.vlm.action_head(pooled_hidden_states)
                            l1_loss = torch.nn.functional.l1_loss(pred_actions, gt_actions)
                            l2_loss = torch.nn.functional.mse_loss(pred_actions, gt_actions)

                        # Calculate total loss =>> Weighed sum of language modeling 
                        # loss and action L2 loss 
                        lm_loss = output.loss
                        total_loss = lm_loss_weight * lm_loss + l2_loss_weight * l2_loss

                    # Commit Loss (Prior to Gradient Accumulation Normalization)
                    metrics.commit(total_loss=total_loss)

                    # Normalize Loss to account for Gradient Accumulation --> Backward!
                    # [IMPORTANT] Technically speaking, doing gradient accumulation in this way is "incorrect"; this is
                    #             because in general, each batch has a *different number of masked out tokens* (because
                    #             we're instruct-tuning). Taking the mean over two unbalanced means != the right thing!
                    #
                    #             HOWEVER -- at least at the 7B scale, the "naive" approach is just as performant as
                    #             the "correct" implementation, without adding extra complexity.
                    #
                    # That being said =>> at the 13B scale, *no matter what we tried, ANY gradient accumulation is just
                    #   really bad for downstream performance. Initial investigation shows that BF16 accumulation
                    #   just really tanks in precision... and don't have a good/clean way to fix this. Would love for
                    #   someone to PR and fix this (and I'd greatly appreciate it!!!)
                    normalized_loss = total_loss / self.grad_accumulation_steps
                    normalized_loss.backward()

                    # === Commit Metrics === #
                    metrics.commit(
                        lm_loss=lm_loss,
                        l1_loss=l1_loss, 
                        l2_loss=l2_loss, 
                    )

                    # Compute metrics per dataset --> only on rank_zero since we don't log them on other workers anyways
                    if overwatch.is_rank_zero():
                        datasets = set(batch["dataset_names"])
                        if len(datasets) > 1:
                            for ds in datasets:
                                ds_mask = torch.tensor([elem == ds for elem in batch["dataset_names"]])
                                if using_diffusion_action_head:
                                    ds_mask = ds_mask.expand((self.vlm.action_head.n_diffusion_samples, ) + ds_mask.shape)
                                    l1_loss_ds = torch.nn.functional.l1_loss(pred_eps[ds_mask], noise[ds_mask])
                                    l2_loss_ds = torch.nn.functional.mse_loss(pred_eps[ds_mask], noise[ds_mask])
                                else:
                                    l1_loss_ds = torch.nn.functional.l1_loss(pred_actions[ds_mask], gt_actions[ds_mask])
                                    l2_loss_ds = torch.nn.functional.mse_loss(pred_actions[ds_mask], gt_actions[ds_mask])

                                metrics.commit_for_dataset(
                                    dataset_name=ds,
                                    l1_loss=l1_loss_ds, 
                                    l2_loss=l2_loss_ds,
                                )

                    # Step =>> Only if Done w/ Gradient Accumulation
                    if (train_idx + 1) % self.grad_accumulation_steps == 0:
                        metrics.commit(update_step_time=True)

                        # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality-assumptions
                        self.clip_grad_norm()

                        # Optimizer & LR Scheduler Step
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()

                        # Push Metrics
                        metrics.commit(global_step=metrics.global_step + 1, lr=self.lr_scheduler.get_last_lr()[0])
                        status = metrics.push()

                        # Check for Save Interval or Max Steps & Save Checkpoint
                        if (terminate := (self.max_steps is not None and metrics.global_step >= self.max_steps)) or (
                            (metrics.global_step % save_interval) == 0):
                            self.save_checkpoint(
                                metrics.run_dir,
                                metrics.global_step, 
                                epoch, 
                                total_loss.item(),
                                only_trainable=not save_full_model, 
                                overwrite=overwrite)
                            dist.barrier()

                            if terminate:
                                return

                        # Update Progress Bar
                        progress.update()
                        progress.set_description(status)

            # Save checkpoint at the end (if `self.max_steps` is None)
            if (self.max_steps is None) and ((metrics.global_step % save_interval) != 0):
                self.save_checkpoint(
                    metrics.run_dir, 
                    metrics.global_step, 
                    epoch, 
                    total_loss.item(),
                    only_trainable=not save_full_model, 
                    overwrite=overwrite)
                dist.barrier()