"""
materialize.py

Factory class for initializing Open-X RLDS-backed datasets, given specified data mixture parameters; provides and
exports individual functions for clear control flow.
"""

from pathlib import Path
from typing import Tuple, Type, Optional

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from prismatic.conf.datasets import DatasetConfig
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.util.data_utils import (
    PaddedCollatorForActionPrediction, 
    PaddedCollatorForLanguageModelingAndActionPrediction
)
from prismatic.vla.datasets import (
    EpisodicRLDSDataset, 
    RLDSBatchTransform, 
    RLDSDataset,
    RLDSQnABatchTransform,
    RLDSQnADataset,
    RLDSQnAJSONDataset,
)


def get_vla_dataset_and_collator(
    data_root_dir: Path,
    data_mix: str,
    image_transform: ImageTransform,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    default_image_resolution: Tuple[int, int, int],
    padding_side: str = "right",
    predict_stop_token: bool = True,
    shuffle_buffer_size: int = 100_000,
    train: bool = True,
    episodic: bool = False,
    image_aug: bool = False,
) -> Tuple[Dataset, ActionTokenizer, PaddedCollatorForActionPrediction]:
    """Initialize RLDS Dataset (wraps TFDS), ActionTokenizer, and initialize transform/collation functions."""
    action_tokenizer = ActionTokenizer(tokenizer)
    batch_transform = RLDSBatchTransform(
        action_tokenizer, tokenizer, image_transform, prompt_builder_fn, predict_stop_token=predict_stop_token
    )
    collator = PaddedCollatorForActionPrediction(
        tokenizer.model_max_length, tokenizer.pad_token_id, padding_side=padding_side
    )

    # Build RLDS Iterable Dataset
    cls = RLDSDataset if not episodic else EpisodicRLDSDataset
    dataset = cls(
        data_root_dir,
        data_mix,
        batch_transform,
        resize_resolution=default_image_resolution[1:],
        shuffle_buffer_size=shuffle_buffer_size,
        train=train,
        image_aug=image_aug,
    )

    return dataset, action_tokenizer, collator


def get_prismatic_vla_dataset_and_collator(
    data_root_dir: Path,
    data_mix: str,
    image_transform: ImageTransform,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    default_image_resolution: Tuple[int, int, int],
    padding_side: str = "right",
    shuffle_buffer_size: int = 100_000,
    train: bool = True,
    image_aug: bool = False,
) -> Tuple[Dataset, PaddedCollatorForLanguageModelingAndActionPrediction]:
    """Initialize RLDS QnA Dataset (wraps TFDS) and initialize transform/collation functions."""
    batch_transform = RLDSQnABatchTransform(tokenizer, image_transform, prompt_builder_fn)
    collator = PaddedCollatorForLanguageModelingAndActionPrediction(
        tokenizer.model_max_length, tokenizer.pad_token_id, padding_side=padding_side
    )

    # Build RLDS QnA Iterable Dataset
    cls = RLDSQnADataset
    dataset = cls(
        data_root_dir,
        data_mix,
        batch_transform,
        resize_resolution=default_image_resolution[1:],
        shuffle_buffer_size=shuffle_buffer_size,
        train=train,
        image_aug=image_aug,
    )

    return dataset, collator


def get_prismatic_vla_json_dataset_and_collator(
    stage: str,
    dataset_cfg: DatasetConfig,
    image_transform: ImageTransform,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    padding_side: str = "right",
    train_val_split_3qna: Optional[float] = None,
    train_val_split_multi_qna: Optional[float] = None,
    randomize_qnas: bool = False,
) -> Tuple[Dataset, PaddedCollatorForLanguageModelingAndActionPrediction]:
    assert stage == "lora-finetune", f"Stage `{stage}` is not supported!"

    dataset_root_dir = dataset_cfg.dataset_root_dir
    collator = PaddedCollatorForLanguageModelingAndActionPrediction(
        tokenizer.model_max_length, tokenizer.pad_token_id, padding_side=padding_side
    )
    
    annotation_json, image_dir = dataset_cfg.finetune_stage_components
    dataset = RLDSQnAJSONDataset(
        dataset_root_dir / annotation_json,
        dataset_root_dir / image_dir,
        image_transform,
        tokenizer,
        prompt_builder_fn=prompt_builder_fn,
        train_val_split_3qna=train_val_split_3qna,
        train_val_split_multi_qna=train_val_split_multi_qna,
        randomize_qnas=randomize_qnas,
    )
    return dataset, collator