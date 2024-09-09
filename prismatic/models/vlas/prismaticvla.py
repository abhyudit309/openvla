"""
prismaticvla.py

PyTorch Module defining PrismaticVLA as a lightweight wrapper around a PrismaticVLM.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from prismatic.models.backbones.llm import LLMBackbone
from prismatic.models.backbones.vision import VisionBackbone
from prismatic.models.vlms.prismatic import PrismaticVLM
from prismatic.overwatch import initialize_overwatch

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


class PrismaticVLA(PrismaticVLM):
    def __init__(
        self,
        model_id: str,
        vision_backbone: VisionBackbone,
        llm_backbone: LLMBackbone,
        enable_mixed_precision_training: bool = True,
        arch_specifier: str = "gelu-mlp",
        use_action_head: bool = True,
        action_head_configs: Optional[Dict] = None,
        use_action_head_for_inference: bool = False,
        norm_stats: Optional[Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]]] = None,
    ) -> None:
        super().__init__(
            model_id=model_id,
            vision_backbone=vision_backbone,
            llm_backbone=llm_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
            arch_specifier=arch_specifier,
            use_action_head=use_action_head,
            action_head_configs=action_head_configs,
            use_action_head_for_inference=use_action_head_for_inference,
        )
        
        self.norm_stats = norm_stats

    @torch.inference_mode()
    def generate_response_and_predict_action(
        self, 
        image: Image, 
        prompt_text: str, 
        unnorm_key: Optional[str] = None,
        return_action: bool = True,
        **kwargs: str
    ) -> Tuple[str, Optional[np.ndarray]]:
        """
        Core function for PrismaticVLA inference; maps input image and task instruction to continuous action.

        @param image: PIL Image as [height, width, 3]
        @param prompt_text: Text string
        @param unnorm_key: Optional dataset name for retrieving un-normalizing statistics; if None, checks that model
                           was trained only on a single dataset, and retrieves those statistics.
        @param return_action: Whether to return action along with generated text.

        @return Unnormalized (continuous) action vector --> end-effector deltas.
        """
        # For now, only support generation with a batch size of 1 for simplicity
        image_transform, tokenizer = self.vision_backbone.image_transform, self.llm_backbone.tokenizer

        # Prepare Inputs
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.device)
        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.device) for k, v in pixel_values.items()}
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")
        
        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.llm_backbone.half_precision_dtype
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.enable_mixed_precision_training):
            output = super(PrismaticVLM, self).generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict_in_generate=True,
                **kwargs
            )

        # => Note: 'forward'/'generate' does NOT generate actions for us since we are doing inference, 
        # so we do it here explicitly. The last layer output should only have embeddings for the prompt 
        # text (time step = 0), and NOT the response!
        actions = None
        if return_action:
            llm_last_layer_output = output.hidden_states[0][-1]
            actions = self.action_head(llm_last_layer_output)[0].to(dtype=torch.float32).cpu().numpy()

            # TODO: Unnormalizing actions (if applicable!)

        # Get the generated response
        generated_ids = output.sequences
        generated_text = tokenizer.decode(generated_ids[0, input_ids.shape[1] :], skip_special_tokens=True).strip()

        return generated_text, actions

    @staticmethod
    def _check_unnorm_key(norm_stats: Dict, unnorm_key: str) -> str:
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, please pass a `unnorm_key` from the following "
                f"options to choose the statistics used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        # Error Handling
        assert (
            unnorm_key in norm_stats
        ), f"The `unnorm_key` you chose is not in the set of available statistics; choose from: {norm_stats.keys()}"

        return unnorm_key

    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)

        return len(self.norm_stats[unnorm_key]["action"]["q01"])

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict:
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)

        return self.norm_stats[unnorm_key]["action"]
