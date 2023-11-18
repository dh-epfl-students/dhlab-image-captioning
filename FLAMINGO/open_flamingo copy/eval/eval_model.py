import abc
import argparse
from typing import List
from torch.nn.parallel import DistributedDataParallel as DDP
from PIL import Image


class BaseEvalModel(abc.ABC):
    """Base class encapsulating functionality needed to evaluate a model."""

    def __init__(self, args: List[str]):
        """Initialize model.

        Args:
            args: arguments to model. These should be parsed, or if the model
                has no applicable arguments, an error should be thrown if `args`
                is non-empty.
        """

    def init_distributed(self):
        """Wrap model as DDP."""
        self.model = DDP(self.model, device_ids=[self.device])

    def set_device(self, device):
        """Set device for model."""
        self.device = device
        self.model = self.model.to(device)

    def get_outputs(
        self,
        batch_text: List[str],
        batch_images: List[List[Image.Image]],
        min_generation_length: int,
        max_generation_length: int,
        num_beams: int,
        length_penalty: float,
    ) -> List[str]:
        """Get outputs for a batch of images and text.

        Args:
            batch_text: list of text strings, with the text "<image>" in place
                of any images to be included.
            batch_images: images to provide to model. Should be a list of lists,
              where each list contains the images for a single example.
            max_generation_length: maximum length of the generated caption.
                Defaults to 10.
            num_beams: number of beams to use for beam search. Defaults to 3.
            length_penalty: length penalty for beam search. Defaults to -2.0.

        Returns:
            List of decoded output strings.
        """

    def vqa_prompt(self, question, answer=None) -> str:
        """Get the prompt to use for VQA evaluation. If the answer is not provided, it should be left blank to be generated by the model.

        Returns:
            The prompt to use for VQA.
        """

    def caption_prompt(self, caption=None) -> str:
        """Get the prompt to use for caption evaluation. If the caption is not provided, it should be left blank to be generated by the model.

        Returns:
            The prompt to use for captioning.
        """

    def get_rank_classifications(
        self,
        batch_text: List[str],
        batch_images: List[List[Image.Image]],
        all_class_names: List[str],
        use_cache: bool,
        normalize_length: bool,
    ):
        """
        Returns a (B, |all_class_names|) tensor containing the logprobs for each class name.
        Args:
            batch_text: list of text strings, with the text "<image>" in place
                of any images to be included.
            batch_images: images to provide to model. Should be a list of lists,
                where each list contains the images for a single example.
            all_class_names: list of all class names.
            use_cache: whether to cache the context to speed up evaluations.
            normalize_length: whether to normalize logprobs by the length of the
                class name
        Returns:
            (B, |all_class_names|) tensor containing the logprobs for each class name.
        """
