import logging
import folder_paths
from nodes import LoadImage
from comfy.cli_args import args
import comfy.model_management
import comfy.clip_vision
import comfy.controlnet
import comfy.utils
import comfy.sd
import comfy.sample
import comfy.samplers
import comfy.diffusers_load
import torch

import os
import sys


sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "comfy"))


logger = logging.getLogger(__file__)


class LoadImageWithSwitch(LoadImage):
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(
            os.path.join(input_dir, f))]
        return {"required":
                {"image": (sorted(files), {"image_upload": True})},
                "optional": {
                    "enabled": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                }
                }

    CATEGORY = "image"

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image_with_switch"

    def load_image_with_switch(self, image, enabled):
        logger.debug("start load image")
        if not enabled:
            return None, None
        return self.load_image(image)


class ImageBatchOneOrMore:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image1": ("IMAGE",)},
                "optional": {"image2": ("IMAGE",), "image3": ("IMAGE",), "image4": ("IMAGE",), "image5": ("IMAGE",), "image6": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "batch"

    CATEGORY = "image"

    def batch(self, image1, image2=None, image3=None, image4=None, image5=None, image6=None):
        images = [image1]
        for other_image in [image2, image3, image4, image5, image6]:
            if other_image is not None:
                try:
                    if image1.shape[1:] != image2.shape[1:]:
                        other_image = comfy.utils.common_upscale(
                            other_image.movedim(-1, 1), image1.shape[2], image1.shape[1], "bilinear", "center").movedim(1, -1)
                    images.append(other_image)
                except Exception as e:
                    logger.exception(e)
        s = torch.cat(images, dim=0)
        return (s,)


NODE_CLASS_MAPPINGS = {
    "LoadImageWithSwitch": LoadImageWithSwitch,
    "ImageBatchOneOrMore": ImageBatchOneOrMore,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageWithSwitch": "Load Image with switch",
    "ImageBatchOneOrMore": "Batch Images One or More",
}
