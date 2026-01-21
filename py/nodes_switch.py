
import torch
import math
import comfy.utils
import node_helpers
from nodes import VAEEncode
from comfy_extras.nodes_post_processing import ImageScaleToTotalPixels
from comfy_extras.nodes_edit_model import ReferenceLatent

class ImageScaleToTotalPixelsSwitch(ImageScaleToTotalPixels):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_method": (ImageScaleToTotalPixels.upscale_methods,),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 16.0, "step": 0.01}),
                "resolution_steps": ("INT", {"default": 1, "min": 1, "max": 256}),
            },
            "optional": {
                "enabled": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute_switch"
    CATEGORY = "utils/switch"

    def execute_switch(self, image, upscale_method, megapixels, resolution_steps, enabled=True):
        if not enabled:
            return (image,)
        
        # Logic copied from ImageScaleToTotalPixels.execute
        samples = image.movedim(-1,1)
        total = megapixels * 1024 * 1024

        scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
        width = round(samples.shape[3] * scale_by / resolution_steps) * resolution_steps
        height = round(samples.shape[2] * scale_by / resolution_steps) * resolution_steps

        s = comfy.utils.common_upscale(samples, int(width), int(height), upscale_method, "disabled")
        s = s.movedim(1,-1)
        return (s,)

class ReferenceLatentSwitch(ReferenceLatent):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
            },
            "optional": {
                "latent": ("LATENT",),
                "enabled": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "execute_switch"
    CATEGORY = "utils/switch"

    def execute_switch(self, conditioning, latent=None, enabled=True):
        if not enabled:
            return (conditioning,)
        
        # Logic copied from ReferenceLatent.execute
        if latent is not None:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": [latent["samples"]]}, append=True)
        return (conditioning,)

class VAEEncoderSwitch(VAEEncode):
    @classmethod
    def INPUT_TYPES(s):
        base_inputs = super().INPUT_TYPES()
        # Add enabled to optional if exists, or create optional
        if "optional" not in base_inputs:
            base_inputs["optional"] = {}
        base_inputs["optional"]["enabled"] = ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"})
        return base_inputs

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode_switch"
    CATEGORY = "utils/switch"

    def encode_switch(self, pixels, vae, enabled=True):
        if not enabled:
            return (None,) # Bypass not fully possible for Image->Latent, return None
        
        return super().encode(vae, pixels)

NODE_CLASS_MAPPINGS = {
    "ImageScaleToTotalPixelsSwitch": ImageScaleToTotalPixelsSwitch,
    "ReferenceLatentSwitch": ReferenceLatentSwitch,
    "VAEEncoderSwitch": VAEEncoderSwitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageScaleToTotalPixelsSwitch": "Image Scale To Total Pixels (Switch)",
    "ReferenceLatentSwitch": "Reference Latent (Switch)",
    "VAEEncoderSwitch": "VAE Encoder (Switch)",
}
