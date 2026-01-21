import math
import comfy.utils
import node_helpers
from nodes import VAEEncode
from comfy_api.latest import io
from comfy_extras.nodes_post_processing import ImageScaleToTotalPixels
from comfy_extras.nodes_edit_model import ReferenceLatent


class ImageScaleToTotalPixelsSwitch(ImageScaleToTotalPixels):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ImageScaleToTotalPixelsSwitch",
            category="utils/switch",
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("upscale_method", options=cls.upscale_methods),
                io.Float.Input("megapixels", default=1.0, min=0.01, max=16.0, step=0.01),
                io.Int.Input("resolution_steps", default=1, min=1, max=256),
                io.Boolean.Input("enabled", default=True),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    def execute(cls, image, upscale_method, megapixels, resolution_steps, enabled=True) -> io.NodeOutput:
        if not enabled:
            return io.NodeOutput(image)
        return super().execute(image, upscale_method, megapixels, resolution_steps)


class ReferenceLatentSwitch(ReferenceLatent):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ReferenceLatentSwitch",
            category="utils/switch",
            description="ReferenceLatent with switch. When disabled, returns conditioning directly.",
            inputs=[
                io.Conditioning.Input("conditioning"),
                io.Latent.Input("latent", optional=True),
                io.Boolean.Input("enabled", default=True),
            ],
            outputs=[
                io.Conditioning.Output(),
            ]
        )

    @classmethod
    def execute(cls, conditioning, latent=None, enabled=True) -> io.NodeOutput:
        if not enabled:
            return io.NodeOutput(conditioning)
        return super().execute(conditioning, latent)


class VAEEncoderSwitch(VAEEncode):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pixels": ("IMAGE",),
                "vae": ("VAE",),
            },
            "optional": {
                "enabled": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode_switch"
    CATEGORY = "utils/switch"

    def encode_switch(self, pixels, vae, enabled=True):
        if not enabled:
            return (None,)
        return self.encode(vae, pixels)


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
