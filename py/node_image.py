from comfy_extras.nodes_mask import ImageCompositeMasked
import torch

class ImageCompositeWatermark(ImageCompositeMasked):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "destination": ("IMAGE",),
                "watermark": ("IMAGE",),
                "position": (["bottom_right", "bottom_center", "bottom_left"], {"default": "bottom_right"}),
                "resize_ratio": ("FLOAT", {"default": 1, "min": 0, "max": 10, "step": 0.1}),
                "margin": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
            },
            "optional": {
                "mask": ("MASK",),
                "enabled": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "invert_mask": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "composite_watermark"
    CATEGORY = "utils/image"

    def composite_watermark(self, destination, watermark, position, resize_ratio, margin, mask=None, enabled=True, invert_mask=False):
        if not enabled:
            return (destination,)
        
        if resize_ratio != 1:
            watermark = torch.nn.functional.interpolate(
                watermark.movedim(-1, 1), scale_factor=resize_ratio, mode="bicubic", antialias=True).movedim(1, -1).clamp(0.0, 1.0)
            if mask is not None:
                mask = torch.nn.functional.interpolate(mask.unsqueeze(
                    0), scale_factor=resize_ratio, mode="bicubic", antialias=True).squeeze(0).clamp(0.0, 1.0)


        # 计算水印的位置
        dest_h, dest_w = destination.shape[1:3]
        water_h, water_w = watermark.shape[1:3]
        
        # 计算y坐标 - 总是在底部
        y = dest_h - water_h - margin
        
        x = 0
        # 根据position计算x坐标
        if position == "bottom_left":
            x = margin
        elif position == "bottom_center":
            x = (dest_w - water_w) // 2
        elif position == "bottom_right":
            x = dest_w - water_w - margin
            
        if invert_mask and mask is not None:
            mask = 1.0 - mask

            
        return self.composite(destination, watermark, x, y, False, mask)
    

NODE_CLASS_MAPPINGS = {

    #image
    "ImageCompositeWatermark": ImageCompositeWatermark,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Image
    "ImageCompositeWatermark": "Image Composite Watermark",
}
