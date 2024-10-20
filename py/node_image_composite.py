import torch
import logging
logger = logging.getLogger(__file__)
MAX_RESOLUTION=16384

def composite(destination, source, x, y, mask=None, resize_source=False, resize_mode='bilinear'):
    device = destination.device
    batch_size, _, dest_height, dest_width = destination.shape
    if not resize_source:
        _, _, source_height, source_width = source.shape
    else:
        source_height, source_width = dest_height, dest_width
    
    if x > dest_width or y > dest_height:
        return destination
    
    left = x
    top =  y

    visible_width = min(dest_width - left, source_width)
    visible_height = min(dest_height - top, source_height)

    if resize_source:
        resize_height, resize_width = dest_height, dest_width
    else:
        resize_height, resize_width = source_height, source_width

    for i in range(batch_size):
        source_slice = source[i:i+1].to(device)
        
        if resize_source:
            source_slice = torch.nn.functional.interpolate(source_slice, size=(resize_height, resize_width), mode=resize_mode)

        if mask is None:
            mask_slice = torch.ones_like(source_slice)
        else:
            mask_slice = mask[i:i+1].to(device)
            mask_slice = torch.nn.functional.interpolate(mask_slice.reshape((-1, 1, mask_slice.shape[-2], mask_slice.shape[-1])), 
                                                         size=(resize_height, resize_width), mode=resize_mode)

        mask_slice = mask_slice[:, :, :visible_height, :visible_width]

        dest_portion = destination[i:i+1, :, top:top+visible_height, left:left+visible_width]
        source_portion = source_slice[:, :, :visible_height, :visible_width]

        dest_portion.mul_(1 - mask_slice)
        dest_portion.add_(source_portion * mask_slice)

        # Free up memory
        del source_slice, mask_slice, dest_portion, source_portion
        torch.cuda.empty_cache()

    return destination


class ImageCompositeMaskedOneByOne:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "destination": ("IMAGE",),
                "source": ("IMAGE",),
                "x": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "y": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "resize_source": ("BOOLEAN", {"default": False}),
                "resize_source_mode":(["nearest", "bilinear", "bicubic", "area", "nearest-exact"],)
            },
            "optional": {
                "mask": ("MASK",),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "composite"

    CATEGORY = "utils/image"

    def composite(self, destination, source, x, y, resize_source, resize_source_mode= "bilinear", mask = None):
        destination = destination.clone().movedim(-1, 1)
        output = composite(destination, source.movedim(-1, 1), x, y, mask, resize_source, resize_source_mode).movedim(1, -1)
        return (output,)
    

NODE_CLASS_MAPPINGS = {
    #image
    "ImageCompositeMaskedOneByOne": ImageCompositeMaskedOneByOne,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Image
    "ImageCompositeMaskedOneByOne": "image composite masked one bye one",

}
