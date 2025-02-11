from .utils import *
# this node is original from ComfyUI-LayerStyle, modified the logic of crop to specific size and others
import torch
import logging
from PIL import Image

logger = logging.getLogger(__name__)



class CropByMaskToSpecificSize:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "image": ("IMAGE", ),  #
                "mask": ("MASK",),
                "invert_mask": ("BOOLEAN", {"default": False}),  # 反转mask#
                "top_reserve": ("FLOAT", {"default": 0.1, "min": 0, "max": 1, "step": 0.01}),
                "bottom_reserve": ("FLOAT", {"default": 0.1, "min": 0, "max": 1, "step": 0.01}),
                "left_reserve": ("FLOAT", {"default": 0.1, "min": 0, "max": 1, "step": 0.01}),
                "right_reserve": ("FLOAT", {"default": 0.1, "min": 0, "max": 1, "step": 0.01}),
                "width": ("INT", {"default": 1024, "min": 200, "max": 4096, "step": 2}),
                "height": ("INT", {"default": 1024, "min": 200, "max": 4096, "step": 2}),
                "width_padding_position":(["left","center","right"],{"default":"center",}),
                "height_padding_position":(["top","center","bottom"],{"default":"center"}),
            },
            "optional": {
                "crop_box": ("BOX",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BOX", "IMAGE",)
    RETURN_NAMES = ("croped_image", "croped_mask", "crop_box", "box_preview")
    FUNCTION = 'crop_by_mask'
    CATEGORY = 'utils/mask'

    def crop_by_mask(self, image, mask, invert_mask,
                     top_reserve, bottom_reserve,
                     left_reserve, right_reserve,
                     width, height, 
                     width_padding_position, height_padding_position,
                     crop_box=None
                     ):

        ret_images = []
        ret_masks = []
        l_images = []
        l_masks = []

        for l in image:
            l_images.append(torch.unsqueeze(l, 0))
        if mask.dim() == 2:
            mask = torch.unsqueeze(mask, 0)
        # 如果有多张mask输入，使用第一张
        if mask.shape[0] > 1:
            logger.warn(f"Warning: Multiple mask inputs, using the first.")
            mask = torch.unsqueeze(mask[0], 0)
        if invert_mask:
            mask = 1 - mask
        l_masks.append(tensor2pil(torch.unsqueeze(mask, 0)).convert('L'))

        _mask = mask2image(mask)
        preview_image = tensor2pil(mask).convert('RGBA')
        if crop_box is None:
            x = 0
            y = 0
            (x, y, w, h) = mask_area(_mask)
            left_reserve = left_reserve * w
            top_reserve = top_reserve * h
            right_reserve = right_reserve * w
            bottom_reserve = bottom_reserve * h

            canvas_width, canvas_height = tensor2pil(torch.unsqueeze(image[0], 0)).convert('RGBA').size
            x1 = x - left_reserve if x - left_reserve > 0 else 0
            y1 = y - top_reserve if y - top_reserve > 0 else 0
            x2 = x + w + right_reserve if x + w + right_reserve < canvas_width else canvas_width
            y2 = y + h + bottom_reserve if y + h + bottom_reserve < canvas_height else canvas_height

            # 计算当前裁剪框的宽高
            current_width = x2 - x1
            current_height = y2 - y1
            
            # 计算目标宽高比和当前宽高比
            target_ratio = width / height
            current_ratio = current_width / current_height
            
            # 根据比例调整裁剪框
            if current_ratio < target_ratio:
                # 需要增加宽度
                needed_width = current_height * target_ratio
                width_increase = needed_width - current_width
                x1 = max(0, x1 - width_increase / 2)
                x2 = min(canvas_width, x2 + width_increase / 2)
            else:
                # 需要增加高度
                needed_height = current_width / target_ratio
                height_increase = needed_height - current_height
                y1 = max(0, y1 - height_increase / 2)
                y2 = min(canvas_height, y2 + height_increase / 2)

            logger.info(f"Box detected. x={x1},y={y1},width={width},height={height}")
            crop_box = (int(x1), int(y1), int(x2), int(y2))
            preview_image = draw_rect(preview_image, x, y, w, h, line_color="#F00000",
                                      line_width=(w + h) // 100)
        preview_image = draw_rect(preview_image, crop_box[0], crop_box[1],
                                  crop_box[2] - crop_box[0], crop_box[3] - crop_box[1],
                                  line_color="#00F000",
                                  line_width=(crop_box[2] - crop_box[0] + crop_box[3] - crop_box[1]) // 200)
        for i in range(len(l_images)):
            _canvas = tensor2pil(l_images[i]).convert('RGBA')
            _mask = l_masks[0]
            
            # 裁剪图像和遮罩
            cropped_image = _canvas.crop(crop_box)
            cropped_mask = _mask.crop(crop_box)
            
            # 计算缩放比例
            crop_width = crop_box[2] - crop_box[0]
            crop_height = crop_box[3] - crop_box[1]
            scale_w = width / crop_width
            scale_h = height / crop_height
            scale = min(scale_w, scale_h)
            
            # 按比例缩放
            new_w = int(crop_width * scale)
            new_h = int(crop_height * scale)
            resized_image = cropped_image.resize((new_w, new_h), Image.LANCZOS)
            resized_mask = cropped_mask.resize((new_w, new_h), Image.LANCZOS)
            
            # 创建目标尺寸的灰色背景
            final_image = Image.new('RGBA', (width, height), (128, 128, 128, 255))
            final_mask = Image.new('L', (width, height), 0)
            
            # 计算粘贴位置（居中）
            if width_padding_position == "center":
                paste_x = (width - new_w) // 2
            elif width_padding_position == "left":
                paste_x = width - new_w                
            elif width_padding_position == "right":
                paste_x = 0
              
            
            if height_padding_position == "center":
                paste_y = (height - new_h) // 2
            elif height_padding_position == "top":
                paste_y = height - new_h                
            elif height_padding_position == "bottom":
                paste_y = 0
            

            # 粘贴调整后的图像和遮罩
            final_image.paste(resized_image, (paste_x, paste_y))
            final_mask.paste(resized_mask, (paste_x, paste_y))
            
            ret_images.append(pil2tensor(final_image))
            ret_masks.append(image2mask(final_mask))

        logger.info(f"Processed {len(ret_images)} image(s).")
        return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0), list(crop_box), pil2tensor(preview_image),)


NODE_CLASS_MAPPINGS = {
   "CropByMaskToSpecificSize": CropByMaskToSpecificSize
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: CropByMask To Specific Size": "LayerUtility: CropByMask To Specific Size"
}