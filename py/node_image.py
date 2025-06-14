from comfy_extras.nodes_mask import ImageCompositeMasked
import torch
import torch.nn.functional as F
class ImageCompositeWatermark(ImageCompositeMasked):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "destination": ("IMAGE",),
                "watermark": ("IMAGE",),
                "position": (["top_right", "top_center", "top_left", "bottom_right", "bottom_center", "bottom_left"], {"default": "bottom_right"}),
                "resize_ratio": ("FLOAT", {"default": 1, "min": 0, "max": 10, "step": 0.05}),
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
        
        dest_h, dest_w = destination.shape[1:3]
        water_h, water_w = watermark.shape[1:3]

        scale = 1
        if water_h > dest_h or water_w > dest_w:
            # 计算需要的缩放比例
            scale_h = dest_h / water_h
            scale_w = dest_w / water_w
            scale = min(scale_h, scale_w) 


        if resize_ratio != 1 or scale != 1:
            watermark = torch.nn.functional.interpolate(
                watermark.movedim(-1, 1), scale_factor=resize_ratio * scale, mode="bicubic", antialias=True).movedim(1, -1).clamp(0.0, 1.0)
            if mask is not None:
                mask = torch.nn.functional.interpolate(mask.unsqueeze(
                    0), scale_factor=resize_ratio * scale, mode="bicubic", antialias=True).squeeze(0).clamp(0.0, 1.0)

        water_h, water_w = watermark.shape[1:3]
        # 计算y坐标
        if position.startswith("top"):
            y = margin
        else:  # bottom positions
            y = dest_h - water_h - margin
        
        x = 0
        # 根据position计算x坐标
        if position.endswith("left"):
            x = margin
        elif position.endswith("center"):
            x = (dest_w - water_w) // 2
        elif position.endswith("right"):
            x = dest_w - water_w - margin
            
        if invert_mask and mask is not None:
            mask = 1.0 - mask

            
        return self.composite(destination, watermark, x, y, False, mask)
    
class ImageTransition:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "first_image": ("IMAGE",),
                "last_image": ("IMAGE",),
                "frames": ("INT", {"default": 24, "min": 2, "max": 120, "step": 1}),
                "transition_type": (["uniform", "smooth"], {"default": "uniform"}),
                "smooth_effect": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_transition"
    CATEGORY = "utils/image"

    def generate_transition(self, first_image, last_image, frames, transition_type="uniform", smooth_effect=1.0):
        # 生成插值权重
        if transition_type == "uniform":
            weights = torch.linspace(0, 1, frames)
        else:  # sigmoid
            x = torch.linspace(-20, 20, frames)
            weights = torch.sigmoid(x / smooth_effect)
        
        # 创建输出张量列表
        output_frames = []
        
        # 生成过渡帧
        for w in weights:
            # 使用权重进行插值
            transition_frame = first_image * (1 - w) + last_image * w
            output_frames.append(transition_frame)
        
        # 将所有帧拼接在一起
        result = torch.cat(output_frames, dim=0)
        
        return (result,)

class ImageMaskColorAverage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            }
        }
    
    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("COLOR_DEC", "COLOR_HEX")
    FUNCTION = "calculate_average_color"
    CATEGORY = "utils/image"

    def calculate_average_color(self, image, mask):
        # 确保mask是二维的
        if len(mask.shape) > 2:
            mask = mask.squeeze()
        
        # 将mask扩展为与图像相同的通道数
        expanded_mask = mask.unsqueeze(-1).expand(-1, -1, 3)
        
        # 计算mask区域的像素总数
        pixel_count = torch.sum(mask)
        
        if pixel_count == 0:
            # 如果mask中没有选中区域，返回黑色
            return (0, "#000000")
        
        # 计算mask区域的颜色总和
        masked_image = image * expanded_mask.unsqueeze(0)
        color_sum = torch.sum(masked_image, dim=[0, 1, 2])
        
        # 计算平均颜色
        avg_color = color_sum / pixel_count
        
        # 转换为0-255范围的整数
        r = int(avg_color[0].item() * 255)
        g = int(avg_color[1].item() * 255)
        b = int(avg_color[2].item() * 255)
        
        # 生成十六进制颜色代码
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        
        # 计算十进制颜色值 (R*65536 + G*256 + B)
        dec_color = r * 65536 + g * 256 + b
        
        return (dec_color, hex_color)


class ImagesConcanateToGrid:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image1": ("IMAGE",),
            "direction": (
                ['right',
                 'down',
                 ],
                {
                    "default": 'right'
                }),
            "dimension_number": ("INT", {"default": 2, "min": 1, "max": 20, "step": 1}),
        }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "concanate"
    CATEGORY = "utils/image"

    def concanate(self, image1, direction='right', dimension_number=2):
        # 检查图像维度，如果不是4维直接返回
        if len(image1.shape) != 4:
            return (image1,)
        
        batch_size = image1.shape[0]
        
        # 如果批次大小为1，直接返回
        if batch_size == 1:
            return (image1,)
        
        # 将批次图像分离为单独的图像列表
        images = [image1[i:i+1] for i in range(batch_size)]
        
        # 根据方向计算行数和列数
        if direction == 'right':
            cols = dimension_number
            rows = (batch_size + cols - 1) // cols  # 向上取整
        else:  # direction == 'down'
            rows = dimension_number
            cols = (batch_size + rows - 1) // rows  # 向上取整
        
        # 创建网格来存放图像
        grid_rows = []
        
        for row in range(rows):
            row_images = []
            for col in range(cols):
                idx = row * cols + col if direction == 'right' else col * rows + row
                if idx < len(images):
                    row_images.append(images[idx])
                else:
                    # 如果没有足够的图像，用黑色图像填充
                    black_image = torch.zeros_like(images[0])
                    row_images.append(black_image)
            
            # 水平拼接当前行的图像
            if row_images:
                row_concat = torch.cat(row_images, dim=2)  # 在宽度维度拼接
                grid_rows.append(row_concat)
        
        # 垂直拼接所有行
        if grid_rows:
            result = torch.cat(grid_rows, dim=1)  # 在高度维度拼接
        else:
            result = image1
        
        return (result,)

class NeedImageSizeAndCount:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",)}}
    
    RETURN_TYPES = ("INT", "INT", "INT")
    RETURN_NAMES = ("width", "height", "count")
    FUNCTION = "get_image_size_and_count"
    CATEGORY = "utils/image"

    def get_image_size_and_count(self, image):
        return (image.shape[2], image.shape[1], image.shape[0])

NODE_CLASS_MAPPINGS = {
    #image
    "ImageCompositeWatermark": ImageCompositeWatermark,
    "ImageTransition": ImageTransition,
    "ImageMaskColorAverage": ImageMaskColorAverage,
    "ImagesConcanateToGrid": ImagesConcanateToGrid,
    "NeedImageSizeAndCount": NeedImageSizeAndCount,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Image
    "ImageCompositeWatermark": "Image Composite Watermark",
    "ImageTransition": "Image Transition",
    "ImageMaskColorAverage": "Image Mask Color Average",
    "ImagesConcanateToGrid": "Images Concanate To Grid",
    "NeedImageSizeAndCount": "get Image Size And Count by utils",
}
