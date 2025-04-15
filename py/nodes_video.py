import torch
import numpy as np
import logging
logger = logging.getLogger(__name__)

class FrameAdjuster:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{
            "images": ("IMAGE",),
            "duration": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 60.0, "step": 0.1}),
            "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0}),
            "remove_frames": ("INT", {"default": 0, "min": 0, "max": 20, "step": 1}),
            },
            "optional": {
                "extend_tail_frame_if_adjust":("BOOLEAN", {"default": False})
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "FLOAT")
    RETURN_NAMES = ("images", "frame_count", "fps")
    FUNCTION = "adjust_frames"
    CATEGORY = "utils"

    def adjust_frames(self, images: torch.Tensor, duration: float, fps: float, remove_frames: int, extend_tail_frame_if_adjust: bool = False):
        if remove_frames > 0:
            images = images[:-remove_frames]
        batch_size = images.shape[0]
        min_frames = int(fps * duration)
        max_frames = int(fps * (duration + 1)) - 1
        
        # 如果在目标范围内，直接返回
        if min_frames <= batch_size <= max_frames:
            return (images, len(images), fps)
        
        # 如果帧数过少，需要插值
        if batch_size < min_frames:
            target_frames = min_frames + 5  if not extend_tail_frame_if_adjust else min_frames

        # 如果帧数过多，需要减帧
        if batch_size > max_frames:
            target_frames = max_frames - 5
        indices = np.linspace(0, batch_size - 1, target_frames)
        indices = np.floor(indices).astype(int)
        new_images = images[indices]


        if extend_tail_frame_if_adjust:
            unique, counts = np.unique(indices, return_counts=True)
            repeat_count = np.min(counts[:-1]) if len(counts) > 1 else int(fps // 2)
            logger.info(f"repeat_count: {repeat_count}, unique: {unique}, counts: {counts}")
            new_images = torch.cat([new_images, images[-1].unsqueeze(0).repeat(repeat_count, 1, 1, 1)], dim=0)
        return (new_images, len(new_images), fps)

class ImageTransitionLeftToRight:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{
            "before_image": ("IMAGE",),
            "after_image": ("IMAGE",),
            "duration": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 60.0, "step": 0.1}),
            "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0}),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "FLOAT", "FLOAT")
    RETURN_NAMES = ("images", "duration", "fps")
    FUNCTION = "create_transition"
    CATEGORY = "utils"

    def create_transition(self, before_image: torch.Tensor, after_image: torch.Tensor, duration: float, fps: float):
        print(f"before_image: {before_image.shape}, after_image: {after_image.shape}")
        # 确保输入是单张图片，如果是批次则取第一张
        if len(before_image.shape) == 4 and before_image.shape[0] > 1:
            before_image = before_image[0:1]
        if len(after_image.shape) == 4 and after_image.shape[0] > 1:
            after_image = after_image[0:1]
            
        # 获取目标尺寸（前图的尺寸）
        _, target_width = before_image.shape[1:3]
        
        
        adjusted_after = self.check_and_resizee_size(before_image, after_image)
        
        # 计算总帧数
        total_frames = int(duration * fps)
        
        # 创建过渡帧
        frames = []
        
        for i in range(total_frames):
            # 计算当前过渡位置 (0.0 到 1.0)
            progress = i / (total_frames - 1) if total_frames > 1 else 1.0
            
            # 计算过渡线的x坐标
            transition_x = int(target_width * progress)
            
            # 创建新帧
            new_frame = torch.zeros_like(before_image)
            
            # 从左到右过渡：左侧显示后图，右侧显示前图
            # 先填充整个前图
            new_frame[0] = before_image[0]
            
            # 然后在左侧填充后图（覆盖前图）
            if transition_x > 0:
                new_frame[0, :, :transition_x,:] = adjusted_after[0, :,  :transition_x, :]
            
            frames.append(new_frame)
        
        # 合并所有帧
        result = torch.cat(frames, dim=0)
        
        return (result, duration, fps)

    def check_and_resizee_size(self, before_image, after_image):
                # 获取目标尺寸（前图的尺寸）
        target_height, target_width = before_image.shape[1:3]
        
        # 获取后图的原始尺寸
        after_height, after_width = after_image.shape[1:3]
        adjusted_after = after_image
        if target_height != after_height or target_width != after_width:
            # 保持比例调整后图尺寸
            # 先计算缩放比例
            scale = min(target_height / after_height, target_width / after_width)
            new_height = int(after_height * scale)
            new_width = int(after_width * scale)
            
            # 缩放后图
            resized_after = torch.nn.functional.interpolate(
                after_image.movedim(-1, 1), size=(new_height, new_width), mode='bicubic', align_corners=False
            ).movedim(1, -1).clamp(0.0, 1.0)
            # 创建空白画布（与前图尺寸相同）
            adjusted_after = torch.zeros_like(before_image)
            
            # 计算居中位置
            y_offset = (target_height - new_height) // 2
            x_offset = (target_width - new_width) // 2
            
            # 将缩放后的图像放置在画布中央
            adjusted_after[:,:,:,:] = resized_after[0, y_offset:y_offset+new_height, x_offset:x_offset+new_width,:] 
        return adjusted_after

NODE_CLASS_MAPPINGS = {
    "FrameAdjuster": FrameAdjuster,
    "ImageTransitionLeftToRight": ImageTransitionLeftToRight
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FrameAdjuster": "Frame Adjuster",
    "ImageTransitionLeftToRight": "Image Transition Left to Right"
}
