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

class ImageTransitionBase:
    """图像过渡效果的基类"""
    
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
            "optional": {
               "bounce_back": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "FLOAT", "FLOAT")
    RETURN_NAMES = ("images", "duration", "fps")
    FUNCTION = "create_transition"
    CATEGORY = "utils"

    def create_transition(self, before_image: torch.Tensor, after_image: torch.Tensor, duration: float, fps: float, bounce_back: bool = False):
        # 确保输入是单张图片，如果是批次则取第一张
        if len(before_image.shape) == 4 and before_image.shape[0] > 1:
            before_image = before_image[0:1]
        if len(after_image.shape) == 4 and after_image.shape[0] > 1:
            after_image = after_image[0:1]
            
        # 调整后图尺寸以匹配前图
        adjusted_after = self.check_and_resize_size(before_image, after_image)
        
        # 计算总帧数
        total_frames = int(duration * fps)
        
        # 创建过渡帧
        frames = []
        
        if bounce_back:
            # 如果启用回弹效果，先从左到右，再从右到左
            # 前半段：从左到右
            half_frames = total_frames // 2
            for i in range(half_frames):
                progress = i / (half_frames - 1) if half_frames > 1 else 1.0
                new_frame = self.create_transition_frame(before_image, adjusted_after, progress)
                frames.append(new_frame)
            
            # 后半段：从右到左
            for i in range(half_frames, total_frames):
                progress = 1.0 - ((i - half_frames) / (total_frames - half_frames - 1) if total_frames - half_frames > 1 else 0.0)
                new_frame = self.create_transition_frame(before_image, adjusted_after, progress)
                frames.append(new_frame)
        else:
            # 正常的单向过渡
            for i in range(total_frames):
                progress = i / (total_frames - 1) if total_frames > 1 else 1.0
                new_frame = self.create_transition_frame(before_image, adjusted_after, progress)
                frames.append(new_frame)
        
        # 合并所有帧
        result = torch.cat(frames, dim=0)
        
        return (result, duration, fps)

    def create_transition_frame(self, before_image: torch.Tensor, after_image: torch.Tensor, progress: float):
        """创建过渡帧的抽象方法，子类需要实现"""
        raise NotImplementedError("子类必须实现create_transition_frame方法")

    def check_and_resize_size(self, before_image, after_image):
        # 获取目标尺寸（前图的尺寸）
        before_height, before_width = before_image.shape[1:3]
        
        # 获取后图的原始尺寸
        after_height, after_width = after_image.shape[1:3]
        
        # 如果尺寸相同，直接返回
        if before_height == after_height and before_width == after_width:
            return after_image

        # 计算宽高比
        before_ratio = before_width / before_height
        after_ratio = after_width / after_height
        
        logger.debug(f"before_image: {before_image.shape}, after_image: {after_image.shape}")
        
        # 调整后图尺寸，填充满目标尺寸（可能需要裁剪）
        if after_ratio > before_ratio:
            # 后图更宽，需要裁剪宽度
            new_width = int(after_height * before_ratio)           
            
            # 计算裁剪的起始位置（居中裁剪）
            start_x = (after_width - new_width) // 2
            logger.debug(f"start_x: {start_x}, new_width: {new_width}")
            # 裁剪后图
            cropped_after = after_image[:, :, start_x:start_x+new_width, :]
        else:
            # 后图更高，需要裁剪高度
            new_height = int(after_width / before_ratio)
            
            # 计算裁剪的起始位置（居中裁剪）
            start_y = (after_height - new_height) // 2
            logger.debug(f"start_y: {start_y}, new_height: {new_height}")
            # 裁剪后图
            cropped_after = after_image[:, start_y:start_y+new_height, :, :]
        logger.debug(f"cropped_after: {cropped_after.shape}")
        # 缩放到目标尺寸
        adjusted_after = torch.nn.functional.interpolate(
            cropped_after.movedim(-1, 1), 
            size=(before_height, before_width), 
            mode='bicubic', 
            align_corners=False
        ).movedim(1, -1).clamp(0.0, 1.0)
        
        return adjusted_after

class ImageTransitionLeftToRight(ImageTransitionBase):
    """从左到右的图像过渡效果"""
    
    def create_transition_frame(self, before_image: torch.Tensor, after_image: torch.Tensor, progress: float):
        # 获取目标尺寸（前图的尺寸）
        _, target_width = before_image.shape[1:3]
        
        # 计算过渡线的x坐标
        transition_x = int(target_width * progress)
        
        # 创建新帧
        new_frame = torch.zeros_like(before_image)
        
        # 从左到右过渡：左侧显示后图，右侧显示前图
        # 先填充整个前图
        new_frame[0] = before_image[0]
        
        # 然后在左侧填充后图（覆盖前图）
        if transition_x > 0:
            new_frame[0, :, :transition_x, :] = after_image[0, :, :transition_x, :]
        
        return new_frame

class ImageTransitionTopToBottom(ImageTransitionBase):
    """从上到下的图像过渡效果"""
    
    def create_transition_frame(self, before_image: torch.Tensor, after_image: torch.Tensor, progress: float):
        # 获取目标尺寸（前图的尺寸）
        target_height, _ = before_image.shape[1:3]
        
        # 计算过渡线的y坐标
        transition_y = int(target_height * progress)
        
        # 创建新帧
        new_frame = torch.zeros_like(before_image)
        
        # 从上到下过渡：上方显示后图，下方显示前图
        # 先填充整个前图
        new_frame[0] = before_image[0]
        
        # 然后在上方填充后图（覆盖前图）
        if transition_y > 0:
            new_frame[0, :transition_y, :, :] = after_image[0, :transition_y, :, :]
        
        return new_frame

NODE_CLASS_MAPPINGS = {
    "FrameAdjuster": FrameAdjuster,
    "ImageTransitionLeftToRight": ImageTransitionLeftToRight,
    "ImageTransitionTopToBottom": ImageTransitionTopToBottom
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FrameAdjuster": "Frame Adjuster",
    "ImageTransitionLeftToRight": "Image Transition Left to Right",
    "ImageTransitionTopToBottom": "Image Transition Top to Bottom"
}
