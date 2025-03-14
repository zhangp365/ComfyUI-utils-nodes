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

NODE_CLASS_MAPPINGS = {
    "FrameAdjuster": FrameAdjuster
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FrameAdjuster": "Frame Adjuster"
}
