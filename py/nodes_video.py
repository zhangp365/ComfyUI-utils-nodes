import torch
import numpy as np

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
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "FLOAT")
    RETURN_NAMES = ("images", "frame_count", "fps")
    FUNCTION = "adjust_frames"
    CATEGORY = "utils"

    def adjust_frames(self, images, duration, fps, remove_frames):
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
            target_frames = min_frames + 5

        # 如果帧数过多，需要减帧
        if batch_size > max_frames:
            target_frames = max_frames - 5
        indices = np.linspace(0, batch_size - 1, target_frames)
        indices = np.floor(indices).astype(int)
        new_images = images[indices]
        return (new_images, len(new_images), fps)

NODE_CLASS_MAPPINGS = {
    "FrameAdjuster": FrameAdjuster
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FrameAdjuster": "Frame Adjuster"
}
