import torch
import numpy as np

class MaskAreaComparison:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "area_threshold": ("INT", {
                    "default": 40000,
                    "min": 0,
                    "step": 50,
                    "max": 1000000000,
                    "display": "number"
                }),
            }
        }
    
    RETURN_TYPES = ("BOOLEAN", "BOOLEAN")
    RETURN_NAMES = ("is_greater", "is_smaller")
    FUNCTION = "compare_mask_area"
    CATEGORY = "mask/utils"
    
    def compare_mask_area(self, mask, area_threshold):
        # 确保mask是tensor格式
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        
        # 计算mask的实际面积（非零像素数量）
        mask_area = torch.sum(mask > 0.5).item()
        
        # 比较面积
        is_greater = mask_area > area_threshold
        is_smaller = mask_area < area_threshold
        
        return (is_greater, is_smaller)

# 节点映射
NODE_CLASS_MAPPINGS = {
    "MaskAreaComparison": MaskAreaComparison
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskAreaComparison": "Mask Area Comparison"
}
