import torch
    
    
class EmptyConditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}
    
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "get_empty_conditioning"
    CATEGORY = "utils/conditioning"

    def get_empty_conditioning(self):
        return ([[]], )

NODE_CLASS_MAPPINGS = {
    "EmptyConditioning": EmptyConditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EmptyConditioning": "Empty Conditioning",
}
