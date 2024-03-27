import logging
import folder_paths
from nodes import LoadImage
from comfy.cli_args import args
import comfy.model_management
import comfy.clip_vision
import comfy.controlnet
import comfy.utils
import comfy.sd
import comfy.sample
import comfy.samplers
import comfy.diffusers_load
import torch

import os
import sys


sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "comfy"))


logger = logging.getLogger(__file__)


class LoadImageWithSwitch(LoadImage):
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(
            os.path.join(input_dir, f))]
        return {"required":
                {"image": (sorted(files), {"image_upload": True})},
                "optional": {
                    "enabled": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                }
                }

    CATEGORY = "image"

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image_with_switch"

    def load_image_with_switch(self, image, enabled):
        logger.debug("start load image")
        if not enabled:
            return None, None
        return self.load_image(image)


class ImageBatchOneOrMore:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image1": ("IMAGE",)},
                "optional": {"image2": ("IMAGE",), "image3": ("IMAGE",), "image4": ("IMAGE",), "image5": ("IMAGE",), "image6": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "batch"

    CATEGORY = "image"

    def batch(self, image1, image2=None, image3=None, image4=None, image5=None, image6=None):
        images = [image1]
        for other_image in [image2, image3, image4, image5, image6]:
            if other_image is not None:
                try:
                    if image1.shape[1:] != image2.shape[1:]:
                        other_image = comfy.utils.common_upscale(
                            other_image.movedim(-1, 1), image1.shape[2], image1.shape[1], "bilinear", "center").movedim(1, -1)
                    images.append(other_image)
                except Exception as e:
                    logger.exception(e)
        s = torch.cat(images, dim=0)
        return (s,)

class ConcatText:
    """
    This node will concatenate two strings together
    """
    @ classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "text1": ("STRING", {"multiline": True, "defaultBehavior": "input"}),
            "text2": ("STRING", {"multiline": True, "defaultBehavior": "input"}),
            "separator": ("STRING", {"multiline": False, "default": ","}),
        }}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "fun"
    CATEGORY = "utils/text/operations"

    @ staticmethod
    def fun(text1, separator, text2):
        return (text1 + separator + text2,)

class ConcatText:
    """
    This node will concatenate two strings together
    """
    @ classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "text1": ("STRING", {"multiline": True, "defaultBehavior": "input"}),
            "text2": ("STRING", {"multiline": True, "defaultBehavior": "input"}),
            "separator": ("STRING", {"multiline": False, "default": ","}),
        }}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "fun"
    CATEGORY = "utils/text/operations"

    @ staticmethod
    def fun(text1, separator, text2):
        return (text1 + separator + text2,)


class ModifyTextGender:
    """
    This node will concatenate two strings together
    """
    @ classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "text": ("STRING", {"multiline": True, "defaultBehavior": "input"}),
            "gender": ("STRING", {"default": None}),            
        },
        "optional":{
            "age": ("INT", {"default": -1, "min": -1, "max": 120, "step": 1}),
             "enabled": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
        }}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "fun"
    CATEGORY = "utils/text/operations"
    gender_map = {
        'F': {
            'man': 'woman', 'men': 'women', 'sir': 'madam', 'father': 'mother', 
            'husband': 'wife', 'son': 'daughter', 'boy': 'girl', 'brother': 'sister', 
            'uncle': 'aunt', 'grandfather': 'grandmother', 'nephew': 'niece', 
            'groom': 'bride', 'waiter': 'waitress', 'king': 'queen', 'gentleman': 'lady', 
            'prince': 'princess', 'male': 'female', 'fiance': 'fiancee', 
            'actor': 'actress', 'hero': 'heroine', 'he': 'she', 'his': 'her', 
            'him': 'her', 'himself': 'herself',"he's": "she's",
        }
        }
    gender_map['M'] = {value:key for key,value in gender_map['F'].items()}

    @ staticmethod
    def fun(text, gender, age=-1, enabled=True):
        if not enabled or text is None or gender is None or gender.upper() not in ModifyTextGender.gender_map:
            return (text,)
        result = ModifyTextGender.gender_swap(text, gender)
        logger.debug(f"ModifyTextGender result:{result}")
        return (result,)
    
    @ staticmethod
    def gender_swap(text, gender):
        words = text.split()
        mappings = ModifyTextGender.gender_map[gender.upper()]
        for i, word in enumerate(words):
            masks = ""
            case = 'lower'
            original_word = word.lower()    
            if word.endswith(".") or word.endswith(",") or word.endswith("'") or word.endswith('"') or word.endswith(":"):
                case = "masks"
                original_word,  masks= original_word[:-1], original_word[-1]  

            replacement = None    
            for key,value in mappings.items():
                if len(key) == 2:
                    if original_word == key:
                        replacement = value
                        break
                elif original_word.startswith(key) or original_word.endswith(key):                    
                    replacement = original_word.replace(key, value)
                    break
            if replacement is not None:
                if case == "masks":
                    replacement = replacement + masks
                words[i] = replacement
        return ' '.join(words)     

NODE_CLASS_MAPPINGS = {
    "LoadImageWithSwitch": LoadImageWithSwitch,
    "ImageBatchOneOrMore": ImageBatchOneOrMore,
    "ConcatText": ConcatText,
    "ModifyTextGender": ModifyTextGender,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageWithSwitch": "Load Image with switch",
    "ImageBatchOneOrMore": "Batch Images One or More",
    "ConcatText":"Concat text",
    "ModifyTextGender":"Modify Text's Gender"
}
