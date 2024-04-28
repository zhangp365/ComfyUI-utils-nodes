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
import yaml
import os
import sys
app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
config_dir = os.path.join(app_dir,"config")
if not os.path.exists(config_dir):
    os.makedirs(config_dir)

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
                    if image1.shape[1:] != other_image.shape[1:]:
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


class GenderWordsConfig:
    file_path = os.path.join(config_dir,"gender_words_config.yaml")
    if not os.path.exists(file_path):
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
        config = {"gender_map": gender_map, "gender_add_words":{"M":["male",],"F":["female"]}}
        with open(file_path, 'w') as file:
            yaml.dump(config, file)

    config = {}

    @staticmethod
    def load_config():
        with open(GenderWordsConfig.file_path, 'r') as file:
            GenderWordsConfig.config = yaml.safe_load(file)

    @staticmethod
    def get_config():
        return GenderWordsConfig.config

    @staticmethod
    def update_config(new_config):
        GenderWordsConfig.config.update(new_config)
        GenderWordsConfig.save_config()

    @staticmethod
    def save_config():
        with open(GenderWordsConfig.file_path, 'w') as file:
            yaml.dump(GenderWordsConfig.config, file)


class ModifyTextGender:
    """
    This node will modify the prompt string according gender. gender words include M, F
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
    GenderWordsConfig.load_config()   
    
    @ staticmethod
    def fun(text, gender, age=-1, enabled=True):
        gender_map = GenderWordsConfig.get_config().get("gender_map",{})
        if not enabled or text is None or gender is None or gender.upper() not in gender_map:
            return (text,)
        result = ModifyTextGender.gender_swap(text, gender, gender_map)
        
        result = ModifyTextGender.gender_add_words(result,gender)
        logger.info(f"ModifyTextGender result:{result}")
        return (result,)
    
    @ staticmethod
    def gender_add_words(text, gender):
        gender_add_map = GenderWordsConfig.get_config().get("gender_add_words",{})
        prefixes = gender_add_map[gender.upper()]
        result = ", ".join(prefixes + [text])
        return result

    @ staticmethod
    def gender_swap(text, gender, gender_map):
        words = text.split()
        mappings = gender_map[gender.upper()]
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


class ImageConcanate:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image1": ("IMAGE",),
            "image2": ("IMAGE",),
            "direction": (
            [   'right',
                'down',
                'left',
                'up',
            ],
            {
            "default": 'right'
             }),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "concanate"
    CATEGORY = "KJNodes"

    def concanate(self, image1, image2, direction):
        if not image2:
            return (image1,)
        if image1.shape[1:] != image2.shape[1:]:
            image2 = comfy.utils.common_upscale(
                            image2.movedim(-1, 1), image1.shape[2], image1.shape[1], "bilinear", "center").movedim(1, -1)
        if direction == 'right':
            row = torch.cat((image1, image2), dim=2)
        elif direction == 'down':
            row = torch.cat((image1, image2), dim=1)
        elif direction == 'left':
            row = torch.cat((image2, image1), dim=2)
        elif direction == 'up':
            row = torch.cat((image2, image1), dim=1)
        return (row,)

class IntAndIntAddOffsetLiteral:
    RETURN_TYPES = ("INT","INT",)
    RETURN_NAMES = ("int", "int add offset")
    FUNCTION = "get_int"
    CATEGORY = "ImageSaverTools/utils"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"int": ("INT", {"default": 0, "min": 0, "max": 1000000})},
                "optional":{"offset": ("INT", {"default": 1, "step": 1}),}
                }

    def get_int(self, int, offset):
        return (int,int + offset)
    


NODE_CLASS_MAPPINGS = {
    "LoadImageWithSwitch": LoadImageWithSwitch,
    "ImageBatchOneOrMore": ImageBatchOneOrMore,
    "ConcatText": ConcatText,
    "ModifyTextGender": ModifyTextGender,
    "IntAndIntAddOffsetLiteral":IntAndIntAddOffsetLiteral,
    "ImageConcanate":ImageConcanate
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageWithSwitch": "Load Image with switch",
    "ImageBatchOneOrMore": "Batch Images One or More",
    "ConcatText":"Concat text",
    "ModifyTextGender":"Modify Text Gender",
    "IntAndIntAddOffsetLiteral": "Int And Int Add Offset Literal",
    "ImageConcanate":"Image Concanate of utils"
}
