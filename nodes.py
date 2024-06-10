import logging
import folder_paths
from nodes import LoadImage, LoadImageMask
from comfy_extras.nodes_mask import ImageCompositeMasked
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
import numpy as np
from PIL import Image, ImageEnhance
from .color_correct import ColorCorrectOfUtils
import cv2


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

    RETURN_TYPES = ("IMAGE", "MASK", "BOOLEAN")
    RETURN_NAMES = ("image","mask","enabled")
    FUNCTION = "load_image_with_switch"

    def load_image_with_switch(self, image, enabled=True):
        logger.debug("start load image")
        if not enabled:
            return None, None, enabled
        return self.load_image(image) +   (enabled, )


class LoadImageWithoutListDir(LoadImage):
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {"image": ([], {"image_upload": True})},
                "optional": {
                    "enabled": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                }
                }

    CATEGORY = "image"

    RETURN_TYPES = ("IMAGE", "MASK", "BOOLEAN")
    RETURN_NAMES = ("image","mask","enabled")
    FUNCTION = "load_image_with_switch"

    def load_image_with_switch(self, image, enabled=True):
        logger.debug("start load image")
        if not enabled:
            return None, None, enabled
        return self.load_image(image) +   (enabled, )
        
class LoadImageMaskWithSwitch(LoadImageMask):
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"image": (sorted(files), {"image_upload": True}),
                     "channel": (["red", "green", "blue","alpha"], ), },
                "optional": {
                    "enabled": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                    },
                }

    CATEGORY = "mask"

    RETURN_TYPES = ("MASK","BOOLEAN")
    RETURN_NAMES = ("mask","enabled")
    FUNCTION = "load_image_with_switch"
    def load_image_with_switch(self, image, channel, enabled=True):
        if not enabled:
            return (None, enabled)
        return self.load_image(image,channel) +  (enabled, )

    @classmethod
    def VALIDATE_INPUTS(s, image, enabled):
        if not enabled:
            return True
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True
    

class LoadImageMaskWithoutListDir(LoadImageMask):
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"image": ([], {"image_upload": True}),
                        "channel": (["red", "green", "blue","alpha"], ), },
                "optional": {
                    "enabled": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                    },
                }

    CATEGORY = "mask"

    RETURN_TYPES = ("MASK","BOOLEAN")
    RETURN_NAMES = ("mask","enabled")
    FUNCTION = "load_image_with_switch"
    def load_image_with_switch(self, image, channel, enabled=True):
        if not enabled:
            return (None, enabled)
        return self.load_image(image,channel) +  (enabled, )

    @classmethod
    def VALIDATE_INPUTS(s, image, enabled):
        if not enabled:
            return True
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True   


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

class ConcatTextOfUtils:
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
            "gender": ("STRING", {"placeholder":"please Input M for male, F for female"}),            
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


class ImageConcanateOfUtils:
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
    CATEGORY = "image"

    def concanate(self, image1, image2, direction):
        if image2 is None:
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

class SplitMask:

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "mask_prior": ("MASK", ),                
            },
            "optional": {
                "mask_alternative": ("MASK", )
            }
        }

    RETURN_TYPES = ("MASK","MASK","MASK",)
    RETURN_NAMES = ("mask","mask","mask",)
    FUNCTION = 'split_mask'
    CATEGORY = 'mask'

    def split_mask(self, mask_prior,mask_alternative = None):
        mask = mask_prior if mask_prior is not None else mask_alternative
        if mask is None:
            return [torch.zeros((64,64)).unsqueeze(0)] * 3
        ret_masks = []
        gray_image = mask[0].detach().cpu().numpy()

        # 对灰度图像进行阈值化处理，将白色区域转换为二进制掩码
        _, binary_mask = cv2.threshold(gray_image.astype(np.uint8), 0.5, 255, cv2.THRESH_BINARY)

        # 寻找白色区域的轮廓
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        logger.info(f"find mask areas:{len(contours)}")
        if contours is not None and len(contours) > 0:
            # 根据轮廓的面积对其进行排序
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
            contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

            for i, contour in enumerate(contours):
                # 创建一个新的同样尺寸的空图像
                new_mask = np.zeros_like(gray_image)
                # 在空图像中绘制当前轮廓
                cv2.drawContours(new_mask, [contour], -1, (255), thickness=cv2.FILLED)
                ret_masks.append(torch.tensor(new_mask/255))
        else:
            # 如果未找到轮廓，则返回空 tensor
            ret_masks = [torch.tensor(np.zeros_like(gray_image))] * 3
        if len(ret_masks) < 3:
            ret_masks.extend([torch.tensor(np.zeros_like(gray_image))]*(3-len(ret_masks)))

        ret_masks = [torch.unsqueeze(m,0) for m in ret_masks]    
        return ret_masks


class MaskFastGrow:

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "mask": ("MASK", ),                
                "invert_mask": ("BOOLEAN", {"default": True}),  # 反转mask
                "grow": ("INT", {"default": 4, "min": -999, "max": 999, "step": 1}),
                "blur": ("INT", {"default": 4, "min": 0, "max": 999, "step": 1}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = 'mask_grow'
    CATEGORY = 'mask'

    def mask_grow(self, mask, invert_mask, grow, blur):
        if mask.dim() == 2:
            mask = torch.unsqueeze(mask, 0)

        c = 0
        kernel = np.array([[c, 1, c],
                        [1, 1, 1],
                        [c, 1, c]], dtype=np.uint8)

        mask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
        out = []

        for m in mask:
            if invert_mask:
                m = 1 - m
            output = m.numpy().astype(np.float32)

            # Scale the float mask to [0, 255] for OpenCV processing
            output_scaled = (output * 255).astype(np.uint8)

            if grow > 0:
                output_scaled = cv2.dilate(output_scaled, kernel, iterations=grow)
            else:
                output_scaled = cv2.erode(output_scaled, kernel, iterations=-grow)

            # Apply Gaussian blur using OpenCV
            if blur> 0:
                output_blurred = cv2.GaussianBlur(output_scaled, (blur*2+1, blur*2+1), 0)
            else:
                output_blurred = output_scaled

            # Scale back to [0, 1]
            output = output_blurred.astype(np.float32) / 255.0
            out.append(torch.from_numpy(output))

        result = torch.stack(out,dim= 0)
        if result.dim() == 2:
            result = torch.unsqueeze(result, 0)
        return (result,)
        



class IntAndIntAddOffsetLiteral:
    RETURN_TYPES = ("INT","INT",)
    RETURN_NAMES = ("int", "int add offset")
    FUNCTION = "get_int"
    CATEGORY = "number/utils"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"number": ("INT", {"default": 0, "min": 0, "max": 1000000})},
                "optional":{"offset": ("INT", {"default": 1, "step": 1}),}
                }

    def get_int(self, number, offset):
        if number == 0:
            return(0 ,0)
        return (number,number + offset)

class IntMultipleAddLiteral:
    RETURN_TYPES = ("INT", "INT",)
    RETURN_NAMES = ("x", "ax + b")
    FUNCTION = "get_int"
    CATEGORY = "number/utils"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"number": ("INT", {"default": 0, "min": 0, "max": 1000000})},
                "optional":{ "a_aign":(["positive","negative"],{"default": "positive"}),
                            "a": ("FLOAT", {"default": 1.0, "step": 0.05}),"b": ("INT", {"default": 1, "step": 1}),
                           }
                }

    def get_int(self, number, a, b, a_aign):
        if a_aign == "negative":
            a = - a
        return (number, int( a*number + b))


MAX_RESOLUTION=16384
class ImageCompositeMaskedWithSwitch(ImageCompositeMasked):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "destination": ("IMAGE",),
                "source": ("IMAGE",),
                "x": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "y": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "resize_source": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "mask": ("MASK",),
                "enabled": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "invert_mask": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "composite_with_switch"

    CATEGORY = "image"

    def composite_with_switch(self, destination, source, x, y, resize_source, mask = None, enabled = True, invert_mask= False):
        if not enabled:
            return (destination, )
        if invert_mask:
            mask = 1.0 - mask
        return self.composite(destination, source, x, y, resize_source, mask)


NODE_CLASS_MAPPINGS = {
    "LoadImageWithSwitch": LoadImageWithSwitch,
    "LoadImageMaskWithSwitch":LoadImageMaskWithSwitch,
    "LoadImageWithoutListDir":LoadImageWithoutListDir,
    "LoadImageMaskWithoutListDir":LoadImageMaskWithoutListDir,
    "ImageCompositeMaskedWithSwitch":ImageCompositeMaskedWithSwitch,
    "ImageBatchOneOrMore": ImageBatchOneOrMore,
    "ConcatTextOfUtils": ConcatTextOfUtils,
    "ModifyTextGender": ModifyTextGender,
    "IntAndIntAddOffsetLiteral":IntAndIntAddOffsetLiteral,
    "IntMultipleAddLiteral":IntMultipleAddLiteral,
    "ImageConcanateOfUtils":ImageConcanateOfUtils,
    "ColorCorrectOfUtils": ColorCorrectOfUtils,
    "SplitMask":SplitMask,
    "MaskFastGrow":MaskFastGrow,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageWithSwitch": "Load Image with switch",
    "LoadImageWithoutListDir": "Load Image without listing input dir",
    "LoadImageMaskWithSwitch":"Load Image as Mask with switch",
    "LoadImageMaskWithoutListDir":"Load Image as Mask without listing input dir",
    "ImageCompositeMaskedWithSwitch": "Image Composite Masked with switch",
    "ImageBatchOneOrMore": "Batch Images One or More",
    "ConcatTextOfUtils":"Concat text",
    "ModifyTextGender":"Modify Text Gender",
    "IntAndIntAddOffsetLiteral": "Int And Int Add Offset Literal",
    "IntMultipleAddLiteral": "Int Multiple and Add Literal",
    "ImageConcanateOfUtils":"Image Concanate of utils",
    "AdjustColorTemperature": "Adjust color temperature",
    "ColorCorrectOfUtils": "Color Correct Of Utils",
    "SplitMask":"Split Mask by Contours",
    "MaskFastGrow":"MaskGrow fast",
}
