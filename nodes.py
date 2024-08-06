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
from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel

app_dir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.realpath(__file__))))
config_dir = os.path.join(app_dir, "config")
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
    RETURN_NAMES = ("image", "mask", "enabled")
    FUNCTION = "load_image_with_switch"

    def load_image_with_switch(self, image, enabled=True):
        logger.debug("start load image")
        if not enabled:
            return None, None, enabled
        return self.load_image(image) + (enabled, )


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

    RETURN_TYPES = ("IMAGE", "MASK", "BOOLEAN", "STRING")
    RETURN_NAMES = ("image", "mask", "enabled", "filename")
    FUNCTION = "load_image_with_switch"

    def load_image_with_switch(self, image, enabled=True):
        logger.debug("start load image")
        if not enabled:
            return None, None, enabled
        return self.load_image(image) + (enabled, ) + (image,)


class LoadImageMaskWithSwitch(LoadImageMask):
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(
            os.path.join(input_dir, f))]
        return {"required":
                {"image": (sorted(files), {"image_upload": True}),
                 "channel": (["red", "green", "blue", "alpha"], ), },
                "optional": {
                    "enabled": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                },
                }

    CATEGORY = "mask"

    RETURN_TYPES = ("MASK", "BOOLEAN")
    RETURN_NAMES = ("mask", "enabled")
    FUNCTION = "load_image_with_switch"

    def load_image_with_switch(self, image, channel, enabled=True):
        if not enabled:
            return (None, enabled)
        return self.load_image(image, channel) + (enabled, )

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
                 "channel": (["red", "green", "blue", "alpha"], ), },
                "optional": {
                    "enabled": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                },
                }

    CATEGORY = "mask"

    RETURN_TYPES = ("MASK", "BOOLEAN")
    RETURN_NAMES = ("mask", "enabled")
    FUNCTION = "load_image_with_switch"

    def load_image_with_switch(self, image, channel, enabled=True):
        if not enabled:
            return (None, enabled)
        return self.load_image(image, channel) + (enabled, )

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
    file_path = os.path.join(config_dir, "gender_words_config.yaml")
    if not os.path.exists(file_path):
        gender_map = {
            'F': {
                'man': 'woman', 'men': 'women', 'sir': 'madam', 'father': 'mother',
                'husband': 'wife', 'son': 'daughter', 'boy': 'girl', 'brother': 'sister',
                'uncle': 'aunt', 'grandfather': 'grandmother', 'nephew': 'niece',
                'groom': 'bride', 'waiter': 'waitress', 'king': 'queen', 'gentleman': 'lady',
                'prince': 'princess', 'male': 'female', 'fiance': 'fiancee',
                'actor': 'actress', 'hero': 'heroine', 'he': 'she', 'his': 'her',
                'him': 'her', 'himself': 'herself', "he's": "she's",
            }
        }
        gender_map['M'] = {value: key for key,
                           value in gender_map['F'].items()}
        config = {"gender_map": gender_map, "gender_add_words": {
            "M": ["male",], "F": ["female"]}}
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
            "gender_prior": (["", "M", "F"],),
            "text": ("STRING", {"forceInput": True}),
        },
            "optional": {
            "gender_alternative": ("STRING", {"forceInput": True}),
            "enabled": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
        },
            "hidden": {
            "age": ("INT", {"default": -1, "min": -1, "max": 120, "step": 1}),
        }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "fun"
    CATEGORY = "utils/text/operations"
    GenderWordsConfig.load_config()

    @ staticmethod
    def fun(text, gender_prior="", gender_alternative=None, age=-1, enabled=True):
        gender = gender_prior if gender_prior else gender_alternative
        gender_map = GenderWordsConfig.get_config().get("gender_map", {})
        if not enabled or text is None or gender is None or gender.upper() not in gender_map:
            return (text,)
        result = ModifyTextGender.gender_swap(text, gender, gender_map)

        result = ModifyTextGender.gender_add_words(result, gender)
        logger.info(f"ModifyTextGender result:{result}")
        return (result,)

    @ staticmethod
    def gender_add_words(text, gender):
        gender_add_map = GenderWordsConfig.get_config().get("gender_add_words", {})
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
                original_word,  masks = original_word[:-1], original_word[-1]

            replacement = None
            for key, value in mappings.items():
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
                ['right',
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

    RETURN_TYPES = ("MASK", "MASK",)
    RETURN_NAMES = ("mask", "mask",)
    FUNCTION = 'split_mask'
    CATEGORY = 'mask'

    def split_mask(self, mask_prior, mask_alternative=None):
        mask = mask_prior if mask_prior is not None else mask_alternative
        if mask is None:
            return [torch.zeros((64, 64)).unsqueeze(0)] * 2
        ret_masks = []
        gray_image = mask[0].detach().cpu().numpy()

        # 对灰度图像进行阈值化处理，将白色区域转换为二进制掩码
        _, binary_mask = cv2.threshold(gray_image.astype(
            np.uint8), 0.5, 255, cv2.THRESH_BINARY)

        # 寻找白色区域的轮廓
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        logger.info(f"find mask areas:{len(contours)}")
        if contours is not None and len(contours) > 0:
            # 根据轮廓的面积对其进行排序
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
            contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

            for i, contour in enumerate(contours):
                # 创建一个新的同样尺寸的空图像
                new_mask = np.zeros_like(gray_image)
                # 在空图像中绘制当前轮廓
                cv2.drawContours(
                    new_mask, [contour], -1, (255), thickness=cv2.FILLED)
                ret_masks.append(torch.tensor(new_mask/255))
        else:
            # 如果未找到轮廓，则返回空 tensor
            ret_masks = [torch.tensor(np.zeros_like(gray_image))] * 2
        if len(ret_masks) < 2:
            ret_masks.extend(
                [torch.tensor(np.zeros_like(gray_image))]*(2-len(ret_masks)))

        ret_masks = [torch.unsqueeze(m, 0) for m in ret_masks]
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
                output_scaled = cv2.dilate(
                    output_scaled, kernel, iterations=grow)
            else:
                output_scaled = cv2.erode(
                    output_scaled, kernel, iterations=-grow)

            # Apply Gaussian blur using OpenCV
            if blur > 0:
                output_blurred = cv2.GaussianBlur(
                    output_scaled, (blur*2+1, blur*2+1), 0)
            else:
                output_blurred = output_scaled

            # Scale back to [0, 1]
            output = output_blurred.astype(np.float32) / 255.0
            out.append(torch.from_numpy(output))

        result = torch.stack(out, dim=0)
        if result.dim() == 2:
            result = torch.unsqueeze(result, 0)
        return (result,)


class MaskAutoSelector:
    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "mask_prior": ("MASK", ),
            },
            "optional": {
                "mask_alternative": ("MASK", ),
                "mask_third": ("MASK", )
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = 'select_mask'
    CATEGORY = 'mask'

    def select_mask(self, mask_prior=None, mask_alternative=None, mask_third=None):
        if mask_prior is not None:
            mask = mask_prior
        elif mask_alternative is not None:
            mask = mask_alternative
        else:
            mask = mask_third

        if mask is None:
            raise RuntimeError("all mask inputs is None")

        if mask.dim() == 2:
            mask = torch.unsqueeze(mask, 0)
        return (mask,)


class IntAndIntAddOffsetLiteral:
    RETURN_TYPES = ("INT", "INT",)
    RETURN_NAMES = ("int", "int add offset")
    FUNCTION = "get_int"
    CATEGORY = "number/utils"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"number": ("INT", {"default": 0, "min": 0, "max": 1000000})},
                "optional": {"offset": ("INT", {"default": 1, "step": 1}), }
                }

    def get_int(self, number, offset):
        if number == 0:
            return (0, 0)
        return (number, number + offset)


class IntMultipleAddLiteral:
    RETURN_TYPES = ("INT", "INT",)
    RETURN_NAMES = ("x", "ax + b")
    FUNCTION = "get_int"
    CATEGORY = "number/utils"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"number": ("INT", {"default": 0, "min": 0, "max": 1000000})},
                "optional": {"a_aign": (["positive", "negative"], {"default": "positive"}),
                             "a": ("FLOAT", {"default": 1.0, "step": 0.05}), "b": ("INT", {"default": 1, "step": 1}),
                             }
                }

    def get_int(self, number, a, b, a_aign):
        if a_aign == "negative":
            a = - a
        return (number, int(a*number + b))


MAX_RESOLUTION = 16384


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

    def composite_with_switch(self, destination, source, x, y, resize_source, mask=None, enabled=True, invert_mask=False):
        if not enabled:
            return (destination, )
        if invert_mask:
            mask = 1.0 - mask
        return self.composite(destination, source, x, y, resize_source, mask)


class CheckpointLoaderSimpleWithSwitch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                             },
                "optional": {
                "load_model": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "load_clip": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "load_vae": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
        }}
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"

    CATEGORY = "loaders"

    def load_checkpoint(self, ckpt_name, load_model, load_clip, load_vae):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_model=load_model, output_vae=load_vae,
                                                    output_clip=load_clip, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        return out[:3]


class ImageResizeTo8x:
    def __init__(self):
        pass

    ACTION_TYPE_RESIZE = "resize only"
    ACTION_TYPE_CROP = "crop to ratio"
    ACTION_TYPE_PAD = "pad to ratio"
    RESIZE_MODE_DOWNSCALE = "reduce size only"
    RESIZE_MODE_UPSCALE = "increase size only"
    RESIZE_MODE_ANY = "any"
    RETURN_TYPES = ("IMAGE", "MASK",)
    FUNCTION = "resize"
    CATEGORY = "image"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pixels": ("IMAGE",),
                "action": ([s.ACTION_TYPE_RESIZE, s.ACTION_TYPE_CROP, s.ACTION_TYPE_PAD],),
                "smaller_side": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                "larger_side": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                "target_width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                "target_height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                "scale_factor": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "resize_mode": ([s.RESIZE_MODE_DOWNSCALE, s.RESIZE_MODE_UPSCALE, s.RESIZE_MODE_ANY],),
                "side_ratio": ("STRING", {"default": "4:3"}),
                "crop_pad_position": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "pad_feathering": ("INT", {"default": 20, "min": 0, "max": 8192, "step": 1}),
                "all_szie_8x": (["disable", "crop", "resize"],),
            },
            "optional": {
                "mask_optional": ("MASK",),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(s, action, smaller_side, larger_side, scale_factor, resize_mode, side_ratio,target_width,target_height, **_):
        if side_ratio is not None:
            if action != s.ACTION_TYPE_RESIZE and s.parse_side_ratio(side_ratio) is None:
                return f"Invalid side ratio: {side_ratio}"

        if smaller_side is not None and larger_side is not None and scale_factor is not None:
            if int(smaller_side > 0) + int(larger_side > 0) + int(scale_factor > 0) > 1:
                return f"At most one scaling rule (smaller_side, larger_side, scale_factor) should be enabled by setting a non-zero value"

        if scale_factor is not None:
            if resize_mode == s.RESIZE_MODE_DOWNSCALE and scale_factor > 1.0:
                return f"For resize_mode {s.RESIZE_MODE_DOWNSCALE}, scale_factor should be less than one but got {scale_factor}"
            if resize_mode == s.RESIZE_MODE_UPSCALE and scale_factor > 0.0 and scale_factor < 1.0:
                return f"For resize_mode {s.RESIZE_MODE_UPSCALE}, scale_factor should be larger than one but got {scale_factor}"

        if (target_width == 0 and target_height != 0) or (target_width != 0 and target_height == 0):
            return f"targe_width and target_height should be set or unset simultaneously"
        return True

    @classmethod
    def parse_side_ratio(s, side_ratio):
        try:
            x, y = map(int, side_ratio.split(":", 1))
            if x < 1 or y < 1:
                raise Exception("Ratio factors have to be positive numbers")
            return float(x) / float(y)
        except:
            return None

    def vae_encode_crop_pixels(self, pixels):
        dims = pixels.shape[1:3]
        for d in range(len(dims)):
            x = (dims[d] // 8) * 8
            x_offset = (dims[d] % 8) // 2
            if x != dims[d]:
                pixels = pixels.narrow(d + 1, x_offset, x)
        return pixels

    def resize_a_little_to_8x(self, image, mask):
        in_h, in_w = image.shape[1:3]
        out_h = (in_h // 8) * 8
        out_w = (in_w // 8) * 8
        if in_h != out_h or in_w != out_w:
            image, mask = self.interpolate_to_target_size(image, mask, out_h, out_w)
        return image, mask

    def interpolate_to_target_size(self, image, mask, height, width):
        image = torch.nn.functional.interpolate(
                image.movedim(-1, 1), size=(height, width), mode="bicubic", antialias=True).movedim(1, -1).clamp(0.0, 1.0)
        mask = torch.nn.functional.interpolate(mask.unsqueeze(
                0), size=(height, width), mode="bicubic", antialias=True).squeeze(0).clamp(0.0, 1.0)
            
        return image, mask

    def resize(self, pixels, action, smaller_side, larger_side, scale_factor, resize_mode, side_ratio, crop_pad_position, pad_feathering, mask_optional=None, all_szie_8x="disable",target_width=0,target_height=0):
        validity = self.VALIDATE_INPUTS(
            action, smaller_side, larger_side, scale_factor, resize_mode, side_ratio,target_width,target_height)
        if validity is not True:
            raise Exception(validity)

        height, width = pixels.shape[1:3]
        if mask_optional is None:
            mask = torch.zeros(1, height, width, dtype=torch.float32)
        else:
            mask = mask_optional
            if mask.shape[1] != height or mask.shape[2] != width:
                mask = torch.nn.functional.interpolate(mask.unsqueeze(0), size=(
                    height, width), mode="bicubic").squeeze(0).clamp(0.0, 1.0)

        crop_x, crop_y, pad_x, pad_y = (0.0, 0.0, 0.0, 0.0)
        if action == self.ACTION_TYPE_CROP:
            target_ratio = self.parse_side_ratio(side_ratio)
            if height * target_ratio < width:
                crop_x = width - height * target_ratio
            else:
                crop_y = height - width / target_ratio
        elif action == self.ACTION_TYPE_PAD:
            target_ratio = self.parse_side_ratio(side_ratio)
            if height * target_ratio > width:
                pad_x = height * target_ratio - width
            else:
                pad_y = width / target_ratio - height

        if smaller_side > 0:
            if width + pad_x - crop_x > height + pad_y - crop_y:
                scale_factor = float(smaller_side) / (height + pad_y - crop_y)
            else:
                scale_factor = float(smaller_side) / (width + pad_x - crop_x)
        if larger_side > 0:
            if width + pad_x - crop_x > height + pad_y - crop_y:
                scale_factor = float(larger_side) / (width + pad_x - crop_x)
            else:
                scale_factor = float(larger_side) / (height + pad_y - crop_y)

        if (resize_mode == self.RESIZE_MODE_DOWNSCALE and scale_factor >= 1.0) or (resize_mode == self.RESIZE_MODE_UPSCALE and scale_factor <= 1.0):
            scale_factor = 0.0

        if target_width != 0 and target_height!=0:
            pixels, mask = self.interpolate_to_target_size(pixels, mask, target_height, target_width)
            crop_x, crop_y, pad_x, pad_y = (0.0, 0.0, 0.0, 0.0)
        elif scale_factor > 0.0:
            pixels = torch.nn.functional.interpolate(
                pixels.movedim(-1, 1), scale_factor=scale_factor, mode="bicubic", antialias=True).movedim(1, -1).clamp(0.0, 1.0)
            mask = torch.nn.functional.interpolate(mask.unsqueeze(
                0), scale_factor=scale_factor, mode="bicubic", antialias=True).squeeze(0).clamp(0.0, 1.0)
            height, width = pixels.shape[1:3]

            crop_x *= scale_factor
            crop_y *= scale_factor
            pad_x *= scale_factor
            pad_y *= scale_factor

        if crop_x > 0.0 or crop_y > 0.0:
            remove_x = (round(crop_x * crop_pad_position), round(crop_x *
                        (1 - crop_pad_position))) if crop_x > 0.0 else (0, 0)
            remove_y = (round(crop_y * crop_pad_position), round(crop_y *
                        (1 - crop_pad_position))) if crop_y > 0.0 else (0, 0)
            pixels = pixels[:, remove_y[0]:height -
                            remove_y[1], remove_x[0]:width - remove_x[1], :]
            mask = mask[:, remove_y[0]:height - remove_y[1],
                        remove_x[0]:width - remove_x[1]]
        elif pad_x > 0.0 or pad_y > 0.0:
            add_x = (round(pad_x * crop_pad_position), round(pad_x *
                     (1 - crop_pad_position))) if pad_x > 0.0 else (0, 0)
            add_y = (round(pad_y * crop_pad_position), round(pad_y *
                     (1 - crop_pad_position))) if pad_y > 0.0 else (0, 0)

            new_pixels = torch.zeros(pixels.shape[0], height + add_y[0] + add_y[1],
                                     width + add_x[0] + add_x[1], pixels.shape[3], dtype=torch.float32)
            new_pixels[:, add_y[0]:height + add_y[0],
                       add_x[0]:width + add_x[0], :] = pixels
            pixels = new_pixels

            new_mask = torch.ones(
                mask.shape[0], height + add_y[0] + add_y[1], width + add_x[0] + add_x[1], dtype=torch.float32)
            new_mask[:, add_y[0]:height + add_y[0],
                     add_x[0]:width + add_x[0]] = mask
            mask = new_mask

            if pad_feathering > 0:
                for i in range(mask.shape[0]):
                    for j in range(pad_feathering):
                        feather_strength = (
                            1 - j / pad_feathering) * (1 - j / pad_feathering)
                        if add_x[0] > 0 and j < width:
                            for k in range(height):
                                mask[i, k, add_x[0] +
                                     j] = max(mask[i, k, add_x[0] + j], feather_strength)
                        if add_x[1] > 0 and j < width:
                            for k in range(height):
                                mask[i, k, width + add_x[0] - j - 1] = max(
                                    mask[i, k, width + add_x[0] - j - 1], feather_strength)
                        if add_y[0] > 0 and j < height:
                            for k in range(width):
                                mask[i, add_y[0] + j,
                                     k] = max(mask[i, add_y[0] + j, k], feather_strength)
                        if add_y[1] > 0 and j < height:
                            for k in range(width):
                                mask[i, height + add_y[0] - j - 1, k] = max(
                                    mask[i, height + add_y[0] - j - 1, k], feather_strength)
        if all_szie_8x == "crop":
            pixels = self.vae_encode_crop_pixels(pixels)
            mask = self.vae_encode_crop_pixels(mask)
        elif all_szie_8x == "resize":
            pixels, mask = self.resize_a_little_to_8x(pixels, mask)
        return (pixels, mask)


class TextPreview:
    """this node code comes from ComfyUI-Custom-Scripts\py\show_text.py. thanks the orininal writer."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    FUNCTION = "notify"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)

    CATEGORY = "utils"

    def notify(self, text, unique_id=None, extra_pnginfo=None):
        if unique_id is not None and extra_pnginfo is not None:
            if not isinstance(extra_pnginfo, list):
                logger.warn("Error: extra_pnginfo is not a list")
            elif (
                not isinstance(extra_pnginfo[0], dict)
                or "workflow" not in extra_pnginfo[0]
            ):
                logger.warn(
                    "Error: extra_pnginfo[0] is not a dict or missing 'workflow' key")
            else:
                workflow = extra_pnginfo[0]["workflow"]
                node = next(
                    (x for x in workflow["nodes"] if str(
                        x["id"]) == str(unique_id[0])),
                    None,
                )
                if node:
                    node["widgets_values"] = [text]

        return {"ui": {"text": text}, "result": (text,)}


class MatchImageRatioToPreset:
    def __init__(self):
        self.presets = [
            (704, 1408), (704, 1344), (768, 1344), (768,
                                                    1280), (832, 1216), (832, 1152),
            (896, 1152), (896, 1088), (960, 1088), (960,
                                                    1024), (1024, 1024), (1024, 960),
            (1088, 960), (1088, 896), (1152,
                                       896), (1152, 832), (1216, 832), (1280, 768),
            (1344, 768), (1344, 704), (1408,
                                       704), (1472, 704), (1536, 640), (1600, 640),
            (1664, 576), (1728, 576)
        ]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT")
    RETURN_NAMES = ("standard_width", "standard_height", "min", "max")
    FUNCTION = "forward"

    CATEGORY = "utils"

    def forward(self, image):
        h, w = image.shape[1:-1]
        aspect_ratio = h / w

        # 计算每个预设的宽高比，并与输入图像的宽高比进行比较
        distances = [abs(aspect_ratio - h/w) for h, w in self.presets]
        closest_index = np.argmin(distances)

        # 选择最接近的预设尺寸
        target_h, target_w = self.presets[closest_index]

        max_v, min_v = max(target_h, target_w), min(target_h, target_w)
        logger.debug((target_w, target_h, min_v, max_v))
        return (target_w, target_h, min_v, max_v)


class UpscaleImageWithModelIfNeed(ImageUpscaleWithModel):

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"upscale_model": ("UPSCALE_MODEL",),
                             "image": ("IMAGE",),
                             "tile_size": ("INT", {"default": 512, "min": 128, "max": 10000}),
                             "threshold_of_xl_area": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 64.0, "step": 0.01}),
                             }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "forward"

    CATEGORY = "image/upscaling"

    def forward(self, image, upscale_model, tile_size=512, threshold_of_xl_area=0.9):
        h, w = image.shape[1:-1]
        percent = h * w / (1024 * 1024)
        if percent > threshold_of_xl_area:
            return (image,)

        return self.upscale(upscale_model, image, tile_size)


NODE_CLASS_MAPPINGS = {
    "LoadImageWithSwitch": LoadImageWithSwitch,
    "LoadImageMaskWithSwitch": LoadImageMaskWithSwitch,
    "LoadImageWithoutListDir": LoadImageWithoutListDir,
    "LoadImageMaskWithoutListDir": LoadImageMaskWithoutListDir,
    "ImageCompositeMaskedWithSwitch": ImageCompositeMaskedWithSwitch,
    "ImageBatchOneOrMore": ImageBatchOneOrMore,
    "ConcatTextOfUtils": ConcatTextOfUtils,
    "ModifyTextGender": ModifyTextGender,
    "IntAndIntAddOffsetLiteral": IntAndIntAddOffsetLiteral,
    "IntMultipleAddLiteral": IntMultipleAddLiteral,
    "ImageConcanateOfUtils": ImageConcanateOfUtils,
    "ColorCorrectOfUtils": ColorCorrectOfUtils,
    "SplitMask": SplitMask,
    "MaskFastGrow": MaskFastGrow,
    "MaskAutoSelector": MaskAutoSelector,
    "CheckpointLoaderSimpleWithSwitch": CheckpointLoaderSimpleWithSwitch,
    "ImageResizeTo8x": ImageResizeTo8x,
    "TextPreview": TextPreview,
    "MatchImageRatioToPreset": MatchImageRatioToPreset,
    "UpscaleImageWithModelIfNeed": UpscaleImageWithModelIfNeed,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageWithSwitch": "Load Image with switch",
    "LoadImageWithoutListDir": "Load Image without listing input dir",
    "LoadImageMaskWithSwitch": "Load Image as Mask with switch",
    "LoadImageMaskWithoutListDir": "Load Image as Mask without listing input dir",
    "ImageCompositeMaskedWithSwitch": "Image Composite Masked with switch",
    "ImageBatchOneOrMore": "Batch Images One or More",
    "ConcatTextOfUtils": "Concat text",
    "ModifyTextGender": "Modify Text Gender",
    "IntAndIntAddOffsetLiteral": "Int And Int Add Offset Literal",
    "IntMultipleAddLiteral": "Int Multiple and Add Literal",
    "ImageConcanateOfUtils": "Image Concanate of utils",
    "AdjustColorTemperature": "Adjust color temperature",
    "ColorCorrectOfUtils": "Color Correct Of Utils",
    "SplitMask": "Split Mask by Contours",
    "MaskFastGrow": "MaskGrow fast",
    "MaskAutoSelector": "Mask auto selector",
    "CheckpointLoaderSimpleWithSwitch": "Load checkpoint with switch",
    "ImageResizeTo8x": "Image resize to 8x",
    "TextPreview": "Preview Text",
    "MatchImageRatioToPreset": "Match image ratio to stardard size",
    "UpscaleImageWithModelIfNeed": "Upscale image using model if need",
}
