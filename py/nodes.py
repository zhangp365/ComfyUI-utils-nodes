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
from math import dist
import folder_paths
from .utils import tensor2np,np2tensor

config_dir = os.path.join(folder_paths.base_path, "config")
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

    CATEGORY = "utils/image"

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

    CATEGORY = "utils/image"

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

    CATEGORY = "utils/mask"

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
                    "mask_repeat_number":("INT", {"default": 1, "min": 1, "step": 1}),
                    },
                }

    CATEGORY = "utils/mask"

    RETURN_TYPES = ("MASK", "BOOLEAN")
    RETURN_NAMES = ("mask", "enabled")
    FUNCTION = "load_image_with_switch"

    def load_image_with_switch(self, image, channel, enabled=True, mask_repeat_number=1):
        if not enabled:
            return (None, enabled)
        mask =  self.load_image(image, channel)[0] 
        mask = mask.unsqueeze(0) if mask.dim() == 2 else mask
        new_mask = mask.expand(mask_repeat_number, -1, -1)
        return (new_mask,  enabled)

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

    CATEGORY = "utils/image"

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
    CATEGORY = "utils/text"

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
            "gender_prior_weight": ("FLOAT", {"default": 1, "min": 0, "max": 3, "step": 0.1}),
            "gender_alternative": ("STRING", {"forceInput": True}),
            "enabled": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
        },
            "hidden": {
            "age": ("INT", {"default": -1, "min": -1, "max": 120, "step": 1}),
        }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "fun"
    CATEGORY = "utils/text"
    GenderWordsConfig.load_config()

    @ staticmethod
    def fun(text, gender_prior="", gender_alternative=None, age=-1, enabled=True,gender_prior_weight=1.0):
        gender= gender_prior if gender_prior else gender_alternative
        weight = gender_prior_weight if gender_prior else 1.0
        gender_map = GenderWordsConfig.get_config().get("gender_map", {})
        if not enabled or text is None or gender is None or gender.upper() not in gender_map:
            return (text,)
        result = ModifyTextGender.gender_swap(text, gender, gender_map)

        result = ModifyTextGender.gender_add_words(result, gender, weight = weight)
        logger.info(f"ModifyTextGender result:{result}")
        return (result,)

    @ staticmethod
    def gender_add_words(text, gender, weight = 1.0):
        gender_add_map = GenderWordsConfig.get_config().get("gender_add_words", {})
        prefixes = gender_add_map[gender.upper()]
        if weight != 1.0:
            prefixes = [f"({prefix}:{weight:.1f})" for prefix in prefixes]
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


class GenderControlOutput:
    """
    This node will modify the prompt string according gender. gender words include M, F
    """
    @ classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "gender_prior": (["", "M", "F"],),
            "male_text": ("STRING",  {"multiline": True, "defaultBehavior": "input"}),
            "male_float": ("FLOAT", {"default": 1, "step": 0.1}),  
            "male_int": ("INT", {"default": 1, "step": 1}),
            "female_text": ("STRING",  {"multiline": True, "defaultBehavior": "input"}),
            "female_float": ("FLOAT", {"default": 1, "step": 0.1}),  
            "female_int": ("INT", {"default": 1, "step": 1}),
        },
            "optional": {
            "gender_alternative": ("STRING", {"forceInput": True}),           
            }
        }

    RETURN_TYPES = ("STRING","FLOAT","INT","BOOLEAN","BOOLEAN")
    RETURN_NAMES = ("gender_text","float","int","is_male","is_female")
    FUNCTION = "fun"
    CATEGORY = "utils/text"

    @ staticmethod
    def fun(gender_prior,male_text,male_float,male_int,female_text,female_float,female_int, gender_alternative=None):
        gender= gender_prior if gender_prior else gender_alternative
        if gender is None or gender.upper() not in ["M", "F"]:
            raise Exception("can't get any gender input.")
        if gender.upper()== "M":
            return (male_text, male_float, male_int, True, False)
        else:
            return (female_text, female_float, female_int, False, True)
        

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
        },
        "optional":{
            "image3": ("IMAGE",),
            "image4": ("IMAGE",),
            "image5": ("IMAGE",),
            "image6": ("IMAGE",),
        }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "concanate"
    CATEGORY = "utils/image"

    def concanate(self, image1, image2=None, image3=None, image4=None, image5=None, image6=None, direction='right'):
        images = [image1]
        
        # 添加非空的image3-6到列表中
        for img in [image2,image3, image4, image5, image6]:
            if img is not None:
                images.append(img)
        
        # 如果只有一张图片，直接返回
        if len(images) == 1:
            return (images[0],)
        
        # 调整所有图片的大小为第一张图片的大小
        for i in range(1, len(images)):
            if images[i].shape[1:] != images[0].shape[1:]:
                images[i] = comfy.utils.common_upscale(
                    images[i].movedim(-1, 1), images[0].shape[2], images[0].shape[1], "bilinear", "center").movedim(1, -1)
        
        # 根据方向拼接图片
        if direction in ['right', 'left']:
            row = torch.cat(images if direction == 'right' else [i for i in reversed(images)], dim=2)
        elif direction in ['down', 'up']:
            row = torch.cat(images if direction == 'down' else [i for i in reversed(images)], dim=1)
        
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
    CATEGORY = 'utils/mask'

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
    CATEGORY = 'utils/mask'

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

class MaskFromFaceModel:

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {                
                "image": ("IMAGE",),
                "max_face_number": ("INT", {"default": -1, "min": -1, "max": 99, "step": 1}),
                "add_bbox_upper_points": ("BOOLEAN", {"default": False}),  # 新增参数
            },
            "optional": {
                "faceanalysis": ("FACEANALYSIS", ),
                "face_model": ("FACE_MODEL", ),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = 'mask_get'
    CATEGORY = 'utils/mask'

    def mask_get(self, image, max_face_number, add_bbox_upper_points, faceanalysis=None, face_model=None):
        if faceanalysis is None and face_model is None:
            raise Exception("both faceanalysis and face_model are none!")
        
        if face_model is None:
            image_np = tensor2np(image)
            image_np = image_np[0] if isinstance(image_np, list) else image_np
            face_model = self.analyze_faces(faceanalysis, image_np)

        if not isinstance(face_model,list):
            face_models = [face_model]
        else:
            face_models = face_model

        if max_face_number !=-1 and len(face_model) > max_face_number:
            face_models = self.remove_unavaible_face_models(face_models=face_models,max_people_number=max_face_number)

        h, w = image.shape[-3:-1]

        result = np.zeros((h, w), dtype=np.uint8)
        for face in face_models:
            points = face.landmark_2d_106.astype(np.int32)  # Convert landmarks to integer format
            
            if add_bbox_upper_points:
                  # 获取bbox的坐标
                x1, y1 = face.bbox[0:2]
                x2, y2 = face.bbox[2:4]
                
                # 计算上边的1/4和3/4位置的点
                width = x2 - x1
                left_quarter_x = x1 + width // 4
                right_quarter_x = x2 - width // 4
                
                # 创建两个新点
                left_quarter_point = np.array([left_quarter_x, y1], dtype=np.int32)
                right_quarter_point = np.array([right_quarter_x, y1], dtype=np.int32)
                
                # 将两个点添加到landmarks中
                points = np.vstack((points, left_quarter_point, right_quarter_point))

            points = points.reshape((-1, 1, 2))  # Reshape for cv2.drawContours

            # Compute the convex hull for the landmarks
            hull = cv2.convexHull(points)

            # Draw the convex hull on the mask as well
            cv2.drawContours(result, [hull], contourIdx=-1, color=255, thickness=cv2.FILLED)

        result = torch.unsqueeze(torch.tensor(np.clip(result/255, 0, 1)), 0)
        return (result,)
    
    def remove_unavaible_face_models(self, face_models, max_people_number):
        max_lengths = []

        # Calculate the maximum length for each group of keypoints
        kpss = [f.kps for f in face_models]
        for keypoints in kpss:
            max_length = self.get_max_distance(keypoints)
            max_lengths.append(max_length)

        sorted_touple = sorted(zip(face_models, max_lengths), key=lambda x:x[1], reverse=True)

        # Filter out keypoints groups that have a maximum length less than one-fourth of the largest maximum length
        filtered_face_models = [
            face_model for face_model, _ in sorted_touple[:max_people_number]
            ]

        return filtered_face_models

    def get_max_distance(self, keypoints):
        max_distance = 0

        # Calculate the distance between every pair of keypoints
        for i in range(len(keypoints)):
            for j in range(i + 1, len(keypoints)):
                if keypoints[i] is not None and keypoints[j] is not None:
                    distance = dist(keypoints[i], keypoints[j])
                    max_distance = max(max_distance, distance)

        return max_distance

    def analyze_faces(self, insightface, img_data: np.ndarray):
        for size in [(size, size) for size in range(640, 320, -320)]:
            insightface.det_model.input_size = size
            face = insightface.get(img_data)
            if face:                
                if 640 not in size:
                    print(f"\033[33mINFO: InsightFace detection resolution lowered to {size}.\033[0m")
                break
        return face 
class MaskCoverFourCorners:

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {                
                "width": ("INT", {"default": 1024, "min": 0, "max": 8096, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 0, "max": 8096, "step": 8}),
                "radius": ("INT", {"default": 100, "min": 0, "max": 8096, "step": 5}),
                "draw_top_left": ("BOOLEAN", {"default": False}),
                "draw_top_right": ("BOOLEAN", {"default": False}),
                "draw_bottom_right": ("BOOLEAN", {"default": True}),
                "draw_bottom_left": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "size_as": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = 'mask_get'
    CATEGORY = 'utils/mask'

    def mask_get(self, size_as=None, width=1024, height=1024, radius=100, 
                            draw_top_left=False, draw_top_right=False, draw_bottom_right=True, draw_bottom_left=False):
        if size_as is not None:
            height, width = size_as.shape[-3:-1]
        result = self.create_mask_with_arcs(width, height, radius,draw_top_left, draw_top_right,draw_bottom_right,draw_bottom_left)

        result = torch.unsqueeze(torch.tensor(np.clip(result/255, 0, 1)), 0)
        return (result,)
    
    def create_mask_with_arcs(self, width=None, height=None, radius=50, 
                            draw_top_left=True, draw_top_right=True, draw_bottom_right=True, draw_bottom_left=True):
        """
        Creates a mask with circular arcs at the corners.
        
        :param width: Width of the mask.
        :param height: Height of the mask.
        :param radius: Radius of the circular arcs.
        :param draw_top_left: Boolean indicating whether to draw an arc at the top-left corner.
        :param draw_top_right: Boolean indicating whether to draw an arc at the top-right corner.
        :param draw_bottom_right: Boolean indicating whether to draw an arc at the bottom-right corner.
        :param draw_bottom_left: Boolean indicating whether to draw an arc at the bottom-left corner.
        :return: Mask image with arcs drawn.
        """       

        # Create a white mask
        mask = np.ones((height, width), dtype=np.uint8) * 255
        
        # Draw arcs on the mask, filling them with black color
        if draw_top_left:  # Top-left corner
            cv2.ellipse(mask, (0, 0), (radius, radius), 0, 0, 90, 0, -1)  # Fill with black
        if draw_top_right:  # Top-right corner
            cv2.ellipse(mask, (width, 0), (radius, radius), 0, 90, 180, 0, -1)  # Fill with black
        if draw_bottom_right:  # Bottom-right corner
            cv2.ellipse(mask, (width, height), (radius, radius), 0, 180, 270, 0, -1)  # Fill with black
        if draw_bottom_left:  # Bottom-left corner
            cv2.ellipse(mask, (0, height), (radius, radius), 0, 0, -90, 0, -1)  # Fill with black
        
        return mask
    

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
    CATEGORY = 'utils/mask'

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
    CATEGORY = "utils/numbers"

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
    CATEGORY = "utils/numbers"

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

    CATEGORY = "utils/image"

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

    CATEGORY = "utils/loaders"

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
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "width", "height")
    FUNCTION = "resize"
    CATEGORY = "utils/image"

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
               "all_size_16x": (["disable", "crop", "resize"],),
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

    def vae_encode_crop_pixels(self, pixels, ratio=8):
        dims = pixels.shape[1:3]
        for d in range(len(dims)):
            x = (dims[d] // ratio) * ratio
            x_offset = (dims[d] % ratio) // 2
            if x != dims[d]:
                pixels = pixels.narrow(d + 1, x_offset, x)
        return pixels

    def resize_a_little_to_ratio(self, image, mask,ratio=8):
        in_h, in_w = image.shape[1:3]
        out_h = (in_h // ratio) * ratio
        out_w = (in_w // ratio) * ratio
        if in_h != out_h or in_w != out_w:
            image, mask = self.interpolate_to_target_size(image, mask, out_h, out_w)
        return image, mask

    def interpolate_to_target_size(self, image, mask, height, width):
        image = torch.nn.functional.interpolate(
                image.movedim(-1, 1), size=(height, width), mode="bicubic", antialias=True).movedim(1, -1).clamp(0.0, 1.0)
        mask = torch.nn.functional.interpolate(mask.unsqueeze(
                0), size=(height, width), mode="bicubic", antialias=True).squeeze(0).clamp(0.0, 1.0)
            
        return image, mask

    def resize(self, pixels, action, smaller_side, larger_side, scale_factor, resize_mode, side_ratio, crop_pad_position, pad_feathering, mask_optional=None, all_szie_8x="disable",target_width=0,target_height=0,all_size_16x="disable"):
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
            target_ratio = target_width / target_height if target_width != 0 and target_height!=0 else self.parse_side_ratio(side_ratio) 
            if height * target_ratio < width:
                crop_x = width - height * target_ratio
            else:
                crop_y = height - width / target_ratio
        elif action == self.ACTION_TYPE_PAD:
            target_ratio = target_width / target_height if target_width != 0 and target_height!=0 else self.parse_side_ratio(side_ratio) 
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

        if scale_factor > 0.0:
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
        
        if target_width != 0 and target_height!=0:
            pixels, mask = self.interpolate_to_target_size(pixels, mask, target_height, target_width)
        
        if all_size_16x == "crop":
            pixels = self.vae_encode_crop_pixels(pixels,16)
            mask = self.vae_encode_crop_pixels(mask,16)
        elif all_size_16x == "resize":
            pixels, mask = self.resize_a_little_to_ratio(pixels, mask, ratio=16)

        elif all_szie_8x == "crop":
            pixels = self.vae_encode_crop_pixels(pixels)
            mask = self.vae_encode_crop_pixels(mask)
        elif all_szie_8x == "resize":
            pixels, mask = self.resize_a_little_to_ratio(pixels, mask, ratio=8)
        
        height, width = pixels.shape[1:3]
        return (pixels, mask, width, height)


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

    CATEGORY = "utils/text"

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

class TextInputAutoSelector:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "component_input": ("STRING", {"multiline": True}),               
            },
            "optional":{
                "alternative_input": ("STRING",{"forceInput": True}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "select_input"
    CATEGORY = "utils/text"

    def select_input(self, component_input, alternative_input=""):
        # 去除组件输入两端的空白字符
        component_input = component_input.strip()
        
        # 如果组件输入为空或只包含空白字符，选择外部输入
        if not component_input:
            selected_input = alternative_input
        else:
            selected_input = component_input
        
        return (selected_input,)


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
                "width_offset": ("INT", {"default": 0, "min": -128, "max": 128, "step": 8}),
                "height_offset": ("INT", {"default": 0, "min": -128, "max": 128, "step": 8}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT")
    RETURN_NAMES = ("standard_width", "standard_height", "min", "max")
    FUNCTION = "forward"

    CATEGORY = "utils/image"

    def forward(self, image, width_offset=0, height_offset=0):
        h, w = image.shape[1:-1]
        aspect_ratio = w / h

        # 计算每个预设的宽高比，并与输入图像的宽高比进行比较
        distances = [abs(aspect_ratio - w/h) for w,h in self.presets]
        closest_index = np.argmin(distances)

        # 选择最接近的预设尺寸
        target_w, target_h = self.presets[closest_index]
        if width_offset != 0:
            target_w += width_offset
        if height_offset != 0:
            target_h += height_offset

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
                             "threshold_of_xl_area": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 64.0, "step": 0.01}),
                             },
                "hidden":{
                            "tile_size": ("INT", {"default": 512, "min": 128, "max": 10000}),
                             }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "forward"

    CATEGORY = "utils/image"

    def forward(self, image, upscale_model, threshold_of_xl_area=0.9):
        h, w = image.shape[1:-1]
        percent = h * w / (1024 * 1024)
        if percent > threshold_of_xl_area:
            return (image,)

        return self.upscale(upscale_model, image)


NODE_CLASS_MAPPINGS = {
    #image
    "LoadImageWithSwitch": LoadImageWithSwitch,
    "LoadImageMaskWithSwitch": LoadImageMaskWithSwitch,
    "LoadImageWithoutListDir": LoadImageWithoutListDir,
    "LoadImageMaskWithoutListDir": LoadImageMaskWithoutListDir,
    "ImageCompositeMaskedWithSwitch": ImageCompositeMaskedWithSwitch,
    "ImageBatchOneOrMore": ImageBatchOneOrMore,
    "ImageConcanateOfUtils": ImageConcanateOfUtils,
    "ColorCorrectOfUtils": ColorCorrectOfUtils,
    "UpscaleImageWithModelIfNeed": UpscaleImageWithModelIfNeed,
    "ImageResizeTo8x": ImageResizeTo8x,

    # text
    "ConcatTextOfUtils": ConcatTextOfUtils,
    "ModifyTextGender": ModifyTextGender,
    "GenderControlOutput": GenderControlOutput,
    "TextPreview": TextPreview,
    "TextInputAutoSelector": TextInputAutoSelector,

    # numbers
    "MatchImageRatioToPreset": MatchImageRatioToPreset,
    "IntAndIntAddOffsetLiteral": IntAndIntAddOffsetLiteral,
    "IntMultipleAddLiteral": IntMultipleAddLiteral,
    
    # mask
    "SplitMask": SplitMask,
    "MaskFastGrow": MaskFastGrow,
    "MaskAutoSelector": MaskAutoSelector,
    "MaskFromFaceModel": MaskFromFaceModel,
    "MaskCoverFourCorners": MaskCoverFourCorners,

    #loader
    "CheckpointLoaderSimpleWithSwitch": CheckpointLoaderSimpleWithSwitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Image
    "LoadImageWithSwitch": "Load Image with Switch",
    "LoadImageMaskWithSwitch": "Load Image as Mask with Switch",
    "LoadImageWithoutListDir": "Load Image without Listing Input Dir",
    "LoadImageMaskWithoutListDir": "Load Image as Mask without Listing Input Dir",
    "ImageCompositeMaskedWithSwitch": "Image Composite Masked with Switch",
    "ImageBatchOneOrMore": "Batch Images One or More",
    "ImageConcanateOfUtils": "Image Concatenate of Utils",
    "ColorCorrectOfUtils": "Color Correct of Utils",
    "UpscaleImageWithModelIfNeed": "Upscale Image Using Model if Need",
    "ImageResizeTo8x": "Image Resize to 8x",

    # Text
    "ConcatTextOfUtils": "Concat Text",
    "ModifyTextGender": "Modify Text Gender",
    "GenderControlOutput": "Gender Control Output",
    "TextPreview": "Preview Text",
    "TextInputAutoSelector": "Text Input Auto Selector",

    # Number
    "MatchImageRatioToPreset": "Match Image Ratio to Standard Size",
    "IntAndIntAddOffsetLiteral": "Int And Int Add Offset Literal",
    "IntMultipleAddLiteral": "Int Multiple and Add Literal",

    # Mask
    "SplitMask": "Split Mask by Contours",
    "MaskFastGrow": "Mask Grow Fast",
    "MaskAutoSelector": "Mask Auto Selector",
    "MaskFromFaceModel": "Mask from FaceModel",
    "MaskCoverFourCorners": "Mask Cover Four Corners",

    # Loader
    "CheckpointLoaderSimpleWithSwitch": "Load Checkpoint with Switch",
}
