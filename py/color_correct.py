# Adapt from https://github.com/EllangoK/ComfyUI-post-processing-nodes/blob/master/post_processing/color_correct.py

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance


class ColorCorrectOfUtils:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "temperature": (
                    "FLOAT",
                    {"default": 0, "min": -100, "max": 100, "step": 5},
                ),
                "red": (
                    "FLOAT",
                    {"default": 0, "min": -100, "max": 100, "step": 5},
                ),
                "green": (
                    "FLOAT",
                    {"default": 0, "min": -100, "max": 100, "step": 5},
                ),
                "blue": (
                    "FLOAT",
                    {"default": 0, "min": -100, "max": 100, "step": 5},
                ),
                "hue": ("FLOAT", {"default": 0, "min": -90, "max": 90, "step": 5}),
                "brightness": (
                    "FLOAT",
                    {"default": 0, "min": -100, "max": 100, "step": 5},
                ),
                "contrast": (
                    "FLOAT",
                    {"default": 0, "min": -100, "max": 100, "step": 5},
                ),
                "saturation": (
                    "FLOAT",
                    {"default": 0, "min": -100, "max": 100, "step": 5},
                ),
                "gamma": ("FLOAT", {"default": 1, "min": 0.2, "max": 2.2, "step": 0.1}),
                "grain": ("FLOAT", {"default": 0, "min": 0.0, "max": 1, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "color_correct"

    CATEGORY = "Art Venture/Post Processing"

    def color_correct(
        self,
        image: torch.Tensor,
        temperature: float,
        red:float,
        green:float,
        blue:float,
        hue: float,
        brightness: float,
        contrast: float,
        saturation: float,
        gamma: float,
        grain: float,
    ):
        batch_size, height, width, _ = image.shape
        result = torch.zeros_like(image)

        brightness /= 100
        contrast /= 100
        saturation /= 100
        temperature /= -100
        red /= 100
        green /= 100
        blue /= 100

        brightness = 1 + brightness
        contrast = 1 + contrast
        saturation = 1 + saturation

        for b in range(batch_size):
            tensor_image = image[b].numpy()

            modified_image = Image.fromarray((tensor_image * 255).astype(np.uint8))

            # brightness
            modified_image = ImageEnhance.Brightness(modified_image).enhance(brightness)

            # contrast
            modified_image = ImageEnhance.Contrast(modified_image).enhance(contrast)
            modified_image = np.array(modified_image).astype(np.float32)

            # temperature
            if temperature > 0:
                modified_image[:, :, 0] *= 1 + temperature
                modified_image[:, :, 1] *= 1 + temperature * 0.4
            elif temperature < 0:
                modified_image[:, :, 0] *= 1 + temperature * 0.2
                modified_image[:, :, 2] *= 1 - temperature
            
            # red
            modified_image[:, :, 0] *= 1 + red
            # green
            modified_image[:, :, 1] *= 1 + green
            # blue
            modified_image[:, :, 2] *= 1 + blue
            modified_image = np.clip(modified_image, 0, 255) / 255

            # gamma
            modified_image = np.clip(np.power(modified_image, gamma), 0, 1)

            # saturation
            hls_img = cv2.cvtColor(modified_image, cv2.COLOR_RGB2HLS)
            hls_img[:, :, 2] = np.clip(saturation * hls_img[:, :, 2], 0, 1)
            modified_image = cv2.cvtColor(hls_img, cv2.COLOR_HLS2RGB) * 255

            # hue
            hsv_img = cv2.cvtColor(modified_image, cv2.COLOR_RGB2HSV)
            hsv_img[:, :, 0] = (hsv_img[:, :, 0] + hue) % 360
            modified_image = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)

            # grain
            if grain > 0:
                grain_image  = np.random.normal(0, 15, (modified_image.shape[0], modified_image.shape[1], 3)).astype(np.uint8)
                size = modified_image.shape[:2]
                modified_image = cv2.blendLinear(modified_image.astype(np.uint8), grain_image, np.ones(size,dtype=np.float32)*(1-grain),np.ones(size, dtype=np.float32)*grain)

            modified_image = modified_image.astype(np.uint8)
            modified_image = modified_image / 255
            modified_image = torch.from_numpy(modified_image).unsqueeze(0)
            result[b] = modified_image

        return (result,)
