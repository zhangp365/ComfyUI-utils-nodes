import cv2
import numpy as np
from ..r_nudenet.nudenet import NudeDetector
import os
import torch
import folder_paths as comfy_paths
from folder_paths import models_dir
from typing import Union, List
import json
import logging
from .utils import tensor2np,np2tensor

logger = logging.getLogger(__file__)

comfy_paths.folder_names_and_paths["nsfw"] = ([os.path.join(models_dir, "nsfw")], {".pt",".onnx"})

    
class DetectorForNSFW:

    def __init__(self) -> None:
        self.model = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),               
                "detect_size":([640, 320], {"default": 320}),
                "provider": (["CPU", "CUDA", "ROCM"], ),
            },
            "optional": {
                "model_name": (comfy_paths.get_filename_list("nsfw") + [""], {"default": ""}),
                "alternative_image": ("IMAGE",),
                "buttocks_exposed": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
                "female_breast_exposed": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
                "female_genitalia_exposed": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "anus_exposed": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
                "male_genitalia_exposed": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("filtered_image", "detect_result")
    FUNCTION = "filter_exposure"

    CATEGORY = "utils/filter"
    
    all_labels = [
        "FEMALE_GENITALIA_COVERED",
        "FACE_FEMALE",
        "BUTTOCKS_EXPOSED",
        "FEMALE_BREAST_EXPOSED",
        "FEMALE_GENITALIA_EXPOSED",
        "MALE_BREAST_EXPOSED",
        "ANUS_EXPOSED",
        "FEET_EXPOSED",
        "BELLY_COVERED",
        "FEET_COVERED",
        "ARMPITS_COVERED",
        "ARMPITS_EXPOSED",
        "FACE_MALE",
        "BELLY_EXPOSED",
        "MALE_GENITALIA_EXPOSED",
        "ANUS_COVERED",
        "FEMALE_BREAST_COVERED",
        "BUTTOCKS_COVERED",
    ]

    def filter_exposure(self, image, model_name=None, detect_size=320, provider="CPU", alternative_image=None, **kwargs):
        if self.model is None:
            self.init_model(model_name, detect_size, provider)

        if alternative_image is not None:
            alternative_image = tensor2np(alternative_image)
        
        images = tensor2np(image)
        if not isinstance(images, List):
            images = [images]

        results, result_info = [],[]
        for img in images:
            detect_result = self.model.detect(img)
            
            logger.debug(f"nudenet detect result:{detect_result}")
            filtered_results = []
            for item in detect_result:
                label = item['class']
                score = item['score']
                confidence_level = kwargs.get(label.lower())
                if label.lower() in kwargs and score > confidence_level:
                    filtered_results.append(item)
            info = {"detect_result":detect_result}
            if len(filtered_results) == 0:
                results.append(img)
                info["nsfw"] = False
            else:
                placeholder_image = alternative_image if alternative_image is not None else np.ones_like(img) * 255
                results.append(placeholder_image)
                info["nsfw"] = True

            result_info.append(info)

        result_tensor = np2tensor(results)
        result_info = json.dumps(result_info)
        return (result_tensor, result_info,)

    def init_model(self, model_name, detect_size, provider):
        model_path = comfy_paths.get_full_path("nsfw", model_name) if model_name else None
        self.model = NudeDetector(model_path=model_path, providers=[provider + 'ExecutionProvider',], inference_resolution=detect_size)
    

NODE_CLASS_MAPPINGS = {
    #image
    "DetectorForNSFW": DetectorForNSFW,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Image
    "DetectorForNSFW": "detector for the NSFW",

}
