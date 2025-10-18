import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
from typing import Union, List, Dict, Any
from .utils import tensor2np,np2tensor
from ..r_deepface import demography

import folder_paths
import json
import logging
logger = logging.getLogger(__file__)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def prepare_deepface_home():
    deepface_path = os.path.join(folder_paths.models_dir, "deepface")

    # Deepface requires a specific structure within the DEEPFACE_HOME directory
    deepface_dot_path = os.path.join(deepface_path, ".deepface")
    deepface_weights_path = os.path.join(deepface_dot_path, "weights")
    if not os.path.exists(deepface_weights_path):
        os.makedirs(deepface_weights_path)

    os.environ["DEEPFACE_HOME"] = deepface_path


def get_largest_face(faces):
    largest_face = {}
    largest_area = 0
    if len(faces) == 1:
        return faces[0]
    
    for face in faces:
        if 'region' in face:
            w = face['region']['w']
            h = face['region']['h']
            area = w * h
            if area > largest_area:
                largest_area = area
                largest_face = face
    return largest_face


class DeepfaceAnalyzeFaceAttributes:
    '''
        - 'gender' (str): The gender in the detected face. "M" or "F"

        - 'emotion' (str): The emotion in the detected face.
            Possible values include "sad," "angry," "surprise," "fear," "happy,"
            "disgust," and "neutral."

        - 'race' (str): The race in the detected face.
            Possible values include "indian," "asian," "latino hispanic,"
            "black," "middle eastern," and "white."
    '''

    def __init__(self) -> None:
        prepare_deepface_home()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "detector_backend": ([
                     "opencv",
                     "ssd",
                     "dlib",
                     "mtcnn",
                     "retinaface",
                     "mediapipe",
                     "yolov8",
                     "yunet",
                     "fastmtcnn",
                ], {
                    "default": "yolov8",
                }),
            },
            "optional": {
                "analyze_gender": ("BOOLEAN", {"default": True}),
                "analyze_race": ("BOOLEAN", {"default": True}),
                "analyze_emotion": ("BOOLEAN", {"default": True}),
                "analyze_age": ("BOOLEAN", {"default": True}),
                "standard_single_face_image": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING","STRING","STRING","STRING", "STRING")
    RETURN_NAMES = ("gender","race","emotion","age", "json_info")
    FUNCTION = "analyze_face"
    CATEGORY = "utils/face"

    def analyze_face(self, image, detector_backend, analyze_gender=True, analyze_race=True, analyze_emotion=True, analyze_age=True, standard_single_face_image=False):
        # 将图像转换为numpy数组
        img_np = tensor2np(image)
        if isinstance(img_np, List):
            if len(img_np) > 1:
                logger.warn(f"DeepfaceAnalyzeFaceAttributes only support for one image and only analyze the largest face.")
            img_np = img_np[0]

        # 准备actions列表
        actions = []
        if analyze_gender:
            actions.append("gender")
        if analyze_race:
            actions.append("race")
        if analyze_emotion:
            actions.append("emotion")
        if analyze_age:
            actions.append("age")
        
        # 调用analyze函数
        results = demography.analyze(img_np, actions=actions, detector_backend=detector_backend, enforce_detection=False,  is_single_face_image=standard_single_face_image)
        
        # 获取面积最大的脸
        largest_face = get_largest_face(results)

        if not standard_single_face_image and largest_face.get("face_confidence")==0:
            largest_face ={}
        
        gender_map = {"Woman":"F","Man":"M",'':''}
        # 提取结果
        gender = gender_map.get(largest_face.get('dominant_gender', ''),'')if analyze_gender else ''
        race = largest_face.get('dominant_race', '') if analyze_race else ''
        emotion = largest_face.get('dominant_emotion', '') if analyze_emotion else ''
        age = str(largest_face.get('age', '0')) if analyze_age else '0'
        
        json_info= json.dumps(largest_face, cls=NumpyEncoder)
        return (gender, race, emotion, age, json_info)
    
NODE_CLASS_MAPPINGS = {
    #image
    "DeepfaceAnalyzeFaceAttributes": DeepfaceAnalyzeFaceAttributes,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Image
    "DeepfaceAnalyzeFaceAttributes": "Deepface Analyze Face Attributes",

}