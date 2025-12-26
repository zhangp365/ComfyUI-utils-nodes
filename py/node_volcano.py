import torch
import numpy as np
import base64
import io
from PIL import Image
import os
import requests
from typing import List, Optional

import folder_paths
import logging
import yaml
from .utils import tensor2pil, pil2tensor

config_dir = os.path.join(folder_paths.base_path, "config")
if not os.path.exists(config_dir):
    os.makedirs(config_dir)

logger = logging.getLogger(__name__)

class BaseNode:
    def save_config(self, config, config_filename):
        config_path = os.path.join(config_dir, config_filename)
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, indent=4)

    def load_config(self, config_filename):
        config_path = os.path.join(config_dir, config_filename)
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            return config
        except FileNotFoundError:
            return {}

    def pil_to_base64(self, pil_image):
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        img_data = buffer.getvalue()
        return base64.b64encode(img_data).decode('utf-8')

    def encode_image_to_base64(self, image_path):
        with open(image_path, 'rb') as f:
            img_data = f.read()
        return base64.b64encode(img_data).decode('utf-8')

    def download_image(self, image_url, target_image_name, project_path, timeout=120):
        response = requests.get(image_url, timeout=timeout)
        response.raise_for_status()
        
        output_dir = os.path.join(folder_paths.get_output_directory(), project_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        image_path = os.path.join(output_dir, target_image_name)
        with open(image_path, 'wb') as f:
            f.write(response.content)
        
        return image_path

class VolcanoOfficialNode(BaseNode):
    def __init__(self):
        from volcengine.visual.VisualService import VisualService
        self.visual_service = VisualService()
    
    def save_config(self, config):
        super().save_config(config, 'volcano_config.yml')

    def load_config(self):
        return super().load_config('volcano_config.yml')
    
    def setup_credentials(self, ak, sk):
        """设置API凭证"""
        if ak and sk:          
            self.save_config({"ak": ak, "sk": sk})
        else:
            config = self.load_config()
            ak = config.get("ak")
            sk = config.get("sk")
            if not ak or not sk:
                raise Exception("volcano engine ak or sk not found")
        
        self.visual_service.set_ak(ak)
        self.visual_service.set_sk(sk)
        return ak, sk
    
    def request_volcano_api(self, access_key, secret_key, form_data):
        """通用的火山引擎API请求方法"""
        # 设置凭证
        self.setup_credentials(access_key, secret_key)
        
        # 发送请求，如果失败则重试一次
        max_retries = 2
        last_exception = None
        
        for attempt in range(max_retries):
            try:                
                resp = self.visual_service.cv_process(form_data)                
                # 检查响应状态
                if resp.get('code') == 50429:
                    if attempt < max_retries - 1:
                        logger.warning(f"volcano api error, retry... ({attempt + 1}/{max_retries})")
                        continue
                    else:
                        raise Exception(f"volcano api error, retry failed: {resp}")
                
                # 解码返回的图像
                img_base64 = resp['data']['binary_data_base64'][0]
                img_data = base64.b64decode(img_base64)
                result_image = Image.open(io.BytesIO(img_data))
                
                # 删除图片base64，方便print
                resp['data']['binary_data_base64'][0] = ""
                logger.debug(f"volcano api response: {resp}")
                
                return result_image, resp
                
            except Exception as e:
                last_exception = e
                error_msg = str(e)
                
                # 检查是否超qps
                if ('"code":50429' in error_msg):
                    if attempt < max_retries - 1:
                        logger.warning(f"volcano api error, retry... ({attempt + 1}/{max_retries}): {error_msg}")
                        continue
                               
                raise e
        
        raise last_exception

class ArkBaseNode(BaseNode):
    def save_config(self, config):
        super().save_config(config, 'ark_config.yml')

    def load_config(self):
        return super().load_config('ark_config.yml')
    
    def calculate_image_size(self, level, aspect_ratio):
        size_2k = {
            "1:1": (2048, 2048),
            "4:3": (2304, 1728),
            "3:4": (1728, 2304),
            "16:9": (2560, 1440),
            "9:16": (1440, 2560),
            "3:2": (2496, 1664),
            "2:3": (1664, 2496),
            "21:9": (3024, 1296)
        }
        
        if aspect_ratio not in size_2k:
            raise Exception(f"不支持的宽高比: {aspect_ratio}")
        
        width_2k, height_2k = size_2k[aspect_ratio]
        
        if level == "1k":
            width = width_2k // 2
            height = height_2k // 2
        elif level == "2k":
            width = width_2k
            height = height_2k
        elif level == "4k":
            width = width_2k * 2
            height = height_2k * 2
        else:
            raise Exception(f"不支持的等级: {level}")
        
        return f"{width}x{height}"
    
    def setup_credentials(self, api_key, api_url=None, model=None):
        config = self.load_config()
        if api_key:
            config = {"api_key": api_key}
            if api_url:
                config["api_url"] = api_url
            if model:
                config["model"] = model
            self.save_config(config)
        else:            
            api_key = config.get("api_key")
            if not api_key:
                raise Exception("ark api key not found")
        
        api_url = config.get("api_url", "https://ark.ap-southeast.bytepluses.com/api/v3/images/generations")  if api_url is None else api_url
        model = config.get("model", "seedream-4-0-250828")  if model is None else model 
        
        return api_key, api_url, model
    
    def request_ark_api(self, prompt, image_paths=None, max_images=1, image_size="1440x2560", 
                       api_key=None, api_url=None, model=None, timeout=None):
        api_key, api_url, model = self.setup_credentials(api_key, api_url, model)
        
        images = []
        if image_paths and len(image_paths) > 0:
            for image_path in image_paths:
                if image_path.startswith("http://") or image_path.startswith("https://"):
                    images.append(image_path)
                else:
                    images.append(f"data:image/png;base64,{self.encode_image_to_base64(image_path)}")
        
        request_data = {
            "model": model,
            "prompt": prompt,
            "image": images,
            "sequential_image_generation": "auto",
            "sequential_image_generation_options": {
                "max_images": max_images
            },
            "response_format": "url",
            "size": image_size,
            "stream": False,
            "watermark": False
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        logger.info(f"ark请求数据 image_size: {image_size}, prompt: {prompt}")
        response = requests.post(api_url, headers=headers, json=request_data, timeout=timeout)
        response.raise_for_status()
        
        result_data = response.json()
        logger.info(f"ark请求成功，生成{len(result_data.get('data', []))}张图片")
        
        image_paths = []
        for i, image_data in enumerate(result_data.get('data', [])):
            image_url = image_data.get('url')
            if image_url:
                image_path = self.download_image(image_url, f"ark_result_{i}.png", "ark", timeout)
                image_paths.append(image_path)
        
        return image_paths

class VolcanoOutpaintingNode(VolcanoOfficialNode):
    """火山引擎图像扩展节点"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "access_key": ("STRING", {"default": ""}),
                "secret_key": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 999999})
            },
            "optional": {
                "version": ("STRING", {"default": "i2i_outpainting", "tooltip": "this value is the reqKey of the volcano api."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("result_image",)
    FUNCTION = "process_outpainting"
    CATEGORY = "image/volcano"

    def process_outpainting(self, image, mask, access_key, secret_key, seed, version="i2i_outpainting"):
        try:
            # 使用节点库的转换函数
            pil_image = tensor2pil(image)
            pil_mask = tensor2pil(mask)
            
            # 确保mask是灰度图像
            if pil_mask.mode != 'L':
                pil_mask = pil_mask.convert('L')
            
            # 转换为base64
            image_base64 = self.pil_to_base64(pil_image)
            mask_base64 = self.pil_to_base64(pil_mask)
            
            # 构建请求参数
            form_data = {
                "req_key": version,
                "binary_data_base64": [image_base64, mask_base64],
                "top": 0,
                "left": 0,
                "bottom": 0,
                "right": 0,
                "seed": seed
            }
            
            # 调用基类的通用API请求方法
            result_image, _ = self.request_volcano_api(access_key, secret_key, form_data)
            
            # 使用节点库的转换函数转换结果为tensor
            result_tensor = pil2tensor(result_image)
            
            return (result_tensor,)
                
        except Exception as e:
            logger.exception(e)
            raise e

class VolcanoImageEditNode(VolcanoOfficialNode):
    """火山引擎图像编辑节点"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "access_key": ("STRING", {"default": ""}),
                "secret_key": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "scale": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.01})
            },
            "optional": {
                "version": (["byteedit_v2.0","seededit_v3.0"], {"default": "seededit_v3.0"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("edited_image",)
    FUNCTION = "process_image_edit"
    CATEGORY = "image/volcano"

    def process_image_edit(self, image, prompt, access_key, secret_key, seed, scale=0.5, version="seededit_v3.0"):
        try:
            # 使用节点库的转换函数
            pil_image = tensor2pil(image)
            
            # 转换为base64
            image_base64 = self.pil_to_base64(pil_image)
            binary_data = [image_base64]
            # 构建请求参数
            form_data = {
                "req_key": version,
                "binary_data_base64": binary_data,
                "prompt": prompt,
                "seed": seed,
                "scale": scale
            }
            
            # 调用基类的通用API请求方法
            result_image, _ = self.request_volcano_api(access_key, secret_key, form_data)
            
            # 使用节点库的转换函数转换结果为tensor
            result_tensor = pil2tensor(result_image)
            
            return (result_tensor,)
                
        except Exception as e:
            logger.exception(e)
            raise e

class VolcanoArkImageEditNode(ArkBaseNode):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "level": (["1k", "2k", "4k"], {"default": "2k"}),
                "aspect_ratio": (["1:1", "4:3", "3:4", "16:9", "9:16", "3:2", "2:3", "21:9"], {"default": "9:16"}),
                "max_images": ("INT", {"default": 1, "min": 1, "max": 10}),
                "timeout": ("INT", {"default": 120, "min": 10, "max": 600})
            },
            "optional": {
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "api_key": ("STRING", {"default": ""}),
                "api_url": ("STRING", {"default": "", "multiline": True}),
                "model": ("STRING", {"default": ""})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("edited_image",)
    FUNCTION = "process_image_edit"
    CATEGORY = "image/ark"

    def process_image_edit(self, image, prompt, level, aspect_ratio, max_images, timeout, 
                          image2=None, image3=None, api_key="", api_url="", model=""):
        pil_image = tensor2pil(image)
        
        temp_dir = os.path.join(folder_paths.get_temp_directory(), "ark")
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        temp_image_path = os.path.join(temp_dir, "input_image.png")
        pil_image.save(temp_image_path)
        
        image_paths = [temp_image_path]
        
        if image2 is not None:
            pil_image2 = tensor2pil(image2)
            temp_image_path2 = os.path.join(temp_dir, "input_image2.png")
            pil_image2.save(temp_image_path2)
            image_paths.append(temp_image_path2)
        
        if image3 is not None:
            pil_image3 = tensor2pil(image3)
            temp_image_path3 = os.path.join(temp_dir, "input_image3.png")
            pil_image3.save(temp_image_path3)
            image_paths.append(temp_image_path3)
        
        image_size = self.calculate_image_size(level, aspect_ratio)
        
        image_paths_result = self.request_ark_api(
            prompt=prompt,
            image_paths=image_paths,
            max_images=max_images,
            image_size=image_size,
            api_key=api_key if api_key else None,
            api_url=api_url if api_url else None,
            model=model if model else None,
            timeout=timeout
        )
        
        if not image_paths_result or len(image_paths_result) == 0:
            raise Exception("ark api returned no images")
        
        result_image = Image.open(image_paths_result[0])
        result_tensor = pil2tensor(result_image)
        
        return (result_tensor,)

# 节点映射
NODE_CLASS_MAPPINGS = {
    "VolcanoOutpaintingNode": VolcanoOutpaintingNode,
    "VolcanoImageEditNode": VolcanoImageEditNode,
    "VolcanoArkImageEditNode": VolcanoArkImageEditNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VolcanoOutpaintingNode": "volcano outpainting",
    "VolcanoImageEditNode": "volcano image edit",
    "VolcanoArkImageEditNode": "volcano ark image edit"
}
