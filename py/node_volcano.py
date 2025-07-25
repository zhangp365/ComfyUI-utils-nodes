import torch
import numpy as np
import base64
import io
from PIL import Image
import tempfile
import os
from volcengine.visual.VisualService import VisualService
import folder_paths
import logging
import yaml
from .utils import tensor2pil, pil2tensor

config_dir = os.path.join(folder_paths.base_path, "config")
if not os.path.exists(config_dir):
    os.makedirs(config_dir)

logger = logging.getLogger(__name__)

class VolcanoBaseNode:
    """火山引擎视觉服务基类"""
    
    def __init__(self):
        self.visual_service = VisualService()
    
    def save_config(self, config):
        """保存配置到文件"""
        config_path = os.path.join(config_dir, 'volcano_config.yml')
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, indent=4)

    def load_config(self):
        """从文件加载配置"""
        config_path = os.path.join(config_dir, 'volcano_config.yml')
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            return config
        except FileNotFoundError:
            return {}

    def pil_to_base64(self, pil_image):
        """将PIL图像转换为base64字符串"""
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        img_data = buffer.getvalue()
        return base64.b64encode(img_data).decode('utf-8')
    
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

class VolcanoOutpaintingNode(VolcanoBaseNode):
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

class VolcanoImageEditNode(VolcanoBaseNode):
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

# 节点映射
NODE_CLASS_MAPPINGS = {
    "VolcanoOutpaintingNode": VolcanoOutpaintingNode,
    "VolcanoImageEditNode": VolcanoImageEditNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VolcanoOutpaintingNode": "volcano outpainting",
    "VolcanoImageEditNode": "volcano image edit"
}
