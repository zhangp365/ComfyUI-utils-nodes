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

class VolcanoOutpaintingNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "ak": ("STRING", {"default": ""}),
                "sk": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 999999})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("result_image",)
    FUNCTION = "process_outpainting"
    CATEGORY = "image/volcano"

    def save_config(self, config):
        config_path = os.path.join(config_dir, 'volcano_config.yml')
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, indent=4)

    def load_config(self):
        config_path = os.path.join(config_dir, 'volcano_config.yml')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config

    def pil_to_base64(self, pil_image):
        """将PIL图像转换为base64字符串"""
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        img_data = buffer.getvalue()
        return base64.b64encode(img_data).decode('utf-8')

    def request_valcanic_outpainting(self, req_key, ak, sk, image_base64s, seed=42, top=0, left=0, bottom=0, right=0):
        """调用火山引擎图像扩展API"""
        visual_service = VisualService()
        
        if ak and sk:          
            self.save_config({"ak": ak, "sk": sk})
        else:
            config = self.load_config()
            ak = config.get("ak")
            sk = config.get("sk")
            if not ak or not sk:
                raise Exception("vocano engine ak or sk not found")

        visual_service.set_ak(ak)
        visual_service.set_sk(sk)   

        # 请求参数
        form = {
            "req_key": req_key,
            "binary_data_base64": image_base64s,
            "top": top,
            "left": left,
            "bottom": bottom,
            "right": right,
            "seed": seed
        }

        resp = visual_service.cv_process(form)
        
        # 解码返回的图像
        img_base64 = resp['data']['binary_data_base64'][0]
        img_data = base64.b64decode(img_base64)
        # 转换为PIL图像
        result_image = Image.open(io.BytesIO(img_data))

        # 删除图片base64， 方便print
        resp['data']['binary_data_base64'][0] =""

        logger.debug(f"vocano outpainting response: {resp}")
        return result_image, resp

    def process_outpainting(self, image, mask, ak, sk, seed):
        try:
            # 使用节点库的转换函数
            pil_image = tensor2pil(image)
            pil_mask = tensor2pil(mask)
            
            # 确保mask是灰度图像
            if pil_mask.mode != 'L':
                pil_mask = pil_mask.convert('L')
            
            # 直接转换为base64
            image_base64 = self.pil_to_base64(pil_image)
            mask_base64 = self.pil_to_base64(pil_mask)
            
            # 调用API
            result_image, _ = self.request_valcanic_outpainting(
                req_key="i2i_outpainting",
                ak=ak,
                sk=sk,
                image_base64s=[image_base64, mask_base64],
                seed=seed
            )
            
            # 使用节点库的转换函数转换结果为tensor
            result_tensor = pil2tensor(result_image)
            
            return (result_tensor,)
                
        except Exception as e:
            logger.exception(e)
            # 返回原图作为fallback
            return (image,)

# 节点映射
NODE_CLASS_MAPPINGS = {
    "VolcanoOutpaintingNode": VolcanoOutpaintingNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VolcanoOutpaintingNode": "volcano outpainting"
}
