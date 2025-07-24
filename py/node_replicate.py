import os
import sys
sys.path.append(".")
import replicate
import folder_paths
import logging
import yaml
import numpy as np
import requests
from PIL import Image
import io

logger = logging.getLogger(__name__)

config_dir = os.path.join(folder_paths.base_path, "config")
if not os.path.exists(config_dir):
    os.makedirs(config_dir)


def get_config():
    try:
        config_path = os.path.join(config_dir, 'replicate_config.yml')
        with open(config_path, 'r') as f:  
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config
    except:
        return {}

def save_config(config):
    config_path = os.path.join(config_dir, 'replicate_config.yml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, indent=4)


class ReplicateRequstNode:
    def __init__(self, api_key=None):
        config = get_config()
        self.api_key = api_key or config.get("REPLICATE_API_TOKEN")
        if self.api_key is not None:
            self.configure_replicate()

    def configure_replicate(self):
        if self.api_key:
            os.environ["REPLICATE_API_TOKEN"] = self.api_key

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "style of 80s cyberpunk, a portrait photo", "multiline": True}),
                "seed": ("INT", {"default": 42, "min": 0}),               
                "aspect_ratio": (["1:1", "16:9", "21:9", "3:2", "4:3", "5:4", "9:16", "9:21", "2:3", "3:4", "4:5"], {"default": "1:1"}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 100}),
                "guidance": ("FLOAT", {"default": 3.5, "min": 0.1, "max": 100.0, "step": 0.1}),
                "go_fast": ("BOOLEAN", {"default": True}),               
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "lora_path": ("STRING", {"default": ""}),
                "lora_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "extra_lora": ("STRING", {"default": ""}),
                "extra_lora_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "model": ("STRING", {"default": "black-forest-labs/flux-dev-lora"}),
                "num_outputs": ("INT", {"default": 1, "min": 1, "max": 10}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "width", "height", "url")
    FUNCTION = "generate_image"
    CATEGORY = "utils/image"

    def generate_image(self, prompt, seed, aspect_ratio, steps, guidance, go_fast, lora_path="", lora_scale=1.0, 
                      api_key="", extra_lora="", extra_lora_scale=1.0, model="black-forest-labs/flux-dev-lora", 
                      num_outputs=1):
        # 更新API key
        if api_key.strip():
            self.api_key = api_key
            save_config({"REPLICATE_API_TOKEN": self.api_key})
            self.configure_replicate()

        if not self.api_key:
            raise ValueError("API key not found in replicate_config.yml or node input")

        try:
            # 准备输入参数
            input_params = {
                "prompt": prompt,
                "lora_weights": lora_path,
                "seed": seed,
                "aspect_ratio": aspect_ratio,
                "num_inference_steps": steps,
                "guidance": guidance,
                "go_fast": go_fast,
                "lora_scale": lora_scale,
                "output_format": "png",
                "num_outputs": num_outputs
            }

            # 添加额外的LoRA参数
            if extra_lora.strip():
                input_params["extra_lora"] = extra_lora
                input_params["extra_lora_scale"] = extra_lora_scale

            logger.debug(f"调用Replicate API，参数: {input_params}")

            # 调用Replicate API
            output = replicate.run(model, input=input_params)
            
            if not output or len(output) == 0:
                raise Exception("Replicate API返回空结果")

            images = []
            urls = []
            for image_url in output:
                logger.debug(f"生成的图片URL: {image_url}")
                urls.append(str(image_url))
                # 下载图片
                response = requests.get(image_url)
                response.raise_for_status()
                
                # 转换为PIL图像
                image = Image.open(io.BytesIO(response.content))
                width, height = image.size

                image_array = np.array(image)
                if len(image_array.shape) == 3 and image_array.shape[2] == 4:  # RGBA
                    image_array = image_array[:, :, :3]  # 转换为RGB
                images.append(image_array)
            
            from .utils import np2tensor
            image_tensor = np2tensor(images)
            urls_str = ",".join(urls)
            
            return  (image_tensor, width, height, urls_str)

        except Exception as e:
            logger.exception(f"Replicate API调用失败: {str(e)}")
            raise e


NODE_CLASS_MAPPINGS = {
    "ReplicateRequstNode": ReplicateRequstNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ReplicateRequstNode": "Replicate Request",
}

