import time
import comfy.model_management
import tempfile
import io
from PIL import Image
import requests
import numpy as np
import yaml
import logging
import folder_paths
import os
import sys
sys.path.append(".")
from comfy_api.latest._input_impl.video_types import VideoFromFile
from comfy.comfy_types import IO, FileLocator, ComfyNodeABC
from .utils import tensor2pil, np2tensor

logger = logging.getLogger(__name__)

config_dir = os.path.join(folder_paths.base_path, "config")
if not os.path.exists(config_dir):
    os.makedirs(config_dir)


def get_config():
    try:
        config_path = os.path.join(config_dir, 'novita_config.yml')
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config
    except:
        return {}


def save_config(config):
    config_path = os.path.join(config_dir, 'novita_config.yml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, indent=4)


class NovitaVideoRequestNode:
    def __init__(self, api_key=None):
        config = get_config()
        self.api_key = api_key or config.get("NOVITA_API_TOKEN")
        self.base_url = "https://api.novita.ai/v3/async"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def submit_video_request(self, model, prompt, img_url, resolution="720P", duration=5, 
                           prompt_extend=True, seed=None, loras=None, negative_prompt=""):
        """提交视频生成请求"""
        url = f"{self.base_url}/{model}"
        
        data = {
            "input": {
                "prompt": prompt,
                "img_url": img_url
            },
            "parameters": {
                "resolution": resolution,
                "duration": duration,
                "prompt_extend": prompt_extend
            }
        }
        
        if negative_prompt:
            data["input"]["negative_prompt"] = negative_prompt
        
        if seed is not None:
            data["parameters"]["seed"] = seed
            
        if loras:
            data["parameters"]["loras"] = loras
            
        response = requests.post(url, headers=self.headers, json=data)
        response.raise_for_status()
        return response.json()["task_id"]

    def query_video_result(self, task_id):
        """查询视频生成结果"""
        url = f"{self.base_url}/task-result"
        params = {"task_id": task_id}
        
        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        return response.json()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("STRING", {"default": "wan-2.2-i2v"}),
                "prompt": ("STRING", {"default": "A small cat running on the grass", "multiline": True}),
                "img_url": ("STRING", {"default": "", "multiline": True}),
                "resolution": (["480P", "720P", "1080P"], {"default": "720P"}),
                "duration": ("INT", {"default": 5, "min": 5, "max": 8}),
                "prompt_extend": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647, "control_after_generate": True}),
                "api_key": ("STRING", {"default": ""}),
                "timeout": ("INT", {"default": 300, "min": 1, "max": 3000}),
            }
        }

    RETURN_TYPES = (IO.VIDEO, "INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("video", "width", "height", "fps", "url")
    FUNCTION = "generate_video"
    CATEGORY = "utils/video"

    def generate_video(self, model, prompt, img_url, resolution, duration, prompt_extend, 
                      negative_prompt="", seed=0, api_key="", timeout=300):
        
        if api_key.strip():
            self.api_key = api_key
            save_config({"NOVITA_API_TOKEN": self.api_key})
            self.headers["Authorization"] = f"Bearer {self.api_key}"

        if not self.api_key:
            raise ValueError("API key not found in config or node input")

        if not img_url.strip():
            raise ValueError("img_url must be provided")

        try:
            task_id = self.submit_video_request(
                model=model,
                prompt=prompt,
                img_url=img_url,
                resolution=resolution,
                duration=duration,
                prompt_extend=prompt_extend,
                seed=seed if seed > 0 else None,
                negative_prompt=negative_prompt
            )

            logger.debug(f"任务已提交，task_id: {task_id}")

            start_time = time.time()
            while time.time() - start_time < timeout:
                if comfy.model_management.processing_interrupted():
                    raise comfy.model_management.InterruptProcessingException("ComfyUI interrupted")

                result = self.query_video_result(task_id)
                logger.debug(f"查询视频生成结果: {result}")
                task_status = result["task"]["status"]

                if task_status == "TASK_STATUS_SUCCEED":
                    videos = result.get("videos", [])
                    if not videos:
                        raise Exception("No video returned from API")
                    
                    video_url = videos[0]["video_url"]
                    logger.debug(f"视频生成成功: {video_url}")

                    videos_dir = os.path.join(folder_paths.get_output_directory(), "videos_utils_nodes")
                    if not os.path.exists(videos_dir):
                        os.makedirs(videos_dir)

                    video_filename = f"novita_video_{int(time.time())}.mp4"
                    video_path = os.path.join(videos_dir, video_filename)

                    response = requests.get(video_url)
                    response.raise_for_status()

                    with open(video_path, 'wb') as f:
                        f.write(response.content)

                    logger.info(f"视频已保存到: {video_path}")

                    video_input = VideoFromFile(video_path)
                    width, height = video_input.get_dimensions()
                    fps = 24.0

                    return (video_input, width, height, fps, video_url)

                elif task_status == "TASK_STATUS_FAILED":
                    logger.error(f"视频生成失败: {result}")
                    raise Exception(f"视频生成失败: {result}")

                else:
                    logger.info(f"视频生成中，task_status: {task_status}")
                time.sleep(2)

            raise Exception(f"视频生成超时 ({timeout} 秒)")

        except Exception as e:
            logger.exception(f"Novita视频生成失败: {str(e)}")
            raise e


NODE_CLASS_MAPPINGS = {
    "NovitaVideoRequestNode": NovitaVideoRequestNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NovitaVideoRequestNode": "Novita Video Request",
}
