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


class ComfyUIReplicateRun:
    """结合ComfyUI中断机制的Replicate运行器"""

    def __init__(self, timeout_seconds=300, check_interval=1.0):
        self.timeout_seconds = timeout_seconds
        self.check_interval = check_interval

    def run_with_interrupt_check(self, client, ref, input=None, **params):
        """带中断检查的replicate运行"""
        start_time = time.time()

        # 设置wait=False，手动控制轮询
        params['wait'] = False

        try:
            # 创建预测
            if hasattr(ref, 'id'):
                prediction = client.predictions.create(
                    version=ref.id, input=input or {}, **params
                )
            else:
                prediction = client.models.predictions.create(
                    model=ref, input=input or {}, **params
                )

            # 手动轮询，检查中断
            while True:
                # 检查超时
                if time.time() - start_time > self.timeout_seconds:
                    prediction.cancel()
                    raise Exception(f"timeout ({self.timeout_seconds} seconds)")

                # 检查ComfyUI中断信号
                if comfy.model_management.processing_interrupted():
                    prediction.cancel()
                    raise comfy.model_management.InterruptProcessingException(
                        "ComfyUI interrupted")

                # 检查预测状态
                prediction.reload()

                if prediction.status == "succeeded":
                    return prediction.output
                elif prediction.status == "failed":
                    raise Exception(f"prediction failed: {prediction.error}")
                elif prediction.status in ["starting", "processing"]:
                    time.sleep(self.check_interval)
                else:
                    prediction.cancel()
                    raise Exception(f"unknown status: {prediction.status}")

        except Exception as e:
            logging.error(f"Replicate operation failed: {e}")
            raise


class ReplicateRequstNode:
    def __init__(self, api_key=None):
        from replicate.client import Client
        config = get_config()
        self.api_key = api_key or config.get("REPLICATE_API_TOKEN")
        if self.api_key is not None:
            self.configure_replicate()
        self.client = Client(timeout=60)

    def configure_replicate(self):
        if self.api_key:
            os.environ["REPLICATE_API_TOKEN"] = self.api_key

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "style of 80s cyberpunk, a portrait photo", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "The random seed used for creating the noise."}),
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
                "image": ("IMAGE",),
                "timeout": ("INT", {"default": 60, "min": 1, "max": 3000}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "width", "height", "url")
    FUNCTION = "generate_image"
    CATEGORY = "utils/image"

    def generate_image(self, prompt, seed, aspect_ratio, steps, guidance, go_fast, lora_path="", lora_scale=1.0,
                       api_key="", extra_lora="", extra_lora_scale=1.0, model="black-forest-labs/flux-dev-lora",
                       num_outputs=1, image=None, timeout=60):
        # 更新API key
        if api_key.strip():
            self.api_key = api_key
            save_config({"REPLICATE_API_TOKEN": self.api_key})
            self.configure_replicate()

        if not self.api_key:
            raise ValueError(
                "API key not found in replicate_config.yml or node input")

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

            # 处理输入图像
            if image is not None and len(image) > 0:
                # 将tensor转换为PIL图像，然后保存为临时文件
                pil_image = tensor2pil(image[0])  # 取第一张图片

                # 创建临时文件
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                    pil_image.save(temp_file.name, format='PNG')
                    temp_file_path = temp_file.name

                # 使用open()创建文件对象
                input_params["input_image"] = open(temp_file_path, "rb")
                logger.debug(f"已添加输入图像文件: {temp_file_path}")

            logger.debug(f"调用Replicate API，参数: {input_params}")

            runner = ComfyUIReplicateRun(timeout_seconds=timeout, check_interval=1.0)
            # 调用Replicate API
            output = runner.run_with_interrupt_check(self.client, model, input=input_params)

            # 清理临时文件
            if image is not None and len(image) > 0:
                try:
                    input_params["input_image"].close()
                    os.unlink(temp_file_path)
                except:
                    pass

            images = []
            urls = []
            if not isinstance(output, list):
                output = [output]
            for image_url in output:
                logger.debug(f"生成的图片URL: {image_url}")
                urls.append(str(image_url))
                response = requests.get(image_url)
                response.raise_for_status()
                image = Image.open(io.BytesIO(response.content))
                width, height = image.size
                image_array = np.array(image)
                if len(image_array.shape) == 3 and image_array.shape[2] == 4:
                    image_array = image_array[:, :, :3]
                images.append(image_array)

            image_tensor = np2tensor(images)
            urls_str = ",".join(urls)

            return (image_tensor, width, height, urls_str)

        except Exception as e:
            logger.exception(f"Replicate API调用失败: {str(e)}")
            raise e


class ReplicateVideoRequestNode:
    def __init__(self, api_key=None):
        from replicate.client import Client
        config = get_config()
        self.api_key = api_key or config.get("REPLICATE_API_TOKEN")
        if self.api_key is not None:
            self.configure_replicate()
        self.client = Client(timeout=300)

    def configure_replicate(self):
        if self.api_key:
            os.environ["REPLICATE_API_TOKEN"] = self.api_key

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {                
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model": ("STRING", {"default": "wan-video/wan-2.2-i2v-fast"}),
                "num_frames": ("INT", {"default": 81, "min": 81, "max": 121}),
                "resolution": (["480p", "720p"], {"default": "720p"}),
                "frames_per_second": ("INT", {"default": 16, "min": 5, "max": 30, "step": 1}),
            },
            "optional": {
                "image": ("IMAGE",),
                "go_fast": ("BOOLEAN", {"default": True}),
                "sample_shift": ("FLOAT", {"default": 12.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "lora_weights_transformer": ("STRING", {"default": ""}),
                "lora_scale_transformer": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "lora_weights_transformer_2": ("STRING", {"default": ""}),
                "lora_scale_transformer_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "api_key": ("STRING", {"default": ""}),
                "timeout": ("INT", {"default": 300, "min": 1, "max": 3000}),
            }
        }

    RETURN_TYPES = (IO.VIDEO, "INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("video", "width", "height", "fps", "url")
    FUNCTION = "generate_video"
    CATEGORY = "utils/video"

    def generate_video(self, prompt, model, num_frames, resolution, frames_per_second, image=None,
                      go_fast=True, sample_shift=12.0, lora_weights_transformer="", 
                      lora_scale_transformer=1.0, lora_weights_transformer_2="", 
                      lora_scale_transformer_2=1.0, api_key="", timeout=300):
        
        if api_key.strip():
            self.api_key = api_key
            save_config({"REPLICATE_API_TOKEN": self.api_key})
            self.configure_replicate()

        if not self.api_key:
            raise ValueError("API key not found in replicate_config.yml or node input")

        try:
            input_params = {                
                "prompt": prompt,
                "num_frames": num_frames,
                "resolution": resolution,
                "frames_per_second": frames_per_second,
                "go_fast": go_fast,
                "sample_shift": sample_shift,
            }

            if image is not None and len(image) > 0:
                pil_image = tensor2pil(image[0])
                
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                    pil_image.save(temp_file.name, format='PNG')
                    temp_file_path = temp_file.name
                
                input_params["image"] = open(temp_file_path, "rb")

            if lora_weights_transformer.strip():
                input_params["lora_weights_transformer"] = lora_weights_transformer
                input_params["lora_scale_transformer"] = lora_scale_transformer

            if lora_weights_transformer_2.strip():
                input_params["lora_weights_transformer_2"] = lora_weights_transformer_2
                input_params["lora_scale_transformer_2"] = lora_scale_transformer_2

            logger.debug(f"调用Replicate API生成视频，参数: {input_params}")

            runner = ComfyUIReplicateRun(timeout_seconds=timeout, check_interval=1.0)
            output = runner.run_with_interrupt_check(self.client, model, input=input_params)

            if image is not None and len(image) > 0:
                try:
                    input_params["image"].close()
                    os.unlink(temp_file_path)
                except:
                    pass

            if not isinstance(output, list):
                output = [output]

            video_url = output[0] if output else None
            if not video_url:
                raise Exception("No video URL returned from API")

            logger.debug(f"生成的视频URL: {video_url}")

            videos_dir = os.path.join(folder_paths.get_output_directory(), "videos_utils_nodes")
            if not os.path.exists(videos_dir):
                os.makedirs(videos_dir)

            video_filename = f"replicate_video_{int(time.time())}.mp4"
            video_path = os.path.join(videos_dir, video_filename)

            response = requests.get(video_url)
            response.raise_for_status()

            with open(video_path, 'wb') as f:
                f.write(response.content)

            logger.info(f"视频已保存到: {video_path}")

            video_input = VideoFromFile(video_path)
            width, height = video_input.get_dimensions()
            fps = float(frames_per_second)

            return (video_input, width, height, fps, video_url)

        except Exception as e:
            logger.exception(f"Replicate视频生成失败: {str(e)}")
            raise e


NODE_CLASS_MAPPINGS = {
    "ReplicateRequstNode": ReplicateRequstNode,
    "ReplicateVideoRequestNode": ReplicateVideoRequestNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ReplicateVideoRequestNode": "Replicate Video Request",
    "ReplicateRequstNode": "Replicate Image Request",
}

