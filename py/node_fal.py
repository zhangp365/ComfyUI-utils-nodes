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
import fal_client
sys.path.append(".")
from comfy_api.latest._input_impl.video_types import VideoFromFile
from comfy.comfy_types import IO, FileLocator, ComfyNodeABC
from .utils import tensor2pil, np2tensor

logger = logging.getLogger(__name__)

config_dir = os.path.join(folder_paths.base_path, "config")
if not os.path.exists(config_dir):
    os.makedirs(config_dir)


def get_fal_config():
    try:
        config_path = os.path.join(config_dir, 'fal_config.yml')
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config
    except:
        return {}


def save_fal_config(config):
    config_path = os.path.join(config_dir, 'fal_config.yml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, indent=4)


class ComfyUIFalRun:
    """结合ComfyUI中断机制的FAL运行器"""

    def __init__(self, timeout_seconds=300, check_interval=1.0):
        self.timeout_seconds = timeout_seconds
        self.check_interval = check_interval

    def run_with_interrupt_check(self, model, arguments, on_queue_update=None):
        """带中断检查的fal运行"""
        start_time = time.time()

        try:
            logger.info(f"请求fal.ai, 调用模型: {model}")
            
            # 使用线程来运行FAL请求，以便可以检查中断
            import threading
            import queue
            
            result_queue = queue.Queue()
            exception_queue = queue.Queue()
            
            def run_fal():
                try:
                    result = fal_client.subscribe(
                        model,
                        arguments=arguments,
                        with_logs=True,
                        on_queue_update=on_queue_update,
                    )
                    result_queue.put(result)
                except Exception as e:
                    exception_queue.put(e)
            
            # 启动FAL请求线程
            fal_thread = threading.Thread(target=run_fal)
            fal_thread.daemon = True
            fal_thread.start()
            
            # 轮询检查中断和超时
            while fal_thread.is_alive():
                # 检查超时
                if time.time() - start_time > self.timeout_seconds:
                    raise Exception(f"timeout ({self.timeout_seconds} seconds)")

                # 检查ComfyUI中断信号
                if comfy.model_management.processing_interrupted():
                    raise comfy.model_management.InterruptProcessingException(
                        "ComfyUI interrupted")
                
                time.sleep(self.check_interval)
            
            # 检查是否有异常
            if not exception_queue.empty():
                raise exception_queue.get()
            
            # 获取结果
            if not result_queue.empty():
                return result_queue.get()
            else:
                raise Exception("FAL request failed to return result")

        except Exception as e:
            logging.error(f"FAL operation failed: {e}")
            raise


def on_queue_update(update):
    """队列更新回调函数"""
    if isinstance(update, fal_client.InProgress):
        for log in update.logs:
            logger.debug(f"FAL日志: {log['message']}")


class FalVideoRequestNode:
    def __init__(self, api_key=None):
        config = get_fal_config()
        self.api_key = api_key or config.get("FAL_KEY")
        if self.api_key is not None:
            self.configure_fal()

    def configure_fal(self):
        if self.api_key:
            os.environ["FAL_KEY"] = self.api_key

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "输入图像"}),
                "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "视频描述"}),
                "model": ("STRING", {"default": "fal-ai/wan/v2.2-a14b/image-to-video/turbo", "tooltip": "FAL模型名称"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": "", "tooltip": "FAL API密钥"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "随机种子"}),
                "resolution": (["480p", "580p", "720p"], {"default": "720p", "tooltip": "视频分辨率"}),
                "aspect_ratio": (["auto", "16:9", "9:16", "1:1"], {"default": "auto", "tooltip": "宽高比"}),
                "enable_safety_checker": ("BOOLEAN", {"default": False, "tooltip": "启用安全检查"}),
                "enable_prompt_expansion": ("BOOLEAN", {"default": False, "tooltip": "启用提示扩展"}),
                "acceleration": (["none", "regular"], {"default": "regular", "tooltip": "加速模式"}),
                "video_quality": (["low", "medium", "high", "maximum"], {"default": "high", "tooltip": "视频质量"}),
                "video_write_mode": (["fast", "balanced", "small"], {"default": "balanced", "tooltip": "视频写入模式"}),
                "end_image_url": ("STRING", {"default": "", "tooltip": "结束图像URL"}),
                "timeout": ("INT", {"default": 300, "min": 1, "max": 3000, "tooltip": "超时时间(秒)"}),
            }
        }

    RETURN_TYPES = (IO.VIDEO, "INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("video", "width", "height", "fps", "url")
    FUNCTION = "generate_video"
    CATEGORY = "utils/video"

    def generate_video(self, image, prompt, model, api_key="", seed=0, resolution="720p", 
                      aspect_ratio="auto", enable_safety_checker=False, 
                      enable_prompt_expansion=False, acceleration="regular", 
                      video_quality="high", video_write_mode="balanced", 
                      end_image_url="", timeout=300):
        
        if api_key.strip():
            self.api_key = api_key
            save_fal_config({"FAL_KEY": self.api_key})
            self.configure_fal()

        if not self.api_key:
            raise ValueError("API key not found in fal_config.yml or node input")

        try:
            # 处理输入图像
            if image is not None and len(image) > 0:
                pil_image = tensor2pil(image[0])
                
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                    pil_image.save(temp_file.name, format='PNG')
                    temp_file_path = temp_file.name
                
                # 上传图像到FAL
                input_image_object = fal_client.upload_file(temp_file_path)
                logger.info(f"图像已上传到FAL: {input_image_object}")
                
                # 清理临时文件
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
            else:
                raise ValueError("输入图像不能为空")

            # 准备请求参数
            arguments = {
                "image_url": input_image_object,
                "prompt": prompt,
                "resolution": resolution,
                "aspect_ratio": aspect_ratio,
                "enable_safety_checker": enable_safety_checker,
                "enable_prompt_expansion": enable_prompt_expansion,
                "acceleration": acceleration,
                "video_quality": video_quality,
                "video_write_mode": video_write_mode
            }

            # 添加可选参数
            if seed > 0:
                arguments["seed"] = seed
            if end_image_url.strip():
                arguments["end_image_url"] = end_image_url

            logger.debug(f"调用FAL API，参数: {arguments}")

            runner = ComfyUIFalRun(timeout_seconds=timeout, check_interval=1.0)
            result = runner.run_with_interrupt_check(
                model, 
                arguments, 
                on_queue_update
            )

            logger.info(f"FAL API调用成功，输出: {result}")
            
            video_url = result["video"]["url"]
            logger.debug(f"生成的视频URL: {video_url}")

            # 创建视频保存目录
            videos_dir = os.path.join(folder_paths.get_output_directory(), "videos_fal")
            if not os.path.exists(videos_dir):
                os.makedirs(videos_dir)

            # 下载视频
            video_filename = f"fal_video_{int(time.time())}.mp4"
            video_path = os.path.join(videos_dir, video_filename)

            response = requests.get(video_url, timeout=60)
            response.raise_for_status()

            with open(video_path, 'wb') as f:
                f.write(response.content)

            logger.info(f"视频已保存到: {video_path}")

            # 创建视频对象
            video_input = VideoFromFile(video_path)
            width, height = video_input.get_dimensions()
            fps = 24.0  # FAL默认帧率

            return (video_input, width, height, fps, video_url)

        except Exception as e:
            logger.exception(f"FAL视频生成失败: {str(e)}")
            raise e


NODE_CLASS_MAPPINGS = {
    "FalVideoRequestNode": FalVideoRequestNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FalVideoRequestNode": "FAL Video Request",
}
