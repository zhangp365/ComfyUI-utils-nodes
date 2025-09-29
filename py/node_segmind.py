import time
import comfy.model_management
import requests
import yaml
import logging
import folder_paths
import os
import sys
sys.path.append(".")
from comfy_api.latest._input_impl.video_types import VideoFromFile
from comfy.comfy_types import IO, FileLocator, ComfyNodeABC

logger = logging.getLogger(__name__)

config_dir = os.path.join(folder_paths.base_path, "config")
if not os.path.exists(config_dir):
    os.makedirs(config_dir)


def get_segmind_config():
    try:
        config_path = os.path.join(config_dir, 'segmind_config.yml')
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config
    except:
        return {}


def save_segmind_config(config):
    config_path = os.path.join(config_dir, 'segmind_config.yml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, indent=4)




class ComfyUISegmindRun:
    """结合ComfyUI中断机制的Segmind运行器"""

    def __init__(self, timeout_seconds=300, check_interval=1.0):
        self.timeout_seconds = timeout_seconds
        self.check_interval = check_interval

    def run_with_interrupt_check(self, url, data, headers):
        """带中断检查的Segmind运行"""
        start_time = time.time()

        try:
            logger.info(f"请求Segmind API: {url}")
            
            # 使用线程来运行Segmind请求，以便可以检查中断
            import threading
            import queue
            
            result_queue = queue.Queue()
            exception_queue = queue.Queue()
            
            def run_segmind():
                try:
                    response = requests.post(url, json=data, headers=headers, timeout=self.timeout_seconds)
                    response.raise_for_status()
                    result_queue.put(response)
                except Exception as e:
                    exception_queue.put(e)
            
            # 启动Segmind请求线程
            segmind_thread = threading.Thread(target=run_segmind)
            segmind_thread.daemon = True
            segmind_thread.start()
            
            # 轮询检查中断和超时
            while segmind_thread.is_alive():
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
                raise Exception("Segmind request failed to return result")

        except Exception as e:
            logging.error(f"Segmind operation failed: {e}")
            raise


class SegmindVideoRequestNode:
    def __init__(self, api_key=None):
        config = get_segmind_config()
        self.api_key = api_key or config.get("SEGMIND_API_KEY")
        if self.api_key is not None:
            self.configure_segmind()

    def configure_segmind(self):
        if self.api_key:
            os.environ["SEGMIND_API_KEY"] = self.api_key

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("STRING", {"default": "","multiline": True, "tooltip": "输入图像URL"}),
                "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "视频描述"}),
                "model": ("STRING", {"default": "wan-2.2-i2v-fast", "tooltip": "Segmind模型名称"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": "", "tooltip": "Segmind API密钥"}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "随机种子"}),
                "go_fast": ("BOOLEAN", {"default": True, "tooltip": "快速生成模式"}),
                "num_frames": ("INT", {"default": 81, "min": 81, "max": 100, "tooltip": "视频帧数"}),
                "resolution": (["480p","720p"], {"default": "480p", "tooltip": "视频分辨率"}),
                "aspect_ratio": (["16:9", "9:16"], {"default": "16:9", "tooltip": "宽高比"}),
                "sample_shift": ("FLOAT", {"default": 12.0, "min": 1.0, "max": 20.0, "step": 0.1, "tooltip": "图像采样偏移"}),
                "frames_per_second": ("INT", {"default": 16, "min": 5, "max": 24, "tooltip": "帧率"}),
                "timeout": ("INT", {"default": 300, "min": 1, "max": 3000, "tooltip": "超时时间(秒)"}),
            }
        }

    RETURN_TYPES = (IO.VIDEO, "INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("video", "width", "height", "fps", "url")
    FUNCTION = "generate_video"
    CATEGORY = "utils/video"

    def generate_video(self, image, prompt, model, api_key="", seed=1, go_fast=True, 
                      num_frames=81, resolution="480p", aspect_ratio="16:9", 
                      sample_shift=12.0, frames_per_second=16, timeout=300):
        
        if api_key.strip():
            self.api_key = api_key
            save_segmind_config({"SEGMIND_API_KEY": self.api_key})
            self.configure_segmind()

        if not self.api_key:
            raise ValueError("API key not found in segmind_config.yml or node input")

        try:
            # 验证输入图像URL
            if not image or not image.strip():
                raise ValueError("输入图像URL不能为空")

            # 准备请求参数
            data = {
                "image": image.strip(),
                "prompt": prompt,
                "seed": seed,
                "go_fast": go_fast,
                "num_frames": num_frames,
                "resolution": resolution,
                "aspect_ratio": aspect_ratio,
                "sample_shift": sample_shift,
                "frames_per_second": frames_per_second
            }

            # 构建API URL
            url = f"https://api.segmind.com/v1/{model}"
            headers = {'x-api-key': self.api_key}

            logger.debug(f"调用Segmind API，参数: {data}")

            runner = ComfyUISegmindRun(timeout_seconds=timeout, check_interval=1.0)
            response = runner.run_with_interrupt_check(url, data, headers)

            logger.info(f"Segmind API调用成功，状态码: {response}")
            
            # 检查响应内容类型
            content_type = response.headers.get('content-type', '')
            if 'application/json' in content_type:
                # JSON响应，可能包含错误信息
                result = response.json()
                if 'error' in result:
                    raise Exception(f"Segmind API错误: {result['error']}")
                # 如果成功，视频URL可能在result中
                video_url = result.get('video_url') or result.get('url')
                if not video_url:
                    raise Exception("API响应中未找到视频URL")
            else:
                # 直接返回视频文件
                video_url = "segmind_generated_video"

            # 创建视频保存目录
            videos_dir = os.path.join(folder_paths.get_output_directory(), "videos_segmind")
            if not os.path.exists(videos_dir):
                os.makedirs(videos_dir)

            # 保存视频
            video_filename = f"segmind_video_{int(time.time())}.mp4"
            video_path = os.path.join(videos_dir, video_filename)

            if video_url == "segmind_generated_video":
                # 直接保存响应内容
                with open(video_path, 'wb') as f:
                    f.write(response.content)
            else:
                # 从URL下载视频
                video_response = requests.get(video_url, timeout=60)
                video_response.raise_for_status()
                with open(video_path, 'wb') as f:
                    f.write(video_response.content)

            logger.info(f"视频已保存到: {video_path}")

            # 创建视频对象
            video_input = VideoFromFile(video_path)
            width, height = video_input.get_dimensions()
            fps = float(frames_per_second)

            return (video_input, width, height, fps, video_url)

        except Exception as e:
            logger.exception(f"Segmind视频生成失败: {str(e)}")
            raise e


NODE_CLASS_MAPPINGS = {
    "SegmindVideoRequestNode": SegmindVideoRequestNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SegmindVideoRequestNode": "Segmind Video Request",
}
