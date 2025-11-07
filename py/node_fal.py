import time
import comfy.model_management
import tempfile
import io
from io import BytesIO
from PIL import Image
import requests
import numpy as np
import yaml
import logging
import folder_paths
import os
import sys
import torch

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
        import fal_client
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
    import fal_client
    if isinstance(update, fal_client.InProgress):
        for log in update.logs:
            logger.debug(f"FAL日志: {log['message']}")


class BaseFalNode:
    """FAL API 基础节点类，提供通用的 API 调用功能"""
    
    def __init__(self, api_key=None):
        config = get_fal_config()
        self.api_key = api_key or config.get("FAL_KEY")
        if self.api_key is not None:
            self.configure_fal()

    def configure_fal(self):
        if self.api_key:
            os.environ["FAL_KEY"] = self.api_key

    def _upload_image(self, image):
        """上传图像到FAL并返回文件对象"""
        import fal_client
        if image is not None and len(image) > 0:
            pil_image = tensor2pil(image[0])
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                pil_image.save(temp_file.name, format='PNG')
                temp_file_path = temp_file.name
            
            input_image_object = fal_client.upload_file(temp_file_path)
            logger.info(f"图像已上传到FAL: {input_image_object}")
            
            try:
                os.unlink(temp_file_path)
            except:
                pass
                
            return input_image_object
        else:
            raise ValueError("输入图像不能为空")
            
    def _upload_video(self, video_name):
        """上传视频到FAL并返回文件对象"""
        import fal_client
        if video_name and video_name.strip():
            # 从 ComfyUI 的 input 目录拼接完整路径
            input_dir = folder_paths.get_input_directory()
            video_path = os.path.join(input_dir, video_name)
            
            if os.path.exists(video_path):
                input_video_url = fal_client.upload_file(video_path)
                logger.info(f"视频已上传到FAL: {input_video_url}")
                return input_video_url
            else:
                raise ValueError(f"视频文件不存在: {video_path}")
        else:
            raise ValueError("视频文件名不能为空")

    def _call_fal_api(self, model, arguments, timeout=300):
        """调用FAL API的通用方法"""
        runner = ComfyUIFalRun(timeout_seconds=timeout, check_interval=1.0)
        return runner.run_with_interrupt_check(model, arguments, on_queue_update)

    def _download_and_save_video(self, video_url, subfolder="videos_fal"):
        """下载并保存视频文件"""
        videos_dir = os.path.join(folder_paths.get_output_directory(), subfolder)
        if not os.path.exists(videos_dir):
            os.makedirs(videos_dir)

        video_filename = f"fal_video_{int(time.time())}.mp4"
        video_path = os.path.join(videos_dir, video_filename)

        response = requests.get(video_url, timeout=60)
        response.raise_for_status()

        with open(video_path, 'wb') as f:
            f.write(response.content)

        logger.info(f"视频已保存到: {video_path}")
        return video_path

    def _create_video_object(self, video_path, fps=24.0):
        """创建视频对象并返回相关信息"""
        video_input = VideoFromFile(video_path)
        width, height = video_input.get_dimensions()
        return video_input, width, height, fps

    def _download_and_convert_image(self, image_url):
        """下载图片并转换为ComfyUI的IMAGE格式"""
        response = requests.get(image_url, timeout=60)
        response.raise_for_status()
        
        image_bytes = BytesIO(response.content)
        pil_image = Image.open(image_bytes).convert("RGB")
        width, height = pil_image.size
        
        image_array = np.array(pil_image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).unsqueeze(0)
        
        return image_tensor, width, height


class FalImage2VideoRequestNode(BaseFalNode):
    """FAL 图像转视频请求节点"""
    
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
            input_image_object = self._upload_image(image)

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

            if seed > 0:
                arguments["seed"] = seed
            if end_image_url.strip():
                arguments["end_image_url"] = end_image_url

            logger.debug(f"调用FAL API，参数: {arguments}")

            result = self._call_fal_api(model, arguments, timeout)
            logger.info(f"FAL API调用成功，输出: {result}")
            
            video_url = result["video"]["url"]
            logger.debug(f"生成的视频URL: {video_url}")

            video_path = self._download_and_save_video(video_url)
            video_input, width, height, fps = self._create_video_object(video_path)

            return (video_input, width, height, fps, video_url)

        except Exception as e:
            logger.exception(f"FAL视频生成失败: {str(e)}")
            raise e


class FalVideo2VideoRequestNode(BaseFalNode):
    """FAL 视频转视频请求节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_name": ("STRING", {"default": "", "tooltip": "输入视频文件名（位于input目录）"}),
                "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "视频描述"}),
                "model": ("STRING", {"default": "fal-ai/wan-22-vace-fun-a14b/pose", "tooltip": "FAL模型名称"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": "", "tooltip": "FAL API密钥"}),
                "negative_prompt": ("STRING", {"default": "letterboxing, borders, black bars, bright colors, overexposed, static, blurred details, subtitles, style, artwork, painting, picture, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, still picture, cluttered background, three legs, many people in the background, walking backwards", "multiline": True, "tooltip": "负面提示词"}),
                "match_input_num_frames": ("BOOLEAN", {"default": True, "tooltip": "匹配输入视频帧数"}),
                "num_frames": ("INT", {"default": 81, "min": 81, "max": 241, "tooltip": "生成帧数"}),
                "match_input_frames_per_second": ("BOOLEAN", {"default": True, "tooltip": "匹配输入视频帧率"}),
                "frames_per_second": ("INT", {"default": 16, "min": 5, "max": 30, "tooltip": "视频帧率"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "随机种子"}),
                "resolution": (["auto", "240p", "360p", "480p", "580p", "720p"], {"default": "auto", "tooltip": "视频分辨率"}),
                "aspect_ratio": (["auto", "16:9", "1:1", "9:16"], {"default": "auto", "tooltip": "宽高比"}),
                "num_inference_steps": ("INT", {"default": 30, "min": 1, "max": 100, "tooltip": "推理步数"}),
                "guidance_scale": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 20.0, "step": 0.1, "tooltip": "引导强度"}),
                "shift": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip": "偏移参数"}),
                "ref_image": ("IMAGE", {"tooltip": "参考图像"}),
                "ref_image_urls": ("STRING", {"default": "", "multiline": True, "tooltip": "参考图像URL列表，每行一个"}),
                "first_frame": ("IMAGE", {"tooltip": "首帧图像"}),
                "last_frame": ("IMAGE", {"tooltip": "末帧图像"}),
                "enable_safety_checker": ("BOOLEAN", {"default": False, "tooltip": "启用安全检查"}),
                "enable_prompt_expansion": ("BOOLEAN", {"default": False, "tooltip": "启用提示扩展"}),
                "preprocess": ("BOOLEAN", {"default": True, "tooltip": "预处理输入视频"}),
                "acceleration": (["none", "regular"], {"default": "regular", "tooltip": "加速模式"}),
                "video_quality": (["low", "medium", "high", "maximum"], {"default": "high", "tooltip": "视频质量"}),
                "video_write_mode": (["fast", "balanced", "small"], {"default": "balanced", "tooltip": "视频写入模式"}),
                "num_interpolated_frames": ("INT", {"default": 0, "min": 0, "max": 100, "tooltip": "插值帧数"}),
                "temporal_downsample_factor": ("INT", {"default": 0, "min": 0, "max": 10, "tooltip": "时间下采样因子"}),
                "enable_auto_downsample": ("BOOLEAN", {"default": False, "tooltip": "启用自动下采样"}),
                "auto_downsample_min_fps": ("FLOAT", {"default": 15.0, "min": 1.0, "max": 30.0, "step": 0.1, "tooltip": "自动下采样最小帧率"}),
                "interpolator_model": (["rife", "film"], {"default": "film", "tooltip": "插值模型"}),
                "sync_mode": ("BOOLEAN", {"default": False, "tooltip": "同步模式"}),
                "timeout": ("INT", {"default": 300, "min": 1, "max": 3000, "tooltip": "超时时间(秒)"}),
            }
        }

    RETURN_TYPES = (IO.VIDEO, "INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("video", "width", "height", "fps", "url")
    FUNCTION = "generate_video"
    CATEGORY = "utils/video"


    def generate_video(self, video_name, prompt, model, api_key="", negative_prompt="", 
                      match_input_num_frames=True, num_frames=81, 
                      match_input_frames_per_second=True, frames_per_second=16,
                      seed=0, resolution="auto", aspect_ratio="auto",
                      num_inference_steps=30, guidance_scale=5.0, shift=5.0,
                      ref_image=None, ref_image_urls="", first_frame=None, last_frame=None,
                      enable_safety_checker=False, enable_prompt_expansion=False,
                      preprocess=True, acceleration="regular", video_quality="high",
                      video_write_mode="balanced", num_interpolated_frames=0,
                      temporal_downsample_factor=0, enable_auto_downsample=False,
                      auto_downsample_min_fps=15.0, interpolator_model="film",
                      sync_mode=False, timeout=300):
        
        if api_key.strip():
            self.api_key = api_key
            save_fal_config({"FAL_KEY": self.api_key})
            self.configure_fal()

        if not self.api_key:
            raise ValueError("API key not found in fal_config.yml or node input")

        try:
            input_video_url = self._upload_video(video_name)

            arguments = {
                "video_url": input_video_url,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "match_input_num_frames": match_input_num_frames,
                "num_frames": num_frames,
                "match_input_frames_per_second": match_input_frames_per_second,
                "frames_per_second": frames_per_second,
                "resolution": resolution,
                "aspect_ratio": aspect_ratio,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "shift": shift,
                "enable_safety_checker": enable_safety_checker,
                "enable_prompt_expansion": enable_prompt_expansion,
                "preprocess": preprocess,
                "acceleration": acceleration,
                "video_quality": video_quality,
                "video_write_mode": video_write_mode,
                "num_interpolated_frames": num_interpolated_frames,
                "temporal_downsample_factor": temporal_downsample_factor,
                "enable_auto_downsample": enable_auto_downsample,
                "auto_downsample_min_fps": auto_downsample_min_fps,
                "interpolator_model": interpolator_model,
                "sync_mode": sync_mode
            }

            if seed > 0:
                arguments["seed"] = seed
            
            # 处理参考图片URL列表
            ref_urls = []
            if ref_image is not None and len(ref_image) > 0:
                ref_image_url = self._upload_image(ref_image)
                ref_urls.append(ref_image_url)
            if ref_image_urls.strip():
                additional_urls = [url.strip() for url in ref_image_urls.split('\n') if url.strip()]
                ref_urls.extend(additional_urls)
            if ref_urls:
                arguments["ref_image_urls"] = ref_urls
                
            if first_frame is not None and len(first_frame) > 0:
                first_frame_object = self._upload_image(first_frame)
                arguments["first_frame_url"] = first_frame_object
            if last_frame is not None and len(last_frame) > 0:
                last_frame_object = self._upload_image(last_frame)
                arguments["last_frame_url"] = last_frame_object

            logger.debug(f"调用FAL API，参数: {arguments}")

            result = self._call_fal_api(model, arguments, timeout)
            logger.info(f"FAL API调用成功，输出: {result}")
            
            video_url = result["video"]["url"]
            logger.debug(f"生成的视频URL: {video_url}")

            video_path = self._download_and_save_video(video_url, "videos_fal_v2v")
            video_input, width, height, fps = self._create_video_object(video_path)

            return (video_input, width, height, fps, video_url)

        except Exception as e:
            logger.exception(f"FAL视频转视频生成失败: {str(e)}")
            raise e


class FalFunControlVideoRequestNode(BaseFalNode):
    """FAL 控制视频生成请求节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "视频生成提示词"}),
                "video_name": ("STRING", {"default": "", "tooltip": "输入视频文件名（位于input目录）"}),
                "model": ("STRING", {"default": "fal-ai/wan-fun-control", "tooltip": "FAL模型名称"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": "", "tooltip": "FAL API密钥"}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "负面提示词"}),
                "num_inference_steps": ("INT", {"default": 27, "min": 1, "max": 100, "tooltip": "推理步数"}),
                "guidance_scale": ("FLOAT", {"default": 6.0, "min": 1.0, "max": 20.0, "step": 0.1, "tooltip": "引导强度"}),
                "shift": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip": "偏移参数"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "随机种子"}),
                "match_input_num_frames": ("BOOLEAN", {"default": True, "tooltip": "匹配输入视频帧数"}),
                "num_frames": ("INT", {"default": 81, "min": 1, "max": 1000, "tooltip": "生成帧数"}),
                "match_input_fps": ("BOOLEAN", {"default": True, "tooltip": "匹配输入视频帧率"}),
                "fps": ("INT", {"default": 16, "min": 1, "max": 60, "tooltip": "视频帧率"}),
                "preprocess_video": ("BOOLEAN", {"default": False, "tooltip": "预处理视频"}),
                "preprocess_type": (["depth", "pose"], {"default": "depth", "tooltip": "预处理类型"}),
                "ref_image": ("IMAGE", {"tooltip": "参考图像"}),
                "timeout": ("INT", {"default": 300, "min": 1, "max": 3000, "tooltip": "超时时间(秒)"}),
            }
        }

    RETURN_TYPES = (IO.VIDEO, "INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("video", "width", "height", "fps", "url")
    FUNCTION = "generate_video"
    CATEGORY = "utils/video"

    def generate_video(self, prompt, video_name, model, api_key="", negative_prompt="",
                      num_inference_steps=27, guidance_scale=6.0, shift=5.0, seed=0,
                      match_input_num_frames=True, num_frames=81, match_input_fps=True, fps=16,
                      preprocess_video=False, preprocess_type="depth", ref_image=None, timeout=300):
        
        if api_key.strip():
            self.api_key = api_key
            save_fal_config({"FAL_KEY": self.api_key})
            self.configure_fal()

        if not self.api_key:
            raise ValueError("API key not found in fal_config.yml or node input")

        try:
            input_video_url = self._upload_video(video_name)

            arguments = {
                "prompt": prompt,
                "control_video_url": input_video_url,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "shift": shift,
                "match_input_num_frames": match_input_num_frames,
                "num_frames": num_frames,
                "match_input_fps": match_input_fps,
                "fps": fps,
                "preprocess_video": preprocess_video,
                "preprocess_type": preprocess_type
            }

            if negative_prompt.strip():
                arguments["negative_prompt"] = negative_prompt
            if seed > 0:
                arguments["seed"] = seed
            if ref_image is not None and len(ref_image) > 0:
                ref_image_url = self._upload_image(ref_image)
                arguments["reference_image_url"] = ref_image_url

            logger.info(f"调用FAL API，参数: {arguments}")

            result = self._call_fal_api(model, arguments, timeout)
            logger.info(f"FAL API调用成功，输出: {result}")
            
            video_url = result["video"]["url"]
            logger.debug(f"生成的视频URL: {video_url}")

            video_path = self._download_and_save_video(video_url, "videos_fal_control")
            video_input, width, height, fps = self._create_video_object(video_path)

            return (video_input, width, height, fps, video_url)

        except Exception as e:
            logger.exception(f"FAL控制视频生成失败: {str(e)}")
            raise e


class QwenEditPlusLoraNode(BaseFalNode):
    """FAL Qwen Image Edit Plus LoRA 节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image (first, required)"}),
                "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Prompt for image generation"}),
                "model": ("STRING", {"default": "fal-ai/qwen-image-edit-plus-lora", "tooltip": "FAL model name"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": "", "tooltip": "FAL API key"}),
                "image_2": ("IMAGE", {"tooltip": "Input image (second, optional)"}),
                "image_3": ("IMAGE", {"tooltip": "Input image (third, optional)"}),
                "image_4": ("IMAGE", {"tooltip": "Input image (fourth, optional)"}),
                "image_size": (["auto", "square_hd", "square", "portrait_4_3", "portrait_16_9", "landscape_4_3", "landscape_16_9", "custom"], {"default": "auto", "tooltip": "Image size"}),
                "custom_width": ("INT", {"default": 1280, "min": 1, "max": 4096, "tooltip": "Custom width (used when image_size is custom)"}),
                "custom_height": ("INT", {"default": 720, "min": 1, "max": 4096, "tooltip": "Custom height (used when image_size is custom)"}),
                "num_inference_steps": ("INT", {"default": 28, "min": 1, "max": 100, "tooltip": "Number of inference steps"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "Random seed"}),
                "guidance_scale": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 20.0, "step": 0.1, "tooltip": "Guidance scale"}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 10, "tooltip": "Number of images to generate"}),
                "enable_safety_checker": ("BOOLEAN", {"default": False, "tooltip": "Enable safety checker"}),
                "output_format": (["jpeg", "png"], {"default": "png", "tooltip": "Output format"}),
                "negative_prompt": ("STRING", {"default": " ", "multiline": True, "tooltip": "Negative prompt"}),
                "acceleration": (["none", "regular"], {"default": "regular", "tooltip": "Acceleration mode"}),
                "lora1_url": ("STRING", {"default": "", "tooltip": "LoRA 1 URL"}),
                "lora1_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1, "tooltip": "LoRA 1 weight"}),
                "lora2_url": ("STRING", {"default": "", "tooltip": "LoRA 2 URL"}),
                "lora2_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1, "tooltip": "LoRA 2 weight"}),
                "lora3_url": ("STRING", {"default": "", "tooltip": "LoRA 3 URL"}),
                "lora3_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1, "tooltip": "LoRA 3 weight"}),
                "timeout": ("INT", {"default": 300, "min": 1, "max": 3000, "tooltip": "Timeout in seconds"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "generate_image"
    CATEGORY = "utils/image"

    def generate_image(self, image, prompt, model, api_key="", image_2=None, image_3=None, image_4=None,
                      image_size="auto", custom_width=1280, custom_height=720, num_inference_steps=28, seed=0,
                      guidance_scale=4.0, num_images=1, enable_safety_checker=False,
                      output_format="png", negative_prompt=" ", acceleration="regular",
                      lora1_url="", lora1_weight=1.0, lora2_url="", lora2_weight=1.0,
                      lora3_url="", lora3_weight=1.0, timeout=300):
        
        if api_key.strip():
            self.api_key = api_key
            save_fal_config({"FAL_KEY": self.api_key})
            self.configure_fal()

        if not self.api_key:
            raise ValueError("API key not found in fal_config.yml or node input")

        try:
            url_list = []
            
            image_url = self._upload_image(image)
            url_list.append(image_url)
            
            if image_2 is not None and len(image_2) > 0:
                image2_url = self._upload_image(image_2)
                url_list.append(image2_url)
            
            if image_3 is not None and len(image_3) > 0:
                image3_url = self._upload_image(image_3)
                url_list.append(image3_url)
            
            if image_4 is not None and len(image_4) > 0:
                image4_url = self._upload_image(image_4)
                url_list.append(image4_url)

            arguments = {
                "prompt": prompt,
                "image_urls": url_list,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "sync_mode": False,
                "num_images": num_images,
                "enable_safety_checker": enable_safety_checker,
                "output_format": output_format,
                "negative_prompt": negative_prompt,
                "acceleration": acceleration
            }

            if seed > 0:
                arguments["seed"] = seed

            if image_size == "custom":
                arguments["image_size"] = {
                    "width": custom_width,
                    "height": custom_height
                }
            elif image_size != "auto":
                arguments["image_size"] = image_size

            lora_list = []
            if lora1_url.strip():
                lora_list.append({"path": lora1_url.strip(), "scale": lora1_weight})
            if lora2_url.strip():
                lora_list.append({"path": lora2_url.strip(), "scale": lora2_weight})
            if lora3_url.strip():
                lora_list.append({"path": lora3_url.strip(), "scale": lora3_weight})
            
            if lora_list:
                arguments["loras"] = lora_list

            logger.debug(f"调用FAL API，参数: {arguments}")

            result = self._call_fal_api(model, arguments, timeout)
            logger.info(f"FAL API调用成功，输出: {result}")
            
            if not result.get("images") or len(result["images"]) == 0:
                raise ValueError("FAL API返回结果中没有图片")

            first_image = result["images"][0]
            image_url = first_image["url"]
            width = first_image.get("width", 0)
            height = first_image.get("height", 0)
            
            logger.debug(f"生成的图片URL: {image_url}, 尺寸: {width}x{height}")

            image_tensor, actual_width, actual_height = self._download_and_convert_image(image_url)
            
            if width == 0 or height == 0:
                width, height = actual_width, actual_height

            if num_images > 1:
                image_tensors = [image_tensor]
                for img in result["images"][1:num_images]:
                    img_tensor, _, _ = self._download_and_convert_image(img["url"])
                    image_tensors.append(img_tensor)
                image_tensor = torch.cat(image_tensors, dim=0)

            return (image_tensor, width, height)

        except Exception as e:
            logger.exception(f"FAL图片编辑失败: {str(e)}")
            raise e


NODE_CLASS_MAPPINGS = {
    "FalImage2VideoRequestNode": FalImage2VideoRequestNode,
    "FalVideo2VideoRequestNode": FalVideo2VideoRequestNode,
    "FalFunControlVideoRequestNode": FalFunControlVideoRequestNode,
    "QwenEditPlusLoraNode": QwenEditPlusLoraNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FalImage2VideoRequestNode": "FAL Image2Video Request",
    "FalVideo2VideoRequestNode": "FAL Video2Video Request",
    "FalFunControlVideoRequestNode": "FAL Fun Control Video Request",
    "QwenEditPlusLoraNode": "FAL Qwen Edit Plus LoRA",
}
