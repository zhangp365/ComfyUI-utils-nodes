# this code is original from https://github.com/ShmuelRonen/ComfyUI-Gemini_Flash_2.0_Exp, added cache and gender support
import os
import sys
sys.path.append(".")
import google.generativeai as genai
from contextlib import contextmanager
from collections import OrderedDict
import folder_paths
import logging
import yaml
from google.api_core import retry
from google.generativeai.types import RequestOptions
logger = logging.getLogger(__name__)

config_dir = os.path.join(folder_paths.base_path, "config")
if not os.path.exists(config_dir):
    os.makedirs(config_dir)


def get_config():
    try:
        config_path = os.path.join(config_dir, 'gemini_config.yml')
        with open(config_path, 'r') as f:  
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config
    except:
        return {}

def save_config(config):
    config_path = os.path.join(config_dir, 'gemini_config.yml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, indent=4)

@contextmanager
def temporary_env_var(key: str, new_value):
    old_value = os.environ.get(key)
    if new_value is not None:
        os.environ[key] = new_value
    elif key in os.environ:
        del os.environ[key]
    try:
        yield
    finally:
        if old_value is not None:
            os.environ[key] = old_value
        elif key in os.environ:
            del os.environ[key]

class LRUCache(OrderedDict):
    def __init__(self, capacity):
        super().__init__()
        self.capacity = capacity

    def get(self, key):
        if key not in self:
            return None
        self.move_to_end(key)
        return self[key]

    def put(self, key, value):
        if key in self:
            self.move_to_end(key)
        self[key] = value
        if len(self) > self.capacity:
            self.popitem(last=False)

class GeminiPromptEnhance:
    default_prompt = "### Instruction: 1.Edit and enhance the text description of the image. \nAdd quality descriptors, like 'A high-quality photo, an 8K photo.' \n2.Add lighting descriptions based on the scene, like 'The lighting is natural and bright, casting soft shadows.' \n3.Add scene descriptions according to the context, like 'The overall mood is serene and peaceful.' \n4.If a person is in the scene, include a description of the skin, such as 'natural skin tones and ensure the skin appears realistic with clear, fine details.' \n\n5.Only output the result of the text, no others.\n### Text:"

    def __init__(self, api_key=None, proxy=None):
        config = get_config()
        self.api_key = api_key or config.get("GEMINI_API_KEY")
        self.proxy = proxy or config.get("PROXY")
        self.cache_size = 500  # 缓存最大条数
        self.cache_file = os.path.join(config_dir, 'prompt_cache_gemini.yml')
        self.cache = LRUCache(self.cache_size)
        self.last_prompt = ""
        if self.api_key is not None:
            self.configure_genai()

    def load_cache(self):
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache_data = yaml.load(f, Loader=yaml.FullLoader)
                    # 重新创建LRU缓存
                    for k, v in cache_data.items():
                        self.cache.put(k, v)
        except Exception as e:
            logger.error(f"加载缓存出错: {str(e)}")
            self.cache = LRUCache(self.cache_size)

    def save_cache(self):
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                yaml.dump(dict(self.cache), f, indent=4)
        except Exception as e:
            logger.error(f"保存缓存出错: {str(e)}")

    def configure_genai(self):
        genai.configure(api_key=self.api_key, transport='rest')

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": cls.default_prompt, "multiline": True}),
            },
            "optional": {
                "text_input": ("STRING", {"default": "", "multiline": True}),
                "api_key": ("STRING", {"default": ""}),
                "proxy": ("STRING", {"default": ""}),
                "max_output_tokens": ("INT", {"default": 8192, "min": 1, "max": 8192}),
                "temperature": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.1}),
                "gender_prior": (["","M", "F"], {"default": ""}),
                "gender_alternative": ("STRING", {"forceInput": True}),
                "enabled": ("BOOLEAN", {"default": True}),                
                "request_exception_handle": (["bypass","raise_exception","output_exception"], {"default":"bypass"}),
                "model": (["gemini-2.0-flash","gemini-2.5-flash-lite", "gemini-2.5-flash"], {"default": "gemini-2.5-flash-lite"})           
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("generated_content",)
    FUNCTION = "generate_content"
    CATEGORY = "utils/text"

    def prepare_content(self, prompt, text_input, gender=""):
        gender_word = "male" if gender == "M" else "female" if gender == "F" else gender
        if "### Instruction" not in prompt:
            prompt = f"### Instruction:" + "\n".join([f"{i+1}.{line}" for i, line in enumerate(prompt.split("\n")) if line.strip()])
        if gender_word:
            gender_instruction = f"### Instruction:\n0. Edit and enhance the text below,must replacing the main object's traits with those provided in ({gender_word}), and ensure they are well-integrated into the narrative. "
            prompt = prompt.replace("### Instruction:", gender_instruction)
        if "### Text:" not in prompt:
            prompt = prompt + "\n### Text:"
        text_content = prompt if not text_input else f"{prompt} \n{text_input}"
        logger.debug(f"text_content: {text_content}")
        return [{"text": text_content}]

    def generate_content(self, prompt, text_input=None, api_key="", proxy="",
                        max_output_tokens=8192, temperature=0.4, gender_prior="",gender_alternative="", enabled=True, request_exception_handle="bypass", model="gemini-2.0-flash"):
        if not enabled:
            return (text_input,)
        if prompt is None or prompt.strip() == "":
            prompt = self.default_prompt

        if prompt != self.last_prompt:
            self.last_prompt = prompt
            self.cache.clear()
            logger.info(f"clear cache for new prompt: {prompt}")

        gender = gender_prior if gender_prior else gender_alternative
        # 生成缓存键
        cache_key = f"{text_input or ''}_{gender}"
        
        # 检查缓存
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return (cached_result,)

        # Set all safety settings to block_none by default
        safety_settings = [
            {"category": "harassment", "threshold": "NONE"},
            {"category": "hate_speech", "threshold": "NONE"},
            {"category": "sexually_explicit", "threshold": "NONE"},
            {"category": "dangerous_content", "threshold": "NONE"},
            {"category": "civic", "threshold": "NONE"}
        ]

        # Only update API key if explicitly provided in the node
        if api_key.strip():
            self.api_key = api_key
            save_config({"GEMINI_API_KEY": self.api_key, "PROXY": self.proxy})
            self.configure_genai()
        
        # Only update proxy if explicitly provided in the node    
        if proxy.strip():
            self.proxy = proxy
            save_config({"GEMINI_API_KEY": self.api_key, "PROXY": self.proxy})

        if not self.api_key:
            raise ValueError("API key not found in gemini_config.yml or node input")

        model_name = f'models/{model}'
        model = genai.GenerativeModel(model_name)

        # Apply fixed safety settings to the model
        model.safety_settings = safety_settings

        generation_config = genai.types.GenerationConfig(
            max_output_tokens=max_output_tokens,
            temperature=temperature
        )
        logger.debug(f"self.proxy: {self.proxy}")
        if self.proxy:
            with temporary_env_var('HTTP_PROXY', self.proxy), temporary_env_var('HTTPS_PROXY', self.proxy):
                generated_content = self.do_request(model, generation_config, prompt, text_input, gender,cache_key, request_exception_handle)
        else:
            generated_content = self.do_request(model, generation_config,  prompt, text_input, gender,cache_key, request_exception_handle)
        logger.debug(f"gender_alternative: {gender_alternative}, text_input: {text_input}, gender: {gender}, \ngenerated_content: {generated_content}")
        return (generated_content,)

    def do_request(self, model, generation_config, prompt, text_input, gender, cache_key, request_exception_handle="bypass"):
        try:           
            content_parts = self.prepare_content(prompt, text_input, gender)
            response = model.generate_content(content_parts, generation_config=generation_config, request_options= RequestOptions(
                timeout=8))
            generated_content = response.text
            
            if generated_content.startswith("I'm sorry"):
                raise Exception(f"Gemini returned an rejection: {generated_content}")
            # 更新缓存
            self.cache.put(cache_key, generated_content)
            self.save_cache()
            
        except Exception as e:
            logger.exception(e)
            if request_exception_handle == "raise_exception":
                raise e
            elif request_exception_handle == "output_exception":
                generated_content = f"Error: {str(e)}"
            else:
                generated_content = text_input
        return generated_content
        
NODE_CLASS_MAPPINGS = {
    "GeminiPromptEnhance": GeminiPromptEnhance,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiPromptEnhance": "Gemini prompt enhance",
}

# add a test code here
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info("start test")
    enhance = GeminiPromptEnhance()
    result = enhance.generate_content(enhance.default_prompt, "a photo of a beautiful girl ",gender_alternative= "M")
    logger.info(result)
    
