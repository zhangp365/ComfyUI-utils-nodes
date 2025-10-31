
from urllib.parse import urlparse, parse_qs, unquote
import requests
import os
from tqdm import tqdm
import folder_paths
import logging
logger = logging.getLogger(__name__)


class ModelsDownloaderOfUtils:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "url_0": ("STRING", {"default": "", "multiline": True}),
                "url_1": ("STRING", {"default": "", "multiline": True}),
                "url_2": ("STRING", {"default": "", "multiline": True}),
                "url_3": ("STRING", {"default": "", "multiline": True}),
                "subdirectory": ("STRING", {"default": "loras"}),
                "proxy": ("STRING", {"default": ""}),
                "timeout": ("INT", {"default": 300, "min": 1, "max": 3600}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("model_name_0", "model_name_1", "model_name_2", "model_name_3")
    FUNCTION = "download_models"
    CATEGORY = "utils/download"

    def extract_filename_from_url(self, url):
        if not url or not url.strip():
            return None
        
        parsed = urlparse(url)
        params = parse_qs(parsed.query)        
     
        path_part = unquote(parsed.path)
        filename = os.path.basename(path_part)
        
        if "?" in filename:
            filename = filename.split("?")[0]
        
        if not filename:
            raise ValueError(f"cannot extract filename from url: {url}")
        
        if "." not in filename:
            if "format" in params:
                format_param = params["format"][0]
                extension_map = {
                    "SafeTensor": "safetensors",
                    "PickleTensor": "ckpt",
                }
                ext = extension_map.get(format_param, format_param.lower())
                filename = f"{filename}.{ext}"
            else:
                raise ValueError(f"cannot extract filename from url: {url}")
        
        return filename

    def download_file(self, url, filepath, proxy, timeout):
        proxies = {"http": proxy, "https": proxy} if proxy and proxy.strip() else None
        
        response = requests.get(url, proxies=proxies, timeout=timeout, stream=True)
        response.raise_for_status()
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        total_size = int(response.headers.get("content-length", 0))
        with open(filepath, "wb") as f:
            if total_size > 0:
                for chunk in tqdm(response.iter_content(chunk_size=8192), total=total_size // 8192, desc=f"Downloading {os.path.basename(filepath)}"):
                    f.write(chunk)
            else:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

    def download_models(self, url_0, url_1, url_2, url_3, subdirectory, proxy, timeout):
        urls = [url_0, url_1, url_2, url_3]
        model_names = []
        
        try:
            base_paths = folder_paths.get_folder_paths(subdirectory)
        except Exception as e:
            logger.error(f"Error getting folder paths for {subdirectory}: {e}")
            raise ValueError(f"Subdirectory '{subdirectory}' not found in models directory")
      
        if not base_paths:
            raise ValueError(f"Subdirectory '{subdirectory}' not found in models directory")
        
        base_path = base_paths[0]
        
        for url in urls:
            if not url or not url.strip():
                model_names.append("")
                continue
            
            filename = self.extract_filename_from_url(url)
            if not filename:
                model_names.append("")
                continue
            
            filepath = os.path.join(base_path, filename)
            
            if os.path.exists(filepath):
                logger.info(f"Model {filename} already exists in {base_path}")
                model_names.append(filename)
                continue
            
            try:
                self.download_file(url, filepath, proxy, timeout)
            except Exception as e:
                logger.error(f"Error downloading {url} to {filepath}: {e}")
                if os.path.exists(filepath):
                    os.remove(filepath)
                raise ValueError(f"Error downloading {url} to {filepath}: {e}")
            
            model_names.append(filename)
        
        return tuple(model_names)


NODE_CLASS_MAPPINGS = {
    "ModelsDownloaderOfUtils": ModelsDownloaderOfUtils,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelsDownloaderOfUtils": "Models Downloader",
}