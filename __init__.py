import server
from aiohttp import web
import logging
logger = logging.getLogger(__file__)
import os
import importlib.util
import shutil,filecmp
import __main__

from .py.nodes import GenderWordsConfig


@server.PromptServer.instance.routes.get("/utils_node/reload_gender_words_config")
async def reload_gender_words_config(request):
    try:
        GenderWordsConfig.load_config()
        return web.json_response({"result": "reload successful."})
    except Exception as e:
        logger.exception(e)
        return web.json_response({"error": str(e)})


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

def get_ext_dir(subpath=None, mkdir=False):
    dir = os.path.dirname(__file__)
    if subpath is not None:
        dir = os.path.join(dir, subpath)

    dir = os.path.abspath(dir)

    if mkdir and not os.path.exists(dir):
        os.makedirs(dir)
    return dir

py = get_ext_dir("py")
files = os.listdir(py)
for file in files:
    if not file.endswith(".py"):
        continue
    name = os.path.splitext(file)[0]
    if not name.startswith("node"):
        continue
    try:
        imported_module = importlib.import_module(".py.{}".format(name), __name__)
        NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **imported_module.NODE_CLASS_MAPPINGS}
        NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, **imported_module.NODE_DISPLAY_NAME_MAPPINGS}
    except Exception as e:
        logger.exception(e)

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
