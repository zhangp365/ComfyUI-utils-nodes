import server
from aiohttp import web
import logging
logger = logging.getLogger(__file__)
import os
import shutil,filecmp
import __main__

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, GenderWordsConfig
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]


@server.PromptServer.instance.routes.get("/utils_node/reload_gender_words_config")
async def reload_gender_words_config(request):
    try:
        GenderWordsConfig.load_config()
        return web.json_response({"result": "reload successful."})
    except Exception as e:
        logger.exception(e)
        return web.json_response({"error": str(e)})

# This will copy the required javascript into the correct location so it gets loaded for decorating
def update_javascript():
    extensions_folder = os.path.join(os.path.dirname(os.path.realpath(__main__.__file__)),
                                     "web" + os.sep + "extensions" + os.sep + "ComfyUI-utils-nodes")
    javascript_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)),"web" ,"js")

    if not os.path.exists(extensions_folder):
        print("Creating frontend extension folder: " + extensions_folder)
        os.mkdir(extensions_folder)

    result = filecmp.dircmp(javascript_folder, extensions_folder)

    if result.left_only or result.diff_files:
        print('Update to javascripts files detected')
        file_list = list(result.left_only)
        file_list.extend(x for x in result.diff_files if x not in file_list)

        for file in file_list:
            print(f'Copying {file} to extensions folder')
            src_file = os.path.join(javascript_folder, file)
            dst_file = os.path.join(extensions_folder, file)
            if os.path.exists(dst_file):
                os.remove(dst_file)
            shutil.copy(src_file, dst_file)


update_javascript()