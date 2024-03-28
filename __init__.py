import server
from aiohttp import web
import logging
logger = logging.getLogger(__file__)



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
