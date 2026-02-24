import ray 
import asyncio
from tests.workers.rollout.my_tools_sever import WebSearchToolClient

@ray.remote
def save_web_search_cache_remote(base_url: str, parameters: dict):
    async def _save():
        client = WebSearchToolClient(base_url=base_url, parameters=parameters)
        return await client.save_cache()
    return asyncio.run(_save())