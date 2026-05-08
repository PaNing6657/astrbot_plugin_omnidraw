"""图片 Provider 基类。"""
import aiohttp
import asyncio
import base64
import json
import mimetypes
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from astrbot.api import logger
from ..models import ProviderConfig

class BaseProvider(ABC):
    # 使用类变量或实例变量存储每个节点的轮询位置
    _key_indices: Dict[str, int] = {}

    def __init__(self, config: ProviderConfig, session: aiohttp.ClientSession):
        self.config = config
        self.session = session
        if self.config.id not in BaseProvider._key_indices:
            BaseProvider._key_indices[self.config.id] = 0

    def get_current_key(self) -> str:
        if not self.config.api_keys:
            return ""
        idx = BaseProvider._key_indices[self.config.id]
        key = self.config.api_keys[idx % len(self.config.api_keys)]
        BaseProvider._key_indices[self.config.id] = (idx + 1) % len(self.config.api_keys)
        return key

    def encode_local_image_to_base64(self, image_path: str) -> Optional[str]:
        """将本地图片文件转为 API 兼容的 Base64 字符串"""
        if not image_path or not os.path.exists(image_path):
            return None
        
        logger.info(f"[{self.config.id}] 正在将本地参考图转为 Base64: {image_path}")
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                mime_type = mimetypes.guess_type(image_path)[0] or "image/png"
                return f"data:{mime_type};base64,{encoded_string}"
        except Exception as e:
            logger.error(f"❌ 读取本地图片失败: {e}")
            return None

    def get_reference_images(self, **kwargs: Any) -> List[str]:
        refs: List[str] = []
        for key in ("user_refs", "persona_refs"):
            value = kwargs.get(key)
            if isinstance(value, (list, tuple)):
                refs.extend(str(item) for item in value if item)

        for key in ("user_ref", "persona_ref"):
            value = kwargs.get(key)
            if value:
                refs.append(str(value))

        seen = set()
        return [ref for ref in refs if not (ref in seen or seen.add(ref))]

    @abstractmethod
    async def generate_image(self, prompt: str, **kwargs: Any) -> str:
        pass

    async def _poll_task_result(self, task_id: str) -> str:
        base_url = self.config.base_url.rstrip("/")
        poll_url = f"{base_url}/v1/tasks/{task_id}"
        headers = {
            "Authorization": "Bearer " + self.get_current_key(),
            "Content-Type": "application/json",
        }

        logger.info(f"⏳ [异步轮询] 首次查询前等待 15 秒，Task ID: {task_id}")
        await asyncio.sleep(15)

        max_total_wait = max(60, int(self.config.timeout))
        start_time = time.time()
        poll_interval = max(3, self.config.async_poll_interval)

        while True:
            elapsed = time.time() - start_time
            if elapsed >= max_total_wait:
                raise RuntimeError(f"异步任务轮询超时，已等待 {max_total_wait} 秒。")

            await asyncio.sleep(poll_interval)

            try:
                async with self.session.get(poll_url, headers=headers, timeout=30) as response:
                    if response.status >= 400:
                        logger.warning(f"⚠️ 轮询请求失败: HTTP {response.status}")
                        continue
                    data = await response.json()

                status = str(data.get("status", data.get("task_status", ""))).upper()
                logger.info(f"⏳ [异步轮询] Task ID: {task_id}, 状态: {status}")

                if status == "COMPLETED":
                    result = data.get("result", {})
                    images = result.get("images", [])
                    if images and isinstance(images, list):
                        img_data = images[0]
                        if isinstance(img_data, dict):
                            urls = img_data.get("url", [])
                            if urls and isinstance(urls, list):
                                return urls[0]
                    raise RuntimeError(f"任务完成但未找到图片 URL。API 返回: {data}")

                if status in {"FAILED", "FAIL", "FAILURE"}:
                    error_msg = data.get("error", {}).get("message", data.get("message", "未知失败原因"))
                    raise RuntimeError(f"异步任务失败: {error_msg}")

            except RuntimeError:
                raise
            except Exception as exc:
                logger.warning(f"⚠️ 轮询请求异常: {exc}")
                continue
