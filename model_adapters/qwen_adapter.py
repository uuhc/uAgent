import requests
from loguru import logger
from model_adapters.base_adapter import BaseAdapter


class QWenAdapter(BaseAdapter):
    def __init__(self, api_key: str, model_name: str) -> None:
        self.api_key = api_key
        self.model_name = model_name

    def image_to_text(self, prompt: str, img_base64: str) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}"
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
        response = requests.post(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            headers=headers,
            json=payload,
        )
        result = response.json()
        content = result.get("choices")[0].get("message", {}).get("content", "")
        usage = result.get("usage")
        logger.info(f"result: {content}, usage: {usage}")
        return content

    def text_to_image(self, prompt: str):
        pass

    def text_to_text(self, prompt: str) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt},
            ],
        }
        response = requests.post(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            headers=headers,
            json=payload,
        )
        result = response.json()
        content = result.get("choices")[0].get("message", {}).get("content", "")
        usage = result.get("usage")
        logger.info(f"result: {content}, usage: {usage}")
        return content
