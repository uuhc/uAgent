# model_manager.py
import os
from model_adapters.qwen_adapter import QWenAdapter


class ModelManager:
    def get_adapter(self, model_name: str):
        if model_name == "qwen-vl-max":
            api_key = os.getenv("API_KEY_TONGYI")
            if api_key:
                return QWenAdapter(api_key=api_key, model_name=model_name)
        # 其他模型的适配器获取逻辑...

    def text_to_text(self, model_name: str, prompt: str) -> str:
        adapter = self.get_adapter(model_name)
        if adapter:
            return adapter.text_to_text(prompt)
        else:
            raise ValueError(f"Model {model_name} is not supported")

    def img_to_text(self, model_name: str, prompt: str, img_base64: str) -> str:
        adapter = self.get_adapter(model_name)
        if adapter:
            return adapter.image_to_text(prompt, img_base64)
        else:
            raise ValueError(f"Model {model_name} is not supported")
