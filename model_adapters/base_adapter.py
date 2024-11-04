# model_adapters/base_adapter.py
from abc import ABC, abstractmethod


class BaseAdapter(ABC):
    @abstractmethod
    def text_to_text(self, prompt: str) -> str:
        pass

    @abstractmethod
    def image_to_text(self, prompt: str, img_base64: str) -> str:
        pass

    @abstractmethod
    def text_to_image(self, prompt: str):
        pass
