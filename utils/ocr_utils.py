import base64
import cv2
import logging
import numpy as np
from paddleocr import PaddleOCR
from typing import List
from loguru import logger


# 通过 logging 模块将 ppocr 日志级别设置为 ERROR
logging.getLogger("ppocr").setLevel(logging.ERROR)

# 初始化 PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # 支持中文


def image_to_base64(image_path: str) -> str:
    """将图片文件转换为 Base64 编码的字符串。

    Args:
        image_path (str): 图片文件的路径。

    Returns:
        str: 图片的 Base64 编码字符串。
    """
    with open(image_path, "rb") as image_file:
        base64_str = base64.b64encode(image_file.read()).decode("utf-8")
    return base64_str


def base64_to_image(base64_str: str) -> np.ndarray:
    """将 Base64 编码的字符串转换为 OpenCV 图像。

    Args:
        base64_str (str): Base64 编码的字符串。

    Returns:
        np.ndarray: OpenCV 格式的图像。
    """
    img_data = base64.b64decode(base64_str)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img


def ocr_from_base64(base64_str: str) -> List:
    """从 Base64 编码的图像字符串中进行 OCR 识别。

    Args:
        base64_str (str): 图像的 Base64 编码字符串。

    Returns:
        List: OCR 识别结果列表。
    """
    img = base64_to_image(base64_str)
    result = ocr.ocr(img, cls=True)
    logger.info(f"base64_str, OCR result: {result}")
    return result


def ocr_from_img_path(img_path: str) -> List:
    """从图片路径中进行 OCR 识别。

    Args:
        img_path (str): 图片的文件路径。

    Returns:
        List: OCR 识别结果列表。
    """
    result = ocr.ocr(img_path, cls=True)
    logger.info(f"img_path: {img_path}, OCR result: {result}")
    return result


if __name__ == "__main__":
    img_path = "test/1.png"
    result = ocr_from_img_path(img_path)

    base64_str = image_to_base64(img_path)
    result = ocr_from_base64(base64_str)
