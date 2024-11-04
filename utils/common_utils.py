import base64


def image_to_base64(image_path: str) -> str:
    """Converts an image to base64 string

    Args:
        image_path (str): _description_

    Returns:
        str: _description_
    """
    with open(image_path, "rb") as image_file:
        base64_string = base64.b64encode(image_file.read()).decode("utf-8")
    return base64_string
