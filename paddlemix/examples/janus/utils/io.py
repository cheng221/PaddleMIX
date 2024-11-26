import paddle
import paddlenlp
import json
from typing import Dict, List
import PIL.Image
import base64
import io


def load_pil_images(conversations: List[Dict[str, str]]) ->List[PIL.Image.Image
    ]:
    """

    Support file path or base64 images.

    Args:
        conversations (List[Dict[str, str]]): the conversations with a list of messages. An example is :
            [
                {
                    "role": "User",
                    "content": "<image_placeholder>
Extract all information from this image and convert them into markdown format.",
                    "images": ["./examples/table_datasets.png"]
                },
                {"role": "Assistant", "content": ""},
            ]

    Returns:
        pil_images (List[PIL.Image.Image]): the list of PIL images.

    """
    pil_images = []
    for message in conversations:
        if 'images' not in message:
            continue
        for image_data in message['images']:
            if image_data.startswith('data:image'):
                _, image_data = image_data.split(',', 1)
                image_bytes = base64.b64decode(image_data)
                pil_img = PIL.Image.open(io.BytesIO(image_bytes))
            else:
                pil_img = PIL.Image.open(image_data)
            pil_img = pil_img.convert('RGB')
            pil_images.append(pil_img)
    return pil_images


def load_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
        return data
