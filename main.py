import requests
from random import shuffle
import psutil
import logging
import sys
import ctypes
import io
import logging
import os

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(stream=sys.stdout),
    ],
)

logger = logging.getLogger(__name__)


def get_memory_free_percent():
    mem_total = psutil.virtual_memory().total
    mem_available = psutil.virtual_memory().available
    usage = mem_available / mem_total * 100
    return usage


def trim_memory():
    libc = ctypes.CDLL("libc.so.6")
    return libc.malloc_trim(0)


class Model:

    def __init__(self):
        self.model_checkpoint = "openai/clip-vit-base-patch32"
        os.system("transformers env")
        self.device = "cpu"
        logger.info(f"Start setting up model {self.model_checkpoint} on {self.device}")
        self.model = AutoModel.from_pretrained(self.model_checkpoint).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_checkpoint, use_fast=False)
        logger.info(f"Finished setting up model {self.model_checkpoint} on {self.device}")

    def _encode(self, inputs: dict) -> list[float]:
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        if "pixel_values" in inputs:
            features = self.model.get_image_features(**inputs)
        else:
            features = self.model.get_text_features(**inputs)

        result = features.cpu().detach().numpy().tolist()

        del inputs
        del features
        if self.device == "cuda":
            torch.cuda.empty_cache()

        # ??????????
        # trim_memory()

        return result

    def encode_images(self, images: list[bytes]) -> list[list[float]]:
        """
        Process images into embeddings
        """
        image_list = [Image.open(io.BytesIO(image)) for image in images]
        with torch.inference_mode():
            inputs = self.processor(
                images=image_list,
                return_tensors="pt",
                padding=True,
            )
            result = self._encode(inputs)
        for image in image_list:
            image.close()
        return result


model = Model()

image_urls = [
    "https://i.pinimg.com/originals/c5/2f/b0/c52fb0e9de148e812c542414ee46206e.jpg",
    "https://i.pinimg.com/736x/bd/17/11/bd17116b655e6ecb00390fe6746707fc.jpg",
    "https://www.pandashop.md/i/products/88/887773.jpg",
    "https://a.d-cd.net/hK31YeVAqagL9l1Q1cdmmO3x8eI-1920.jpg",
    "https://avatars.mds.yandex.net/i?id=f4e1f464eb6e157788284afff19d8fec_l-4434207-images-thumbs&n=13",
    "https://i.pinimg.com/originals/88/bd/5a/88bd5a2b83f1008de0453384be6994f7.png",
    "https://i.pinimg.com/originals/0b/4f/14/0b4f1419e5ceefda9f67c10cc516cb0b.png",
    "https://i.pinimg.com/originals/33/21/51/332151582c4a4868c428f118d302f0a5.png",
    "https://i.pinimg.com/originals/54/22/76/542276c97d0b76facfbc91d505090b2e.png",
    "https://i.pinimg.com/originals/24/ef/77/24ef771f5bc818beb6430efa2725b798.png",
]
image_data = [requests.get(image_url, stream=True).raw.data for image_url in image_urls] * 5
shuffle(image_urls)


def do_model_calls_loop():
    while True:
        try:
            files = []

            for image in image_data:
                files.append(image)

            model.encode_images(files)
            logger.info(
                f"Batch processed. Memory free NO FASTAPI: {get_memory_free_percent():.3f}%"
            )

        except Exception as exc:
            logger.error(f"Cannot call model: {repr(exc)}")


do_model_calls_loop()
