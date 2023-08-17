import hashlib
import os
import urllib
import warnings
from typing import Any, Union, List
from pkg_resources import packaging

import mindspore as ms
from PIL import Image
from mindspore.dataset.vision import Resize, CenterCrop, ToTensor, Normalize, Decode, ToPIL
from mindspore.dataset.transforms import Compose
from tqdm import tqdm
from mindspore import Tensor, load_checkpoint

from .model import build_model
from .simple_tokenizer import SimpleTokenizer as _Tokenizer

try:
    from mindspore.dataset.vision import Inter
    BICUBIC = Inter.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

if packaging.version.parse(ms.__version__) < packaging.version.parse("2.0.0"):
    warnings.warn("MindSpore version 2.0.0 or higher is recommended")

__all__ = ["available_models", "load", "tokenize"]
_tokenizer = _Tokenizer()

_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}


def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True,
                  unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError("Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        ToPIL(),
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711], is_hwc=False),
    ])


# def _transform(n_px,image):
#     if isinstance(image,Image.Image):
#         return Compose([
#             Resize(n_px, interpolation=BICUBIC),
#             CenterCrop(n_px),
#             _convert_image_to_rgb,
#             ToTensor(),
#             Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711], is_hwc=False),
#         ])
#     #判断输入是否为image格式，是的话就不用ToPIL()，否则其他就加ToPIL()
#     else:
#         return Compose([
#             ToPIL(),
#             Resize(n_px, interpolation=BICUBIC),
#             CenterCrop(n_px),
#             _convert_image_to_rgb,
#             ToTensor(),
#             Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711], is_hwc=False),
#         ])

def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())


def load(path: str, device: str = "Ascend", mode: int = 1):
    """Load a CLIP model

    Parameters
    ----------
    path : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the parameter_dict

    device : str
        The device to put the loaded model, must be one of CPU, GPU, Ascend

    mode : int
        Whether to load the optimized JIT model or more hackable non-JIT model (default).
        If yes, GRAPH_MODE is supposed to be used, set mode to be 0;
        else use PYNATIVE_MODE and set mode to be 1.

    Returns
    -------
    model : mindspore.nn.Cell
        The CLIP model

    preprocess : Callable[[PIL.Image], mindspore.Tensor]
        A mindspore vision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    ms.set_context(device_target=device, mode=mode)
    if os.path.isfile(path):
        ckp_dict = load_checkpoint(path)
    else:
        raise RuntimeError(f"Model {path} not found; available models = {available_models()}")

    model = build_model(ckp_dict)
    if str(device).lower() == "cpu":
        model.to_float(ms.float32)
    return model, _transform(model.visual.input_resolution)


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Tensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = ms.ops.zeros((len(all_tokens), context_length), dtype=ms.int32)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = Tensor(tokens)

    return result
