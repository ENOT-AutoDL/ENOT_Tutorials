import urllib.request

import torch
from packaging.version import parse as parse_pkg_version

_BASE_URL = 'https://gitlab.expasoft.com/a.yanchenko/enot-public-data/-/raw/main/tutorials'
_CHKP_BASE_URL = f'{_BASE_URL}/checkpoints'
_ONNX_BASE_URL = f'{_BASE_URL}/onnx'

# Checkpoints:
GETTING_STARTED_PRETRAIN_CHECKPOINT_URL = f'{_CHKP_BASE_URL}/getting_started.pth'
METRIC_LEARNING_REGULAR_PRETRAIN_CHECKPOINT_URL = f'{_CHKP_BASE_URL}/metric_learning_regular-130.pth'
METRIC_LEARNING_ARCFACE_PRETRAIN_CHECKPOINT_URL = f'{_CHKP_BASE_URL}/metric_learning_arc_face-30.pth'
RESOLUTION_SEARCH_PRETRAIN_CHECKPOINT_URL = f'{_CHKP_BASE_URL}/resolution_search_mobilenet_100_300.pth'
PRUNING_IMAGENETTE_CHECKPOINT_URL = f'{_CHKP_BASE_URL}/e2e_imagenette_pruning.pth'

if parse_pkg_version(torch.__version__) < parse_pkg_version('1.9.0'):
    AUTOGEN_PRETRAIN_CHECKPOINT_URL = f'{_CHKP_BASE_URL}/autogeneration_pretrain_torch_181.pth'
    AUTOGEN_SEARCH_CHECKPOINT_URL = f'{_CHKP_BASE_URL}/autogeneration_search_torch_181.pth'
else:
    AUTOGEN_PRETRAIN_CHECKPOINT_URL = f'{_CHKP_BASE_URL}/autogeneration_pretrain.pth'
    AUTOGEN_SEARCH_CHECKPOINT_URL = f'{_CHKP_BASE_URL}/autogeneration_search.pth'

# ONNXs:
QUANTIZATION_MOBILENET_ONNX_URL = f'{_ONNX_BASE_URL}/mobilenet_tiny_imagenet_0_70521.onnx'


def download_checkpoint(url, dst_path):
    urllib.request.urlretrieve(url=url, filename=dst_path)


def download_getting_started_pretrain_checkpoint(dst_path):
    download_checkpoint(url=GETTING_STARTED_PRETRAIN_CHECKPOINT_URL, dst_path=dst_path)


def download_metric_learning_regular_pretrain_checkpoint(dst_path):
    download_checkpoint(url=METRIC_LEARNING_REGULAR_PRETRAIN_CHECKPOINT_URL, dst_path=dst_path)


def download_metric_learning_arcface_pretrain_checkpoint(dst_path):
    download_checkpoint(url=METRIC_LEARNING_ARCFACE_PRETRAIN_CHECKPOINT_URL, dst_path=dst_path)


def download_resolution_search_pretrain_checkpoint(dst_path):
    download_checkpoint(url=RESOLUTION_SEARCH_PRETRAIN_CHECKPOINT_URL, dst_path=dst_path)


def download_autogen_pretrain_checkpoint(dst_path):
    download_checkpoint(url=AUTOGEN_PRETRAIN_CHECKPOINT_URL, dst_path=dst_path)


def download_autogen_search_checkpoint(dst_path):
    download_checkpoint(url=AUTOGEN_SEARCH_CHECKPOINT_URL, dst_path=dst_path)


def download_onnx_mobilenet(dst_path):
    download_checkpoint(url=QUANTIZATION_MOBILENET_ONNX_URL, dst_path=dst_path)


def download_imagenette_mobilenet(dst_path):
    download_checkpoint(url=PRUNING_IMAGENETTE_CHECKPOINT_URL, dst_path=dst_path)
