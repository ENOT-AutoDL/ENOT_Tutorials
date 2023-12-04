import urllib.request

_BASE_URL = 'https://gitlab.expasoft.com/a.yanchenko/enot-public-data/-/raw/main/tutorials'
_CHKP_BASE_URL = f'{_BASE_URL}/checkpoints_renamed'

# Checkpoints:
PRUNING_IMAGENETTE_CHECKPOINT_URL = f'{_CHKP_BASE_URL}/e2e_imagenette_pruning.pth'


def download_checkpoint(url, dst_path):
    urllib.request.urlretrieve(url=url, filename=dst_path)


def download_imagenette_mobilenet(dst_path):
    download_checkpoint(url=PRUNING_IMAGENETTE_CHECKPOINT_URL, dst_path=dst_path)
