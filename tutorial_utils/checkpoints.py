import torch
from google_drive_downloader import GoogleDriveDownloader
from packaging.version import parse as parse_pkg_version

GETTING_STARTED_PRETRAIN_CHECKPOINT_ID = '1lBgHtFs0L-MkbUjUyJbWgkjlHNLXjALc'
METRIC_LEARNING_REGULAR_PRETRAIN_CHECKPOINT_ID = '1imsWOPNeKCJMC0GzJL5ddfNxxHYCbUv1'
METRIC_LEARNING_ARCFACE_PRETRAIN_CHECKPOINT_ID = '1_9bhSCPFsZzDo0jAAM_Jxl_D3lgbVWSJ'
RESOLUTION_SEARCH_PRETRAIN_CHECKPOINT_ID = '1xQpASOiZ9mLidkCSv5Di88p42HKsAsTz'

if parse_pkg_version(torch.__version__) < parse_pkg_version('1.9.0'):
    AUTOGEN_PRETRAIN_CHECKPOINT_ID = '13d_3oVnyjJXVSxCKFULbeRnl4XETPoKo'
    AUTOGEN_SEARCH_CHECKPOINT_ID = '1Rjb-J05fl81bBO6WbgD5RW_WJ-D6mPOp'
else:
    AUTOGEN_PRETRAIN_CHECKPOINT_ID = '18Ss6GxxX_qoTuGUSDYgn3Z_J4G0IeTHh'
    AUTOGEN_SEARCH_CHECKPOINT_ID = '1721Y2jCqdKWF99i7zcxOyMj648tbQljG'


def download_checkpoint(src_id, dst_path, overwrite=True):
    GoogleDriveDownloader.download_file_from_google_drive(
        file_id=src_id,
        dest_path=dst_path,
        overwrite=overwrite,
    )


def download_getting_started_pretrain_checkpoint(dst_path, overwrite=True):
    download_checkpoint(src_id=GETTING_STARTED_PRETRAIN_CHECKPOINT_ID, dst_path=dst_path, overwrite=overwrite)


def download_metric_learning_regular_pretrain_checkpoint(dst_path, overwrite=True):
    download_checkpoint(src_id=METRIC_LEARNING_REGULAR_PRETRAIN_CHECKPOINT_ID, dst_path=dst_path, overwrite=overwrite)


def download_metric_learning_arcface_pretrain_checkpoint(dst_path, overwrite=True):
    download_checkpoint(src_id=METRIC_LEARNING_ARCFACE_PRETRAIN_CHECKPOINT_ID, dst_path=dst_path, overwrite=overwrite)


def download_resolution_search_pretrain_checkpoint(dst_path, overwrite=True):
    download_checkpoint(src_id=RESOLUTION_SEARCH_PRETRAIN_CHECKPOINT_ID, dst_path=dst_path, overwrite=overwrite)


def download_autogen_pretrain_checkpoint(dst_path, overwrite=True):
    download_checkpoint(src_id=AUTOGEN_PRETRAIN_CHECKPOINT_ID, dst_path=dst_path, overwrite=overwrite)


def download_autogen_search_checkpoint(dst_path, overwrite=True):
    download_checkpoint(src_id=AUTOGEN_SEARCH_CHECKPOINT_ID, dst_path=dst_path, overwrite=overwrite)
