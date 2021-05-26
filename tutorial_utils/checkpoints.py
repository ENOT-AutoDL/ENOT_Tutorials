from google_drive_downloader import GoogleDriveDownloader


GETTING_STARTED_PRETRAIN_CHECKPOINT_ID = '1lBgHtFs0L-MkbUjUyJbWgkjlHNLXjALc'
METRIC_LEARNING_REGULAR_PRETRAIN_CHECKPOINT_ID = '1imsWOPNeKCJMC0GzJL5ddfNxxHYCbUv1'
METRIC_LEARNING_ARCFACE_PRETRAIN_CHECKPOINT_ID = '1_9bhSCPFsZzDo0jAAM_Jxl_D3lgbVWSJ'
RESOLUTION_SEARCH_PRETRAIN_CHECKPOINT_ID = '1xQpASOiZ9mLidkCSv5Di88p42HKsAsTz'
AUTOGEN_PRETRAIN_CHECKPOINT_ID = '1YTGa7rcHSWltpWG-2CGh4SYw8iXGEpau'
AUTOGEN_SEARCH_CHECKPOINT_ID = '1EqiDh7lBZrq-bLJE1anQP3xsvcGO0_I1'


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
