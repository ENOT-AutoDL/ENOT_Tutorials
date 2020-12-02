from google_drive_downloader import GoogleDriveDownloader


GETTING_STARTED_PRETRAIN_CHECKPOINT_ID = '1P9yDap43rIbfgGPZVfv-BJL-4H0HO6PP'
METRIC_LEARNING_REGULAR_PRETRAIN_CHECKPOINT_ID = '1wRyuPeq0EBCLfcEe4oHdq8MGXei6ZU4w'
METRIC_LEARNING_ARCFACE_PRETRAIN_CHECKPOINT_ID = '1rvU6t9Rt2zAcKTu3JP6d2gUKYHhSwGFm'


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
