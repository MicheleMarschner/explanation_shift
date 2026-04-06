#!/usr/bin/env python3
from pathlib import Path
import sys
import tarfile

import gdown


FILE_ID = "1YGeHsA141o4Rfwor1sXd0_sdwooDRIT4"
ARCHIVE_NAME = "experiments.tar.gz"
TARGET_DIR_NAME = "experiments"


def download_archive(file_id: str, output_path: Path) -> None:
    url = f"https://drive.google.com/uc?id={file_id}"
    result = gdown.download(url, str(output_path), quiet=False)

    if result is None or not output_path.exists():
        raise RuntimeError("Download with gdown failed.")


def extract_archive(archive_path: Path, extract_to: Path) -> None:
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=extract_to)


def download_experiments(project_root: Path) -> None:
    archive_path = project_root / ARCHIVE_NAME
    target_dir = project_root / TARGET_DIR_NAME

    if target_dir.exists():
        print(f"'{TARGET_DIR_NAME}/' already exists. Skipping download.")
        sys.exit(0)

    print("Downloading precomputed experiment outputs from Google Drive...")
    download_archive(FILE_ID, archive_path)

    print(f"Extracting '{ARCHIVE_NAME}'...")
    extract_archive(archive_path, project_root)

    archive_path.unlink(missing_ok=True)

    if target_dir.exists():
        print(f"Done. '{TARGET_DIR_NAME}/' is now available.")
    else:
        print(
            f"Extraction finished, but '{TARGET_DIR_NAME}/' was not found. "
            "Check whether the archive contains the expected top-level folder."
        )
        sys.exit(1)