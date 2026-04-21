"""Download FashionMNIST locally and upload its raw files to S3.

Run this once before launching the training job. The destination S3 prefix must
match what `launch_training.py` passes as the `training` input channel.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import boto3
from torchvision import datasets

DEFAULT_S3_URI = "s3://dsw-melax-dev-s3/data/hxjeong/sagemaker-tutorial/input"


def upload_dir(local: Path, s3_uri: str) -> None:
    assert s3_uri.startswith("s3://")
    bucket, _, prefix = s3_uri[5:].partition("/")
    s3 = boto3.client("s3")
    for f in local.rglob("*"):
        if not f.is_file():
            continue
        key = f"{prefix.rstrip('/')}/{f.relative_to(local).as_posix()}"
        print(f"  -> s3://{bucket}/{key}")
        s3.upload_file(str(f), bucket, key)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--local-dir", default="./data")
    p.add_argument("--s3-uri", default=DEFAULT_S3_URI)
    args = p.parse_args()

    local = Path(args.local_dir)
    # Triggers download of FashionMNIST/raw/*.gz under local/.
    datasets.FashionMNIST(str(local), train=True, download=True)
    datasets.FashionMNIST(str(local), train=False, download=True)

    print(f"Uploading {local}/FashionMNIST -> {args.s3_uri}/FashionMNIST")
    upload_dir(local / "FashionMNIST", f"{args.s3_uri}/FashionMNIST")
    print("Done.")


if __name__ == "__main__":
    main()
