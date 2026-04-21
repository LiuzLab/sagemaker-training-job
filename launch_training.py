"""Submit the FashionMNIST training job to SageMaker (SDK v3 `ModelTrainer`).

Works both inside SageMaker Studio (role auto-resolved via
`sagemaker.get_execution_role()`) and on a laptop with AWS credentials
configured via `aws configure`.
"""

from __future__ import annotations

import argparse
import os

import boto3
import sagemaker
import sagemaker.core.helper.session_helper as smh
from sagemaker.core.image_uris import retrieve
from sagemaker.core.shapes.shapes import OutputDataConfig, StoppingCondition, Tag
from sagemaker.core.training.configs import Compute, InputData, SourceCode
from sagemaker.train.model_trainer import ModelTrainer

DEFAULT_INPUT = "s3://dsw-melax-dev-s3/data/hxjeong/sagemaker-tutorial/input"
DEFAULT_OUTPUT = "s3://dsw-melax-dev-s3/data/hxjeong/sagemaker-tutorial/output"


def resolve_role(cli_role: str | None) -> str:
    if cli_role:
        return cli_role
    if os.environ.get("SM_TRAINING_ROLE_ARN"):
        return os.environ["SM_TRAINING_ROLE_ARN"]
    # SageMaker SDK v3 dropped the top-level `sagemaker.get_execution_role()`
    # helper, so fall back to STS. Inside Studio / a notebook the caller is
    # already the execution role (assumed via STS), and we just need to
    # rewrite the assumed-role ARN back to the plain role ARN that
    # CreateTrainingJob expects.
    try:
        import boto3
        arn = boto3.client("sts").get_caller_identity()["Arn"]
        if ":assumed-role/" in arn:
            account = arn.split(":")[4]
            role_name = arn.split("assumed-role/")[1].split("/")[0]
            return f"arn:aws:iam::{account}:role/{role_name}"
        if ":role/" in arn:
            return arn
        raise ValueError(f"unsupported caller ARN: {arn}")
    except Exception as e:
        raise SystemExit(
            "Could not resolve an IAM role. Pass --role-arn or export "
            "SM_TRAINING_ROLE_ARN when running outside SageMaker."
        ) from e


def resolve_image(region: str, instance_type: str) -> str:
    # AWS Deep Learning Container with PyTorch pre-installed.
    return retrieve(
        framework="pytorch",
        region=region,
        version="2.3.0",
        py_version="py311",
        instance_type=instance_type,
        image_scope="training",
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-s3", default=DEFAULT_INPUT)
    p.add_argument("--output-s3", default=DEFAULT_OUTPUT)
    p.add_argument("--role-arn", default=None)
    p.add_argument("--instance-type", default="ml.g4dn.xlarge")
    p.add_argument("--instance-count", type=int, default=1)
    p.add_argument("--volume-gb", type=int, default=30)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--no-wait", action="store_true")
    args = p.parse_args()

    role = resolve_role(args.role_arn)
    session = smh.Session(boto_session=boto3.Session())
    region = session.boto_region_name
    image_uri = resolve_image(region, args.instance_type)

    # Command the container runs after SourceCode is staged. `requirements.txt`
    # is picked up automatically by the DLC's training-toolkit, but we pip-install
    # explicitly to keep the command self-documenting.
    train_cmd = " ".join([
        "pip install -r requirements.txt &&",
        f"python training.py --epochs {args.epochs} --batch-size {args.batch_size}",
    ])

    print(f"Region:   {region}")
    print(f"Image:    {image_uri}")
    print(f"Role:     {role}")
    print(f"Instance: {args.instance_type} x {args.instance_count}")
    print(f"Input:    {args.input_s3}")
    print(f"Output:   {args.output_s3}")

    trainer = ModelTrainer(
        sagemaker_session=session,
        training_image=image_uri,
        role=role,
        base_job_name="fashion-mnist",
        source_code=SourceCode(
            source_dir=".",
            command=train_cmd,
            requirements="requirements.txt",
            ignore_patterns=[".git", "__pycache__", ".DS_Store", ".venv", "data"],
        ),
        compute=Compute(
            instance_type=args.instance_type,
            instance_count=args.instance_count,
            volume_size_in_gb=args.volume_gb,
        ),
        input_data_config=[
            # Mounts s3://.../input/ at /opt/ml/input/data/training inside the container.
            InputData(channel_name="training", data_source=args.input_s3),
        ],
        output_data_config=OutputDataConfig(s3_output_path=args.output_s3),
        stopping_condition=StoppingCondition(max_runtime_in_seconds=3600),
        environment={"PYTHONUNBUFFERED": "1"},
        tags=[
            Tag(key="Project", value="sagemaker-tutorial"),
            Tag(key="Owner", value=os.environ.get("USER", "unknown")),
        ],
    )

    trainer.train(wait=not args.no_wait)


if __name__ == "__main__":
    main()
