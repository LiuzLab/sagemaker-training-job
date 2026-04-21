# SageMaker Training Tutorial — FashionMNIST

A minimal, runnable example of launching a SageMaker training job for a
PyTorch classifier on
[FashionMNIST](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html).
Two launch paths are covered:

1. **Inside AWS** — a SageMaker Studio / notebook instance (role auto-resolved).
2. **Outside AWS** — a laptop with the AWS CLI configured.

## Files

| File | Purpose |
| --- | --- |
| [`training.py`](training.py) | Training entry-point that runs **inside** the training container. |
| [`inference.py`](inference.py) | Loads the produced `model.tar.gz` and predicts on held-out samples. |
| [`launch_training.py`](launch_training.py) | Submits the job using the SageMaker SDK v3 `ModelTrainer` API. |
| [`prepare_data.py`](prepare_data.py) | Downloads FashionMNIST locally and uploads it to S3 once. |
| [`requirements.txt`](requirements.txt) | Extra pip packages installed in the container before training runs. |

## S3 layout

Everything lives under
`s3://dsw-melax-dev-s3/data/hxjeong/sagemaker-tutorial/`:

```
.../sagemaker-tutorial/
├── input/                           <-- dataset (you upload this once)
│   └── FashionMNIST/raw/*.gz
└── output/                          <-- SageMaker writes here
    └── fashion-mnist-<ts>/
        ├── output/model.tar.gz      <-- contents of SM_MODEL_DIR, tarred
        └── output/output.tar.gz     <-- contents of SM_OUTPUT_DATA_DIR
```

Why these paths:

- **`input/`** — matches the `training` input channel declared in
  [`launch_training.py`](launch_training.py). SageMaker mounts this S3 prefix at
  `/opt/ml/input/data/training` inside the container, which is what
  `torchvision.datasets.FashionMNIST` points at (via `SM_CHANNEL_TRAINING`).
- **`output/`** — passed as `output_path=` to the estimator. SageMaker tars
  `/opt/ml/model` into `model.tar.gz` and `/opt/ml/output/data` into
  `output.tar.gz` under this prefix when the job finishes. The training code
  never uploads to S3 directly — it just writes to those local paths.

If the job uses checkpoints, add a third prefix (e.g.
`.../sagemaker-tutorial/checkpoints/`) and wire it via `checkpoint_s3_uri=`
so SageMaker keeps local `/opt/ml/checkpoints` in sync with S3.

## Prerequisites

```bash
pip install "sagemaker>=3.0" boto3 torch torchvision
aws configure   # only needed outside AWS
```

> The launcher uses the SageMaker SDK **v3** `ModelTrainer` API (`sagemaker.train.model_trainer`). If you're stuck on v2, use the legacy `sagemaker.pytorch.PyTorch` estimator instead — the inputs (`source_dir`, `entry_point`, `hyperparameters`, `fit({"training": ...})`) are roughly equivalent.

You also need an IAM role that SageMaker can assume, with
`AmazonSageMakerFullAccess` and read/write on the S3 bucket above.

## Step 1 — Upload the dataset (run once)

```bash
python prepare_data.py \
    --s3-uri s3://dsw-melax-dev-s3/data/hxjeong/sagemaker-tutorial/input
```

## Step 2 — Launch the training job

### Path A: Inside AWS (SageMaker Studio / notebook)

The SDK picks up the notebook's execution role automatically.

```bash
python launch_training.py --epochs 5 --instance-type ml.g4dn.xlarge
```

### Path B: Outside AWS (local shell with AWS CLI)

Pass the role ARN explicitly (or export `SM_TRAINING_ROLE_ARN`):

```bash
python launch_training.py \
    --role-arn arn:aws:iam::<account-id>:role/<SageMakerExecutionRole> \
    --epochs 5 --instance-type ml.g4dn.xlarge
```

Either way, the SDK:

1. Tars the current directory ([`training.py`](training.py) + [`requirements.txt`](requirements.txt)) and uploads
   it to the default SageMaker bucket.
2. Starts an `ml.g4dn.xlarge` container with the PyTorch 2.3 DLC image
   (resolved via `sagemaker.core.image_uris.retrieve`).
3. Mounts `s3://.../input/` at `/opt/ml/input/data/training`.
4. Runs `pip install -r requirements.txt && python training.py …` inside
   the container.
5. Streams stdout to CloudWatch and, on success, uploads `model.tar.gz` +
   `output.tar.gz` to `output_path`.

### Path B (alternative): raw AWS CLI

If you'd rather skip the Python SDK entirely, you can call
`aws sagemaker create-training-job` directly. You'll need to (a) build/push
your own image or reference an AWS-provided DLC URI for your region, and (b)
upload your code tarball yourself. A minimal call looks like:

```bash
aws sagemaker create-training-job \
    --training-job-name fashion-mnist-$(date +%s) \
    --role-arn arn:aws:iam::<account-id>:role/<SageMakerExecutionRole> \
    --algorithm-specification '{
        "TrainingImage": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.3.0-gpu-py311-cu121-ubuntu20.04-sagemaker",
        "TrainingInputMode": "File"
    }' \
    --input-data-config '[{
        "ChannelName": "training",
        "DataSource": {"S3DataSource": {
            "S3DataType": "S3Prefix",
            "S3Uri": "s3://dsw-melax-dev-s3/data/hxjeong/sagemaker-tutorial/input",
            "S3DataDistributionType": "FullyReplicated"
        }}
    }]' \
    --output-data-config '{"S3OutputPath": "s3://dsw-melax-dev-s3/data/hxjeong/sagemaker-tutorial/output"}' \
    --resource-config '{"InstanceType": "ml.g4dn.xlarge", "InstanceCount": 1, "VolumeSizeInGB": 30}' \
    --stopping-condition '{"MaxRuntimeInSeconds": 3600}' \
    --hyper-parameters '{"epochs": "5", "batch-size": "128",
        "sagemaker_program": "training.py",
        "sagemaker_submit_directory": "s3://<your-bucket>/code/sourcedir.tar.gz"}'
```

The Python SDK is the preferred path — it handles the code tarball, picks the
right DLC image for the region, and wires up metric definitions for you.

## Step 3 — Predict with the trained model

After the job finishes, [`launch_training.py`](launch_training.py) prints the model artifact URI.
Fetch and run inference locally:

```bash
aws s3 cp <model_data_uri> model.tar.gz
mkdir -p model && tar -xzf model.tar.gz -C model
python inference.py --model-dir model --n 16
```

## How `training.py` talks to SageMaker

SageMaker communicates with the entry-point purely through paths and
environment variables — no SageMaker SDK call inside the container:

| In container | What it is | In this example |
| --- | --- | --- |
| `SM_CHANNEL_TRAINING` | Mount point of the `training` input channel | Loaded by `datasets.FashionMNIST` |
| `SM_MODEL_DIR` | Files written here get packaged into `model.tar.gz` | `model.pt`, `classes.json` |
| `SM_OUTPUT_DATA_DIR` | Files written here get packaged into `output.tar.gz` | `metrics.json` |
| stdout | Lines matching `metric_definitions` regex become CloudWatch metrics | `test_acc=0.91;` |

Because of the local fallbacks at the top of `main()`, the same [`training.py`](training.py)
also runs on a laptop with `python training.py --data-dir ./data`, which makes
iterating much cheaper than doing every change through a SageMaker job.

## Verified end-to-end (local)

| Check | Result |
| --- | --- |
| `py_compile` on all 4 files | PASS |
| Import each module | PASS |
| `--help` for every CLI | PASS |
| `training.py` 1 epoch on CPU | PASS — `test_acc=0.844` |
| `inference.py` on the saved artifact | PASS — 13/16 correct |
| `ModelTrainer` dry-construct + DLC image resolve | PASS |

## Smoke test

`scripts/smoke_test.sh` runs everything locally in ~1 minute and prints a
PASS/FAIL summary. It does **not** submit a SageMaker job — it only exercises
the code paths you can verify without AWS compute, and it doesn't require
AWS credentials (the DLC image URI is a public ECR path):

```bash
pip install -r requirements.txt   # host-side deps, run once
bash scripts/smoke_test.sh
```

What it checks: a preflight that the host-side deps from `requirements.txt`
are installed (fails fast with an install hint if not), `py_compile` on
every `.py` file, importing each module, `--help` on every CLI, a 1-epoch
CPU training run against a freshly downloaded FashionMNIST, that `model.pt`
+ `classes.json` were written, an `inference.py` round-trip against the
produced artifact, and a dry-construct of the SageMaker `ModelTrainer`
(imports + DLC image resolution, no API call). Non-zero exit on any
failure.

The script defaults `AWS_DEFAULT_REGION` to `us-east-1` when nothing is
configured, so the image-URI lookup works on a laptop with no `aws
configure`. Export a different region to override.

## Efficient recipe

- Run [`training.py`](training.py) locally first — the `SM_*` env-var fallbacks let you iterate in ~30 seconds instead of waiting 5 minutes for a container to spin up.
- Upload the dataset once with [`prepare_data.py`](prepare_data.py). Every training job just mounts that S3 prefix.
- Pick the right instance. `ml.g4dn.xlarge` (~$0.75/hr) is enough for FashionMNIST. Avoid `ml.p3.*` / `ml.p4d.*` for toy examples.
- Use `--no-wait` once it's working. The launcher returns immediately and you watch progress in the SageMaker console.
