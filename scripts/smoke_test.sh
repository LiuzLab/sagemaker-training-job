#!/usr/bin/env bash
# Local end-to-end smoke test. Does NOT submit a SageMaker training job.
#
# Verifies: host-side deps, syntax, module imports, each CLI's --help, a
# 1-epoch CPU run of training.py against a freshly-downloaded FashionMNIST,
# inference.py against the produced artifact, and a dry-construct of the
# SageMaker ModelTrainer.
#
# Runs anywhere the README prerequisites are installed (SageMaker Studio,
# a laptop with `pip install -r requirements.txt`, CI, etc.). No AWS
# credentials required — the dry-construct only resolves a DLC image URI.

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d -t smoke_test_XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

PASSED=0
FAILED=0
RESULTS=()

record() {
    local status="$1" name="$2"
    RESULTS+=("$status  $name")
    case "$status" in
        PASS) PASSED=$((PASSED + 1)) ;;
        FAIL) FAILED=$((FAILED + 1)) ;;
    esac
}

run_check() {
    local name="$1"; shift
    echo "── $name"
    if "$@"; then
        record PASS "$name"
    else
        record FAIL "$name"
    fi
}

cd "$REPO_ROOT"

PY_FILES=(training.py inference.py launch_training.py prepare_data.py)

# 0. Preflight: fail fast with an actionable hint if host-side deps are missing.
#    These are the prerequisites from README.md — without them later checks
#    produce confusing errors. `sagemaker.core` only exists in SDK v3.
if ! python - <<'PY'
import importlib.util
import sys


def have(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


missing = [m for m in ("torch", "torchvision", "boto3") if not have(m)]
if not have("sagemaker.core"):
    missing.append("sagemaker>=3.0")
if missing:
    print("Missing host-side dependencies: " + ", ".join(missing), file=sys.stderr)
    print("Install with:  pip install -r requirements.txt", file=sys.stderr)
    sys.exit(1)
PY
then
    exit 1
fi

# 1. Syntax check
run_check "py_compile all .py files" python -m py_compile "${PY_FILES[@]}"

# 2. Module imports
run_check "import all modules" python -c "
import importlib.util, sys
for name in ['training', 'inference', 'launch_training', 'prepare_data']:
    spec = importlib.util.spec_from_file_location(name, f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        print(f'import {name}: FAIL - {e}', file=sys.stderr); sys.exit(1)
print('all imports ok')
"

# 3. --help on every CLI
for f in "${PY_FILES[@]}"; do
    run_check "$f --help" python "$f" --help >/dev/null
done

# 4. 1-epoch CPU training on real FashionMNIST
DATA="$TMP_DIR/data"
MODEL="$TMP_DIR/model"
OUTPUT="$TMP_DIR/output"
mkdir -p "$DATA" "$MODEL" "$OUTPUT"

run_check "download FashionMNIST to tmp" python -c "
from torchvision import datasets
datasets.FashionMNIST('$DATA', train=True,  download=True)
datasets.FashionMNIST('$DATA', train=False, download=True)
"

run_check "training.py 1 epoch on CPU" \
    python training.py --epochs 1 --batch-size 512 \
        --data-dir "$DATA" --model-dir "$MODEL" --output-dir "$OUTPUT"

# 5. Artifacts written
if [[ -f "$MODEL/model.pt" && -f "$MODEL/classes.json" ]]; then
    record PASS "model.pt + classes.json written"
else
    record FAIL "model.pt + classes.json written"
fi

# 6. Inference round-trip
run_check "inference.py on saved artifact" \
    python inference.py --model-dir "$MODEL" --data-dir "$DATA" --n 8

# 7. Dry-construct the SageMaker ModelTrainer (imports + image resolve, no API call).
#    Falls back to us-east-1 when no AWS region is configured so the DLC image
#    lookup works on a laptop without `aws configure`. The resolved URI is a
#    public ECR path — no credentials needed.
: "${AWS_DEFAULT_REGION:=${AWS_REGION:-us-east-1}}"
export AWS_DEFAULT_REGION
run_check "ModelTrainer dry-construct" python <<'PY'
import boto3
import sagemaker.core.helper.session_helper as smh
from sagemaker.core.image_uris import retrieve
from sagemaker.core.shapes.shapes import OutputDataConfig, StoppingCondition, Tag
from sagemaker.core.training.configs import Compute, InputData, SourceCode
from sagemaker.train.model_trainer import ModelTrainer

session = smh.Session(boto_session=boto3.Session())
region = session.boto_region_name
image = retrieve(framework="pytorch", region=region, version="2.3.0",
                 py_version="py311", instance_type="ml.g4dn.xlarge",
                 image_scope="training")
ModelTrainer(
    sagemaker_session=session, training_image=image,
    role="arn:aws:iam::000000000000:role/dummy", base_job_name="smoke",
    source_code=SourceCode(source_dir=".", command="true", requirements="requirements.txt"),
    compute=Compute(instance_type="ml.g4dn.xlarge", instance_count=1, volume_size_in_gb=30),
    input_data_config=[InputData(channel_name="training",
                                 data_source="s3://bucket/prefix")],
    output_data_config=OutputDataConfig(s3_output_path="s3://bucket/out"),
    stopping_condition=StoppingCondition(max_runtime_in_seconds=3600),
    environment={"PYTHONUNBUFFERED": "1"},
    tags=[Tag(key="Project", value="smoke-test")],
)
print("ok")
PY

echo
echo "────────── SUMMARY ──────────"
for r in "${RESULTS[@]}"; do echo "$r"; done
echo "─────────────────────────────"
echo "passed=$PASSED  failed=$FAILED"

if (( FAILED > 0 )); then exit 1; fi
