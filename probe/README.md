# 同干合流 — VACE + MultiTalk probe

Independent clone of [Wan2GP](https://github.com/deepbeepmeep/Wan2GP), pinned to
`9ffd0a6702c3c03c333cc48bf179762d80297a00`. Original license and history are retained.
This directory adds a narrow, headless inference/profiling harness around the
native `WanAny2V` implementation. No model training or DualFlow velocity mixer.

Initial configuration: published Vace Multitalk FusioniX 14B preset, 10 steps,
text CFG 1, audio CFG 4, flow shift 5, 25 fps, SDPA, BF16 compute, no weight
quantization, native VACE conditioning and MultiTalk audio embeddings. Raw full
frame decoded output is the primary result, without source RGB compositing.

The image contains code and dependencies only. Weights are downloaded separately
from their public provider. User media, credentials, rental/account metadata and
human review are kept outside this public repository.

The Docker build checks imports on CPU. It does not verify CUDA execution; the
first rented GPU run must separately pass that check. The image is a minimal
probe runtime and does not promise support for every Wan2GP UI/model family.

GPU records must distinguish checkpoint download, model loading, audio encoding,
VACE/VAE conditioning, denoising, decoding and media export. Report both PyTorch
peak allocated/reserved memory and NVML device memory; do not infer a different
GPU's speed or capacity from model parameter size alone.

Model licenses and upstream terms apply independently of this code repository.

## Published runtime

Public Docker Hub image (anonymous pull, `linux/amd64`):

```text
bigmy1/wan21-vace-multitalk-probe@sha256:cbce7fd90299f60cafc9b6a5dc8bc8e71f02893330d1cd91f166e12793a66c2a
```

Image source revision: `9d2b7ae7370beab0c91465d3d7c24d90857d8880`.
The image uses PyTorch 2.10.0 / CUDA 12.8. `/opt/environment-freeze.txt` records
the installed dependencies and `/opt/source-revision.txt` records its source.
Weights are not embedded in the image. `probe/weights.json` pins all twelve
required public files, approximately 52.40 GB, to a Hugging Face revision.

For Vast SSH mode, use this on-start command to allow noninteractive SSH/rsync:

```sh
touch /root/.no_auto_tmux
mkdir -p /workspace/probe
```

After placing a prepared job and media under `/workspace/probe/inputs/`, run:

```sh
python /opt/Wan2GP/probe/download_weights.py --destination /workspace/probe/weights
python /opt/Wan2GP/probe/run.py \
  --job /workspace/probe/inputs/example/job.json \
  --weights /workspace/probe/weights \
  --output /workspace/probe/results/example
```

The output directory must be new. Each run writes `metrics.json` and
`telemetry.jsonl`, including on failure, and on success writes lossless `raw.mkv`
plus `raw_with_audio.mp4`. The job paths refer to the container filesystem.
Use `python probe/prepare_case.py --help` for the private input preparation
adapter; `--prefix` must match the actual prefix in that input, not an assumed
requirement of this model. Prepared inputs must have `4n+1` frames at 25 fps.

To rebuild, use `probe/Dockerfile` with a full `SOURCE_REVISION` build argument.
The optional manual GitHub workflow publishes to GHCR; its package visibility
must be configured separately before assuming anonymous pulls are available.
