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
