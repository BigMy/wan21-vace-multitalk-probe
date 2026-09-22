"""CPU-only dependency check. Does not claim to validate CUDA kernels/inference."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from unittest.mock import patch

# Upstream attention selects capabilities at import time. This import-only stub
# is confined to image construction; never used by the inference runner.
with patch.object(torch.cuda, 'get_device_capability', return_value=(12, 0)):
    from models.wan.any2video import WanAny2V
    from models.wan.multitalk.multitalk import custom_init, get_embedding, get_window_audio_embeddings
    from models.wan.configs import WAN_CONFIGS
    from mmgp import offload
print('Native WanAny2V/MultiTalk dependency imports passed; no GPU forward performed.')
