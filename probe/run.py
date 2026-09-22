"""Native Wan2GP VACE+MultiTalk inference with stage timing and memory telemetry.

Input job JSON contains source/mask/reference/audio paths and exact frame count.
Inputs must already be 25-fps, 4n+1, at requested dimensions, without any legacy
DualFlow preheat prefix. No source RGB is pasted into the decoded output.
"""
import argparse
import contextlib
import functools
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def sha(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class Recorder:
    def __init__(self, root):
        import pynvml
        import psutil
        self.root = root
        self.stage = 'startup'
        self.events = []
        self.stop = threading.Event()
        self.nv = pynvml
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.process = psutil.Process()
        self.samples = 0
        self.nvml_peak_bytes = 0
        self.cpu_peak_bytes = 0
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()

    def sample(self):
        with (self.root/'telemetry.jsonl').open('w') as out:
            while not self.stop.is_set():
                try:
                    nv, h = self.nv, self.handle
                    used = nv.nvmlDeviceGetMemoryInfo(h).used
                    rss = self.process.memory_info().rss
                    row = {'epoch': time.time(), 'stage': self.stage, 'device_used_bytes': used,
                           'cpu_rss_bytes': rss, 'gpu_utilization_pct': nv.nvmlDeviceGetUtilizationRates(h).gpu,
                           'power_w': nv.nvmlDeviceGetPowerUsage(h)/1000,
                           'temperature_c': nv.nvmlDeviceGetTemperature(h, 0)}
                    out.write(json.dumps(row)+'\n')
                    out.flush()
                    self.samples += 1
                    self.nvml_peak_bytes = max(self.nvml_peak_bytes, used)
                    self.cpu_peak_bytes = max(self.cpu_peak_bytes, rss)
                except Exception as error:
                    out.write(json.dumps({'telemetry_error': str(error)})+'\n')
                self.stop.wait(0.2)

    @contextlib.contextmanager
    def phase(self, label):
        import torch
        torch.cuda.synchronize()
        previous, self.stage = self.stage, label
        tick = time.perf_counter()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            self.events.append({'stage': label, 'seconds': time.perf_counter()-tick,
                                'allocated_bytes': torch.cuda.memory_allocated(),
                                'reserved_bytes': torch.cuda.memory_reserved()})
            self.stage = previous

    def wrap(self, target, method, label):
        original = getattr(target, method)
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            with self.phase(label):
                return original(*args, **kwargs)
        setattr(target, method, wrapped)

    def finish(self):
        self.stop.set()
        self.thread.join(timeout=3)
        return {'events': self.events, 'nvml_peak_device_bytes': self.nvml_peak_bytes,
                'cpu_peak_rss_bytes': self.cpu_peak_bytes, 'telemetry_samples': self.samples,
                'nvml_sample_interval_seconds': 0.2}


def decode(path, grayscale=False):
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(str(path))
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY if grayscale else cv2.COLOR_BGR2RGB))
    cap.release()
    if abs(fps-25) > 0.01:
        raise ValueError(f'Expected 25 fps, got {fps}')
    return np.stack(frames)


def export(rgb, output, fps, audio):
    # Lossless primary recovery plus a browser-friendly viewing encode.
    t, h, w, c = rgb.shape
    cmd = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pixel_format', 'rgb24',
           '-video_size', f'{w}x{h}', '-framerate', str(fps), '-i', '-', '-an',
           '-c:v', 'ffv1', '-level', '3', '-pix_fmt', 'bgr0', str(output/'raw.mkv')]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for frame in rgb:
            process.stdin.write(frame.tobytes())
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('Lossless video export failed')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(output/'raw.mkv'), '-i', audio,
                    '-map', '0:v:0', '-map', '1:a:0', '-t', str(t/fps), '-c:v', 'libx264',
                    '-crf', '15', '-preset', 'fast', '-pix_fmt', 'yuv420p', '-c:a', 'aac',
                    '-b:a', '192k', '-movflags', '+faststart', str(output/'raw_with_audio.mp4')], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    job = json.loads(Path(args.job).read_text())
    weights = Path(args.weights).resolve()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out/'job.json').write_text(json.dumps(job, indent=2))
    report = {'job': job, 'started_epoch': time.time(), 'runner_sha256': sha(__file__),
              'upstream_commit': '9ffd0a6702c3c03c333cc48bf179762d80297a00', 'status': 'running'}
    rec = None
    started = time.perf_counter()
    try:
        import torch
        import numpy as np
        from PIL import Image
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required; never substitute CPU/MPS for this benchmark')
        torch.set_num_threads(min(8, os.cpu_count() or 1))
        torch.cuda.reset_peak_memory_stats()
        rec = Recorder(out)
        gpu = torch.cuda.get_device_properties(0)
        report['hardware'] = {'gpu_name': gpu.name, 'total_vram_bytes': gpu.total_memory,
                              'compute_capability': list(torch.cuda.get_device_capability()),
                              'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
                              'python': platform.python_version(), 'platform': platform.platform(),
                              'driver': rec.nv.nvmlSystemGetDriverVersion(),
                              'power_limit_w': rec.nv.nvmlDeviceGetPowerManagementLimit(rec.handle)/1000}
        with rec.phase('cuda_kernel_smoke'):
            q = torch.randn(1, 4, 256, 64, dtype=torch.bfloat16, device='cuda')
            smoke = torch.nn.functional.scaled_dot_product_attention(q, q, q)
            assert torch.isfinite(smoke).all()
            del q, smoke
        from mmgp import offload
        from models.wan.any2video import WanAny2V
        from models.wan.configs import WAN_CONFIGS
        from models.wan.multitalk.multitalk import custom_init, get_embedding, get_window_audio_embeddings, audio_prepare_multi
        from shared.utils import files_locator as fl
        fl.set_checkpoints_paths([str(weights)])
        offload.shared_state['_attention'] = 'sdpa'

        with rec.phase('input_validation_and_audio_embedding'):
            frames = decode(job['source'])
            mask = decode(job['mask'], grayscale=True)
            n, h, w, _ = frames.shape
            assert n == job['frames'] and (n-1)%4 == 0
            assert (h, w) == (job['height'], job['width']) and mask.shape == (n, h, w)
            assert set(np.unique(mask)).issubset({0, 255}), 'Frozen mask must already be binary'
            assert 0 < np.count_nonzero(mask) < mask.size, 'Require partial editing mask'
            report['input_hashes'] = {key: sha(job[key]) for key in ('source', 'mask', 'reference', 'audio')}
            report['mask_edit_fraction'] = np.count_nonzero(mask)/mask.size
            # Same uint8 127 fill and >127.5 binary region semantics as upstream
            # wgp.preprocess_video_with_mask(... process_type='inpaint', expand_scale=0).
            control = np.where(mask[..., None] > 127, np.uint8(127), frames)
            input_frames = torch.from_numpy(control.copy()).permute(3,0,1,2).float()/127.5-1
            input_masks = torch.from_numpy(mask.copy()).unsqueeze(0).float()/255
            ref = np.array(Image.open(job['reference']).convert('RGB'))
            assert ref.shape == (h,w,3)
            refs = [torch.from_numpy(ref.copy()).permute(2,0,1).unsqueeze(1).float()/127.5-1]
            (out/'input_condition_summary.json').write_text(json.dumps({
                'control_shape': list(input_frames.shape), 'mask_shape': list(input_masks.shape),
                'control_sha256': hashlib.sha256(control.tobytes()).hexdigest(),
                'mask_binary': True, 'inpaint_value_uint8': 127, 'mask_expand_pixels': 0,
                'reference_count': 1, 'legacy_preheat_frames': 0,
                'post_decode_source_compositing': False}, indent=2))
            full_audio, _, _, _ = audio_prepare_multi(job['audio'], None, duration=n/25, min_audio_duration=n/25)
            feat, encoder = custom_init('cpu', str(weights/'chinese-wav2vec2-base'))
            embedding = get_embedding(full_audio, feat, encoder, fps=25)
            audio = get_window_audio_embeddings([embedding], clip_length=n)
            report['audio_embedding_shapes'] = [list(x.shape) for x in audio]
            report['audio_embedding_sha256'] = [hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in audio]
            del encoder, feat, full_audio, embedding, frames, mask, control
            gc.collect()

        preset = json.loads((ROOT/'defaults/vace_multitalk_14B.json').read_text())
        model_def = {**preset['model'], 'vace_class': True, 'multitalk_class': True}
        model_files = [weights/name for name in ('Wan14BT2VFusioniX_fp16.safetensors',
                       'wan2.1_Vace_14B_module_mbf16.safetensors', 'wan2.1_multitalk_14B_mbf16.safetensors')]
        with rec.phase('model_load_and_offload_setup'):
            pipe = WanAny2V(WAN_CONFIGS['t2v-14B'], checkpoint_dir=str(weights),
                            model_filename=list(map(str,model_files)), model_type='vace_multitalk_14B',
                            base_model_type='vace_multitalk_14B', model_def=model_def,
                            text_encoder_filename=str(weights/'umt5-xxl/models_t5_umt5-xxl-enc-bf16.safetensors'),
                            quantizeTransformer=False, dtype=torch.bfloat16, VAE_dtype=torch.float32)
            manager = offload.profile({'transformer': pipe.model, 'text_encoder': pipe.text_encoder.model,
                                       'vae': pipe.vae.model}, profile_no=1, quantizeTransformer=False,
                                      extraModelsToQuantize=[], verboseLevel=1)
            pipe._interrupt = False
        report['model_parameter_count'] = sum(x.numel() for x in pipe.model.parameters())
        report['model_parameter_dtypes'] = sorted({str(x.dtype) for x in pipe.model.parameters()})
        report['module_evidence'] = {'vace_blocks': sum(hasattr(b,'vace') for b in pipe.model.blocks),
                                     'audio_blocks': sum(hasattr(b,'audio_cross_attn') for b in pipe.model.blocks),
                                     'shared_backbone_layers': len(pipe.model.blocks)}
        assert report['module_evidence']['vace_blocks'] == 8
        assert report['module_evidence']['audio_blocks'] == 40
        rec.wrap(pipe.model, 'forward', 'transformer_forward')
        rec.wrap(pipe.text_encoder.model, 'forward', 'text_encoder_forward')
        rec.wrap(pipe.vae, 'encode', 'vae_encode')
        rec.wrap(pipe.vae, 'decode_to_cpu_uint8', 'vae_decode')

        def callback(step=-1, *unused, **kwargs):
            if 'progress_unit' not in kwargs:
                print(json.dumps({'step':step, 'epoch':time.time(), 'phase':kwargs.get('progress_title','denoising')}), flush=True)

        settings = {'frame_num': n, 'width': w, 'height': h, 'sampling_steps': job.get('steps',10),
                    'guide_scale':1., 'guide2_scale':1., 'guide3_scale':1., 'guide_phases':1,
                    'audio_cfg_scale':4., 'shift':5., 'sample_solver':'unipc',
                    'seed':job.get('seed',73001), 'cfg_star_switch':False, 'cfg_zero_step':-1,
                    'apg_switch':False, 'joint_pass':False, 'context_scale':[1.],
                    'color_correction_strength':0., 'VAE_tile_size':0, 'fps':25,
                    'video_prompt_type':'VMI', 'model_type':'vace_multitalk_14B'}
        report['effective_settings'] = {**settings, 'attention':'sdpa', 'quantization':False,
                                        'offload_profile':1, 'dtype':'bfloat16', 'vae_dtype':'float32'}
        torch.cuda.reset_peak_memory_stats()
        with rec.phase('generation_including_conditioning'):
            with torch.inference_mode():
                result = pipe.generate(input_prompt=job['prompt'], input_frames=input_frames,
                                       input_masks=input_masks, input_ref_images=refs, audio_proj=audio,
                                       callback=callback, offloadobj=manager, **settings)
        report['generation_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        report['generation_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        pixels = result['x']
        assert pixels.dtype == torch.uint8 and tuple(pixels.shape) == (3,n,h,w)
        report['rgb_sha256'] = hashlib.sha256(pixels.numpy().tobytes()).hexdigest()
        with rec.phase('video_export'):
            export(pixels.permute(1,2,3,0).contiguous().numpy(), out, 25, job['audio'])
        report['output_frames'] = n
        report['output_seconds'] = n/25
        report['transformer_forward_count'] = sum(x['stage']=='transformer_forward' for x in rec.events)
        assert report['transformer_forward_count'] == 2*settings['sampling_steps']
        report['status'] = 'passed'
    except BaseException:
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
        raise
    finally:
        report['wall_seconds'] = time.perf_counter()-started
        report['ended_epoch'] = time.time()
        if rec:
            report.update(rec.finish())
        (out/'metrics.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
        print(json.dumps({'status': report['status'], 'wall_seconds':report['wall_seconds']}), flush=True)


if __name__ == '__main__':
    main()
