"""Download only the pinned public files needed by this probe; verify identities."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from huggingface_hub import hf_hub_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(__file__).with_name('weights.json').read_text())
    root = Path(args.destination)
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    verified = []
    for entry in manifest['files']:
        tick = time.perf_counter()
        path = Path(hf_hub_download(manifest['repo_id'], entry['path'], revision=manifest['revision'], local_dir=root, token=False))
        if path.stat().st_size != entry['size']:
            raise RuntimeError(f"File size mismatch: {entry['path']}")
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if entry['sha256'] and digest != entry['sha256']:
            raise RuntimeError(f"SHA256 mismatch: {entry['path']}")
        verified.append({**entry, 'sha256_actual': digest, 'download_and_verify_seconds': time.perf_counter()-tick})
        print(json.dumps(verified[-1]), flush=True)
    report = {**manifest, 'verified': verified, 'elapsed_seconds': time.perf_counter()-started}
    (root/'WEIGHTS_VERIFIED.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
