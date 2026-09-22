"""Freeze a legacy padded clip into an independent Wan2GP single-window input."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import cv2
import numpy as np
import soundfile as sf
from PIL import Image


def digest(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_video(path, gray=False):
    cap = cv2.VideoCapture(str(path))
    assert abs(cap.get(cv2.CAP_PROP_FPS)-25) < .01
    items = []
    while True:
        success, frame = cap.read()
        if not success:
            break
        items.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY if gray else cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(items)


def write_video(path, data, gray=False):
    _, h, w = data.shape[:3]
    cmd = ['ffmpeg','-v','error','-y','-f','rawvideo','-pixel_format','gray' if gray else 'rgb24',
           '-video_size',f'{w}x{h}','-framerate','25','-i','-','-an','-c:v','ffv1',
           '-pix_fmt','gray' if gray else 'bgr0',str(path)]
    p = subprocess.Popen(cmd,stdin=subprocess.PIPE)
    p.communicate(data.tobytes())
    assert p.returncode == 0
    assert np.array_equal(read_video(path,gray),data), 'Lossless roundtrip mismatch'


def main():
    p=argparse.ArgumentParser()
    for arg in ('source','mask','audio','output','case','prompt'):
        p.add_argument('--'+arg,required=True)
    p.add_argument('--prefix',type=int,default=12)
    p.add_argument('--body-frames',type=int,required=True)
    p.add_argument('--review-frames',type=int,required=True)
    p.add_argument('--remote-root',default='/workspace/probe/inputs')
    args=p.parse_args()
    root=Path(args.output)
    root.mkdir(parents=True,exist_ok=False)
    source=read_video(args.source)[args.prefix:args.prefix+args.body_frames]
    mask=read_video(args.mask,True)[args.prefix:args.prefix+args.body_frames]
    mask=np.where(mask>127.5,255,0).astype(np.uint8)
    assert len(source)==len(mask)==args.body_frames and (len(source)-1)%4==0
    waveform,sr=sf.read(args.audio,dtype='int16')
    assert sr==16000 and waveform.ndim==1
    audio=waveform[args.prefix*640:(args.prefix+args.body_frames)*640]
    assert len(audio)==args.body_frames*640
    write_video(root/'source.mkv',source)
    write_video(root/'mask.mkv',mask,True)
    sf.write(root/'driver.wav',audio,sr,subtype='PCM_16')
    reread,_=sf.read(root/'driver.wav',dtype='int16')
    assert np.array_equal(reread,audio)
    Image.fromarray(source[0]).save(root/'reference.png')
    control=np.where(mask[...,None]>0,127,source).astype(np.uint8)
    write_video(root/'control_preview.mkv',control)
    picks=sorted(set([0,len(source)//2,len(source)-1]))
    panels=[]
    for idx in picks:
        m=mask[idx]
        overlay=source[idx].astype(float)
        overlay[m>0]=overlay[m>0]*.65+np.array([255,45,70])*.35
        panels.append(np.concatenate([source[idx],overlay.astype(np.uint8),control[idx]],axis=1))
    sheet=np.concatenate(panels,axis=0)
    Image.fromarray(sheet).resize((720, int(sheet.shape[0]*720/sheet.shape[1]))).save(root/'input_panels.jpg')
    base=Path(args.remote_root)/args.case
    job={'case':args.case,'source':str(base/'source.mkv'),'mask':str(base/'mask.mkv'),
         'reference':str(base/'reference.png'),'audio':str(base/'driver.wav'),
         'frames':len(source),'review_frames':args.review_frames,'width':source.shape[2],
         'height':source.shape[1],'fps':25,'steps':10,'seed':73001,'prompt':args.prompt}
    (root/'job.json').write_text(json.dumps(job,indent=2)+'\n')
    audit={'source_inputs':{k:{'path':getattr(args,k),'sha256':digest(getattr(args,k))} for k in ('source','mask','audio')},
           'prefix_removed_frames':args.prefix,'body_frames':len(source),'review_frames':args.review_frames,
           'mask_threshold':'>127.5','mask_dilation':0,'mask_feather':0,'control_fill_uint8':127,
           'reference_policy':'exact first decoded body frame','fps':25,'audio_sr':16000,
           'prepared':{x.name:digest(x) for x in root.iterdir() if x.is_file()},
           'source_rgb_roundtrip_exact':True,'mask_roundtrip_exact':True,'audio_pcm_exact':True}
    (root/'INPUT_FREEZE.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:job[k] for k in ('case','frames','review_frames','width','height')}))


if __name__=='__main__':
    main()
