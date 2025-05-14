import os
import json
import cv2
import numpy as np
import torch
import argparse
import logging
from torchvision import transforms
from utils import load_config, setup_logging
from video_qformer.models.video_qformer_model import VideoQFormerModel
from video_qformer.data import VideoCaptionDataset  # not used but maybe later

# Load config for cache and resources
cfg_top = load_config('video_qformer/config.yaml')
hf_cache_dir = cfg_top['cache']['hf_cache_dir']
os.makedirs(hf_cache_dir, exist_ok=True)
os.environ['HF_HOME'] = hf_cache_dir
os.environ['TRANSFORMERS_CACHE'] = hf_cache_dir
os.environ['HF_DATASETS_CACHE'] = hf_cache_dir
os.environ['HF_METRICS_CACHE'] = hf_cache_dir

# Threads
torch.set_num_threads(cfg_top['resources']['cpu_threads'])
torch.set_num_interop_threads(cfg_top['resources']['cpu_threads'])

def inference(video_path):
    cfg = load_config('video_qformer/config.yaml')
    setup_logging()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Reserve GPU memory fraction
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(
            cfg_top['resources']['gpu_mem_fraction'], torch.cuda.current_device()
        )
    # Single video loader
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466,0.4578275,0.40821073],
                             std=[0.26862954,0.26130258,0.27577711])
    ])
    # Load frames and metadata directly from paths
    json_path = os.path.splitext(video_path)[0] + '.json'
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"No JSON for {video_path}")
    with open(json_path) as f:
        meta = json.load(f)
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, total-1, cfg['data']['num_frames'], dtype=int)
    frames_list = []
    for fid in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
        ret, frame = cap.read()
        if not ret:
            logging.warning(f"Failed to read frame {fid}")
            frame = np.zeros((224,224,3), dtype=np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_list.append(transform(frame))
    cap.release()
    frames = torch.stack(frames_list).unsqueeze(0).to(device)
    # Model
    model = VideoQFormerModel(
        clip_model_name=cfg['model']['clip_model_name'],
        llm_model_name=cfg['model']['llm_model_name'],
        num_query_tokens=cfg['model']['num_query_tokens'],
        qformer_num_layers=cfg['model']['qformer_num_layers'],
        qformer_num_heads=cfg['model']['qformer_num_heads'],
        qformer_mlp_dim=cfg['model']['qformer_mlp_dim'],
        dropout=cfg['model']['dropout'],
        max_caption_len=cfg['train']['max_caption_len'],
    ).to(device)
    model.load_state_dict(torch.load(f"{cfg['train']['output_dir']}/best_model.pth", map_location=device))
    model.eval()
    # Generate
    caps = model.generate(frames)
    print("Generated Captions:")
    for c in caps:
        print(c)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True)
    args = parser.parse_args()
    inference(args.video)
