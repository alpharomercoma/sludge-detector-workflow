import torch
import os
from torch.utils.data import DataLoader
from utils import load_config
cfg_top = load_config('video_qformer/config.yaml')
# Setup huggingface cache
hf_cache = cfg_top['cache']['hf_cache_dir']
os.makedirs(hf_cache, exist_ok=True)
os.environ['HF_HOME'] = hf_cache
os.environ['TRANSFORMERS_CACHE'] = hf_cache
os.environ['HF_DATASETS_CACHE'] = hf_cache
os.environ['HF_METRICS_CACHE'] = hf_cache
# Threads
torch.set_num_threads(cfg_top['resources']['cpu_threads'])
torch.set_num_interop_threads(cfg_top['resources']['cpu_threads'])
import logging
from utils import setup_logging
from video_qformer.data import VideoCaptionDataset
from video_qformer.models.video_qformer_model import VideoQFormerModel


def evaluate():
    cfg = load_config('video_qformer/config.yaml')
    setup_logging()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Data
    val_ds = VideoCaptionDataset(
        cfg['data']['root_dir'],
        cfg['data']['val_split'],
        cfg['data']['num_frames'],
        id_field=cfg['data']['id_field'],
        caption_field=cfg['data']['caption_field']
    )
    loader = DataLoader(val_ds, batch_size=cfg['train']['batch_size'], num_workers=cfg['train']['num_workers'])
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
    # Load best
    model.load_state_dict(torch.load(f"{cfg['train']['output_dir']}/best_model.pth", map_location=device))
    model.eval()
    total_loss = 0
    count = 0
    with torch.no_grad():
        for batch in loader:
            frames = batch['frames'].to(device)
            captions = batch['caption']
            loss, _ = model(frames, captions)
            total_loss += loss.item()
            count += 1
    avg = total_loss / max(count, 1)
    logging.info(f"Evaluation loss: {avg:.4f}")


if __name__ == '__main__':
    evaluate()
