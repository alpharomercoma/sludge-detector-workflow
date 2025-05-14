import json
import logging
import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class VideoCaptionDataset(Dataset):
    """Custom dataset that returns uniformly sampled frames and associated caption."""

    def __init__(
        self,
        root_dir: str,
        split: str,
        num_frames: int,
        id_field: str = "video_id",
        caption_field: str = "human_caption",
        transform: transforms.Compose | None = None,
    ) -> None:
        super().__init__()
        logging.info(f"Loading dataset split='{split}' from {root_dir}")
        self.root_dir = root_dir
        self.split = split
        self.num_frames = num_frames
        self.id_field = id_field
        self.caption_field = caption_field
        self.transform = transform or transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

        self.video_dir = os.path.join(self.root_dir, self.split)
        self.video_files = [f for f in os.listdir(self.video_dir) if f.endswith(".mp4")]
        if not self.video_files:
            raise RuntimeError(f"No video files found in {self.video_dir}")
        logging.info(f"Found {len(self.video_files)} videos in {self.video_dir}")

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.video_files)

    # ------------------------------------------------------------------
    def __getitem__(self, index: int):
        video_file = self.video_files[index]
        video_path = os.path.join(self.video_dir, video_file)
        json_path = os.path.splitext(video_path)[0] + ".json"

        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"Annotation missing for {video_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        video_id = meta.get(self.id_field)
        caption = meta.get(self.caption_field, "")
        if caption is None:
            caption = ""

        frames = self._sample_frames(video_path)
        return {"frames": frames, "caption": caption, "video_id": video_id}

    # ------------------------------------------------------------------
    def _sample_frames(self, video_path: str) -> torch.Tensor:
        # Use OpenCV to extract uniformly spaced frames
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            raise RuntimeError(f"Video appears empty: {video_path}")

        ids = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        frames: list[torch.Tensor] = []
        for fid in ids:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
            ret, frame = cap.read()
            if not ret:
                logging.warning(f"Failed to read frame {fid} from {video_path}. Using black frame placeholder.")
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = self.transform(frame)
            frames.append(frame_tensor)
        cap.release()
        return torch.stack(frames)  # (T,C,H,W)