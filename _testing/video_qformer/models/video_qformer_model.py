from __future__ import annotations

import logging
import os
from typing import List, Tuple, Optional

import torch
from torch import nn
from transformers import (
    CLIPModel,
    CLIPImageProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from .q_former import QFormer

__all__ = ["VideoQFormerModel"]

HF_CACHE_ENV = "HF_HOME"

def _resolve_cache_dir(cache_dir_cfg: Optional[str] = None) -> Optional[str]:
    if cache_dir_cfg:
        return cache_dir_cfg
    return os.environ.get(HF_CACHE_ENV)


class VideoQFormerModel(nn.Module):
    """End-to-end video captioning model based on Q-Former bridging CLIP and Vicuna.

    Notes
    -----
    * CLIP and Vicuna parameters are **frozen** by default to keep memory/compute in budget.
    * Only the Q-Former and the linear projection into Vicuna hidden space are trained.
    """

    def __init__(
        self,
        clip_model_name: str,
        llm_model_name: str,
        num_query_tokens: int,
        qformer_num_layers: int,
        qformer_num_heads: int,
        qformer_mlp_dim: int,
        dropout: float = 0.1,
        max_caption_len: int = 200,
        cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__()

        cache_dir = _resolve_cache_dir(cache_dir)  # Might be None -> default HF cache
        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"

        # ------------------------------
        # Vision encoder (CLIP ViT-L/14)
        # ------------------------------
        logging.info("Loading CLIP vision backbone…")
        self.clip: CLIPModel = CLIPModel.from_pretrained(
            clip_model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.float16 if self.device_type == "cuda" else torch.float32,
        )
        self.image_processor = CLIPImageProcessor.from_pretrained(clip_model_name, cache_dir=cache_dir)
        self.clip.eval()  # freeze
        for p in self.clip.parameters():
            p.requires_grad = False

        vision_width = self.clip.config.vision_config.hidden_size  # 1024 for ViT-L/14
        self.num_query_tokens = num_query_tokens
        self.max_caption_len = max_caption_len

        # ------------------------------
        # Q-Former
        # ------------------------------
        self.q_former = QFormer(
            context_dim=vision_width,
            num_query_tokens=num_query_tokens,
            embed_dim=vision_width,  # stay in same space
            num_layers=qformer_num_layers,
            num_heads=qformer_num_heads,
            mlp_dim=qformer_mlp_dim,
            dropout=dropout,
        )

        # ------------------------------
        # Vicuna LLM (frozen)
        # ------------------------------
        logging.info("Loading Vicuna LLM in 8-bit… (this might take a while)")
        quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
        self.llm: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            cache_dir=cache_dir,
            quantization_config=quant_cfg,
            device_map="auto",
            torch_dtype=(torch.float16 if self.device_type == "cuda" else torch.float32),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name, cache_dir=cache_dir, use_fast=True)
        if self.tokenizer.pad_token is None:
            # For Vicuna add pad token if missing
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm.eval()  # freeze LLM
        for p in self.llm.parameters():
            p.requires_grad = False

        llm_width = self.llm.config.hidden_size  # 4096 in Vicuna 7B

        # Linear projection from vision/Q-Former space into Vicuna embedding space
        self.q2llm = nn.Linear(vision_width, llm_width, bias=False)

        # Final layer norm to stabilise
        self.prefix_norm = nn.LayerNorm(llm_width)

        # We'll train: q_former params + q2llm + prefix_norm

        self.clip.to(self.device_type)  # move to GPU if available

    # -----------------------------
    # Internal helpers
    # -----------------------------
    def _encode_video(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode video frames via CLIP vision model.

        Parameters
        ----------
        frames : torch.Tensor
            shape (B, T, C, H, W)
        Returns
        -------
        torch.Tensor
            Visual tokens flattened across time, shape (B, T*seq_len, C).
        """
        B, T, C, H, W = frames.shape
        frames = frames.view(B * T, C, H, W)
        with torch.no_grad():
            vision_out = self.clip.vision_model(pixel_values=frames)
        tokens = vision_out.last_hidden_state  # (B*T, seq_len, hidden)
        hidden = tokens.shape[-1]
        tokens = tokens.view(B, T * tokens.shape[1], hidden)
        return tokens

    def _prepare_prefix(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        # visual_tokens: (B, N, vision_width)
        queries = self.q_former(visual_tokens)  # (B, num_query, vision_width)
        prefix = self.q2llm(queries)  # (B, num_query, llm_width)
        prefix = self.prefix_norm(prefix)
        return prefix

    # -----------------------------
    # Forward (training)
    # -----------------------------
    def forward(self, frames: torch.Tensor, captions: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute training loss.

        Returns
        -------
        loss : torch.Tensor
            Scalar training loss.
        logits : torch.Tensor
            LLM logits for the caption tokens (excluding prefix).
        """
        device = next(self.parameters()).device
        frames = frames.to(next(self.clip.parameters()).device, dtype=next(self.clip.parameters()).dtype)
        visual_tokens = self._encode_video(frames)
        prefix_embeds = self._prepare_prefix(visual_tokens).to(device)

        # Tokenize captions
        tok = self.tokenizer(
            captions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_caption_len,
        )
        input_ids = tok.input_ids.to(device)
        attention_mask = tok.attention_mask.to(device)

        # Embedding lookup for caption ids
        caption_embeds = self.llm.get_input_embeddings()(input_ids)

        # Concatenate prefix and caption embeddings
        prefix_len = prefix_embeds.shape[1]
        inputs_embeds = torch.cat([prefix_embeds, caption_embeds], dim=1)

        # Labels: ignore prefix positions by setting to -100
        ignore_prefix = torch.full((input_ids.shape[0], prefix_len), -100, dtype=torch.long, device=device)
        labels = torch.cat([ignore_prefix, input_ids], dim=1)

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=torch.cat([torch.ones_like(ignore_prefix), attention_mask], dim=1),
            labels=labels,
            return_dict=True,
        )
        return outputs.loss, outputs.logits

    # -----------------------------
    # Generation (inference)
    # -----------------------------
    @torch.inference_mode()
    def generate(self, frames: torch.Tensor, gen_kwargs: Optional[dict] = None) -> List[str]:
        gen_kwargs = gen_kwargs or {
            "max_new_tokens": 64,
            "num_beams": 3,
            "do_sample": False,
        }
        device = next(self.parameters()).device
        frames = frames.to(next(self.clip.parameters()).device, dtype=next(self.clip.parameters()).dtype)
        visual_tokens = self._encode_video(frames)
        prefix_embeds = self._prepare_prefix(visual_tokens).to(device)

        attn_mask_prefix = torch.ones(prefix_embeds.shape[:2], dtype=torch.long, device=device)
        outputs = self.llm.generate(
            inputs_embeds=prefix_embeds,
            attention_mask=attn_mask_prefix,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **gen_kwargs,
        )
        # outputs are token ids including prefix ids (which are <eos> etc.). We only decode newly generated tokens.
        # Because inputs_embeds were given and not input_ids, generate starts at step 0; we can decode entire sequence.
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [c.strip() for c in captions]