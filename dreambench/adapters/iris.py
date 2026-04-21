"""IRIS world model adapter using eloialonso/iris.

IRIS (Imagination with auto-Regression over an Inner Speech) is a
transformer-based world model that tokenizes observations via a VQ-VAE
and autoregressively predicts future tokens. Pretrained checkpoints for
all 26 Atari 100k games are available on HuggingFace.
"""

import sys
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from dreambench.adapters.base import WorldModelAdapter

# Add iris src to path
_IRIS_PATH = Path(__file__).parent.parent.parent / "third_party" / "iris" / "src"
if str(_IRIS_PATH) not in sys.path:
    sys.path.insert(0, str(_IRIS_PATH))


def _extract_state_dict(state_dict: dict, module_name: str) -> OrderedDict:
    return OrderedDict(
        {k.split(".", 1)[1]: v for k, v in state_dict.items() if k.startswith(module_name)}
    )


class IRISAdapter(WorldModelAdapter):
    """Adapter for IRIS pretrained Atari world models.

    Uses a VQ-VAE tokenizer and autoregressive transformer to predict
    future observations token-by-token.

    Args:
        checkpoint_path: Path to .pt file, or "pretrained" to auto-download.
        game: Atari game name (e.g. "Breakout"). Used for pretrained download.
        device: Torch device string.
        num_actions: Number of discrete actions for this game.
        obs_size: Observation resolution (default 64).
        vocab_size: VQ-VAE codebook size.
        embed_dim: Token embedding dimension.
        tokens_per_block: Tokens per transformer block (1 action + K obs tokens).
        max_blocks: Maximum transformer context length in blocks.
        num_layers: Transformer layers.
        num_heads: Transformer attention heads.
        transformer_embed_dim: Transformer hidden dimension.
    """

    def __init__(
        self,
        checkpoint_path: str = "",
        game: str = "Breakout",
        device: str = "cpu",
        num_actions: int = 4,
        obs_size: int = 64,
        vocab_size: int = 512,
        embed_dim: int = 512,
        tokens_per_block: int = 17,
        max_blocks: int = 20,
        num_layers: int = 10,
        num_heads: int = 4,
        transformer_embed_dim: int = 256,
    ):
        self.device = torch.device(device)
        self.game = game
        self.num_actions = num_actions
        self.obs_size = obs_size
        self._original_obs_shape: Optional[tuple] = None

        # Build tokenizer
        from models.tokenizer import Tokenizer, Encoder, Decoder, EncoderDecoderConfig

        enc_dec_cfg = EncoderDecoderConfig(
            resolution=obs_size,
            in_channels=3,
            z_channels=embed_dim,
            ch=64,
            ch_mult=[1, 1, 1, 1, 1],
            num_res_blocks=2,
            attn_resolutions=[8, 16],
            out_ch=3,
            dropout=0.0,
        )
        encoder = Encoder(config=enc_dec_cfg)
        decoder = Decoder(config=enc_dec_cfg)
        self.tokenizer = Tokenizer(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            encoder=encoder,
            decoder=decoder,
            with_lpips=False,
        ).to(self.device).eval()

        # Build world model
        from models.world_model import WorldModel
        from models.transformer import TransformerConfig

        wm_config = TransformerConfig(
            tokens_per_block=tokens_per_block,
            max_blocks=max_blocks,
            attention="causal",
            num_layers=num_layers,
            num_heads=num_heads,
            embed_dim=transformer_embed_dim,
            embed_pdrop=0.0,
            resid_pdrop=0.0,
            attn_pdrop=0.0,
        )
        self.world_model = WorldModel(
            obs_vocab_size=vocab_size,
            act_vocab_size=num_actions,
            config=wm_config,
        ).to(self.device).eval()

        self.num_obs_tokens = tokens_per_block - 1  # 16 for default config

        # State
        self.obs_tokens: Optional[torch.Tensor] = None
        self.keys_values_wm = None

        # Load checkpoint
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)

    def _load_checkpoint(self, path: str) -> None:
        if path == "pretrained":
            path = self._download_pretrained()

        sd = torch.load(path, map_location=self.device, weights_only=False)
        self.tokenizer.load_state_dict(_extract_state_dict(sd, "tokenizer"), strict=False)
        self.world_model.load_state_dict(_extract_state_dict(sd, "world_model"), strict=False)

    def _download_pretrained(self) -> str:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id="eloialonso/iris",
            filename=f"pretrained_models/{self.game}.pt",
        )
        return path

    def _obs_to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        """Convert uint8 HWC observation to [0, 1] CHW tensor."""
        img = torch.from_numpy(obs.astype(np.float32)).to(self.device)
        if img.ndim == 2:
            img = img.unsqueeze(-1)
        img = img.permute(2, 0, 1)  # HWC -> CHW
        if img.shape[1] != self.obs_size or img.shape[2] != self.obs_size:
            img = F.interpolate(
                img.unsqueeze(0), size=(self.obs_size, self.obs_size),
                mode="bilinear", align_corners=False,
            ).squeeze(0)
        img = img.div(255.0)  # [0, 1]
        return img

    def _tensor_to_obs(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert [0, 1] CHW tensor back to uint8 HWC numpy."""
        img = tensor.clamp(0, 1).mul(255).byte()
        img = img.permute(1, 2, 0)  # CHW -> HWC
        if self._original_obs_shape is not None:
            h, w = self._original_obs_shape[:2]
            if img.shape[0] != h or img.shape[1] != w:
                img_f = img.float().permute(2, 0, 1).unsqueeze(0)
                img_f = F.interpolate(img_f, size=(h, w), mode="bilinear", align_corners=False)
                img = img_f.squeeze(0).permute(1, 2, 0).byte()
        return img.cpu().numpy()

    @torch.no_grad()
    def reset(self, initial_obs: np.ndarray) -> None:
        self._original_obs_shape = initial_obs.shape
        obs_t = self._obs_to_tensor(initial_obs).unsqueeze(0)  # [1, C, H, W]

        # Encode to tokens
        self.obs_tokens = self.tokenizer.encode(obs_t, should_preprocess=True).tokens  # [1, K]

        # Initialize KV cache
        self.keys_values_wm = self.world_model.transformer.generate_empty_keys_values(
            n=1, max_tokens=self.world_model.config.max_tokens
        )
        # Feed initial obs tokens through transformer
        self.world_model(self.obs_tokens, past_keys_values=self.keys_values_wm)

    @torch.no_grad()
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        if self.obs_tokens is None:
            raise RuntimeError("Must call reset() before step()")

        num_passes = 1 + self.num_obs_tokens

        # Check if KV cache is full; if so, refresh
        if self.keys_values_wm.size + num_passes > self.world_model.config.max_tokens:
            self.keys_values_wm = self.world_model.transformer.generate_empty_keys_values(
                n=1, max_tokens=self.world_model.config.max_tokens
            )
            self.world_model(self.obs_tokens, past_keys_values=self.keys_values_wm)

        token = torch.tensor([[action]], dtype=torch.long, device=self.device)
        obs_tokens = []

        for k in range(num_passes):
            outputs = self.world_model(token, past_keys_values=self.keys_values_wm)

            if k == 0:
                reward = float(Categorical(logits=outputs.logits_rewards).sample().squeeze() - 1)
                done = bool(Categorical(logits=outputs.logits_ends).sample().squeeze())

            if k < self.num_obs_tokens:
                token = Categorical(logits=outputs.logits_observations).sample()
                obs_tokens.append(token)

        self.obs_tokens = torch.cat(obs_tokens, dim=1)  # [1, K]

        # Decode tokens to image
        obs_tensor = self._decode_obs_tokens()  # [1, C, H, W]
        obs_np = self._tensor_to_obs(obs_tensor.squeeze(0))

        return obs_np, reward, done

    @torch.no_grad()
    def _decode_obs_tokens(self) -> torch.Tensor:
        embedded = self.tokenizer.embedding(self.obs_tokens)  # [1, K, E]
        h = int(np.sqrt(self.num_obs_tokens))
        from einops import rearrange
        z = rearrange(embedded, "b (h w) e -> b e h w", h=h)
        rec = self.tokenizer.decode(z, should_postprocess=True)  # [1, C, H, W]
        return torch.clamp(rec, 0, 1)

    def get_latent(self) -> np.ndarray:
        if self.obs_tokens is None:
            raise RuntimeError("Must call reset() before get_latent()")
        return self.obs_tokens.flatten().cpu().float().numpy()
