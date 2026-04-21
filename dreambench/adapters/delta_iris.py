"""Delta-IRIS world model adapter using vmicheli/delta-iris.

Delta-IRIS extends IRIS with context-aware tokenization and delta predictions.
A pretrained Crafter checkpoint is available on HuggingFace.
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

# Add delta-iris src to path
_DELTA_IRIS_PATH = Path(__file__).parent.parent.parent / "third_party" / "delta-iris" / "src"
if str(_DELTA_IRIS_PATH) not in sys.path:
    sys.path.insert(0, str(_DELTA_IRIS_PATH))


def _extract_state_dict(state_dict: dict, module_name: str) -> OrderedDict:
    return OrderedDict(
        {k.split(".", 1)[1]: v for k, v in state_dict.items() if k.startswith(module_name)}
    )


class DeltaIRISAdapter(WorldModelAdapter):
    """Adapter for Delta-IRIS pretrained Crafter world model.

    Uses a context-aware VQ tokenizer and transformer to predict future frames.
    The decoder conditions on the previous frame and action (delta prediction).

    Args:
        checkpoint_path: Path to .pt file, or "pretrained" to auto-download.
        device: Torch device string.
        num_actions: Number of discrete actions (17 for Crafter).
        obs_size: Observation resolution (default 64).
    """

    def __init__(
        self,
        checkpoint_path: str = "",
        device: str = "cpu",
        num_actions: int = 17,
        obs_size: int = 64,
    ):
        self.device = torch.device(device)
        self.num_actions = num_actions
        self.obs_size = obs_size
        self._original_obs_shape: Optional[tuple] = None

        # Build models using Delta-IRIS architecture
        from models.tokenizer import Tokenizer, TokenizerConfig
        from models.world_model import WorldModel, WorldModelConfig
        from models.transformer import TransformerConfig
        from models.convnet import FrameCnnConfig

        tokenizer_cfg = TokenizerConfig(
            image_channels=3,
            image_size=obs_size,
            num_actions=num_actions,
            num_tokens=4,
            decoder_act_channels=4,
            codebook_size=1024,
            codebook_dim=64,
            max_codebook_updates_with_revival=0,
            encoder_config=FrameCnnConfig(
                image_channels=3 * 2 + 1,  # prev + curr + action channel
                latent_dim=64,
                num_channels=64,
                mult=[1, 1, 2, 2, 4],
                down=[1, 0, 1, 1, 0],
            ),
            decoder_config=FrameCnnConfig(
                image_channels=3,
                latent_dim=8 + 4 + 64,  # frame_cnn + act_channels + encoder latent
                num_channels=64,
                mult=[1, 1, 2, 2, 4],
                down=[1, 0, 1, 1, 0],
            ),
            frame_cnn_config=FrameCnnConfig(
                image_channels=3,
                latent_dim=8,
                num_channels=64,
                mult=[1, 1, 2, 2, 4],
                down=[1, 0, 1, 1, 0],
            ),
        )
        self.tokenizer = Tokenizer(tokenizer_cfg).to(self.device).eval()

        wm_cfg = WorldModelConfig(
            latent_vocab_size=1024,
            num_actions=num_actions,
            image_channels=3,
            image_size=obs_size,
            latents_weight=1.0,
            rewards_weight=1.0,
            ends_weight=1.0,
            two_hot_rews=True,
            transformer_config=TransformerConfig(
                tokens_per_block=6,  # 1 frame + 1 action + 4 latent tokens
                max_blocks=21,
                num_layers=3,
                num_heads=8,
                embed_dim=512,
                attention="causal",
                embed_pdrop=0.0,
                resid_pdrop=0.0,
                attn_pdrop=0.0,
            ),
            frame_cnn_config=FrameCnnConfig(
                image_channels=3,
                latent_dim=8,
                num_channels=32,
                mult=[1, 1, 2, 2, 4],
                down=[1, 0, 1, 1, 0],
            ),
        )
        self.world_model = WorldModel(wm_cfg).to(self.device).eval()

        # State
        self.obs: Optional[torch.Tensor] = None  # [1, 1, C, H, W]
        self.x: Optional[torch.Tensor] = None
        self.last_latent_token_emb: Optional[torch.Tensor] = None

        # Load checkpoint
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)

    def _load_checkpoint(self, path: str) -> None:
        if path == "pretrained":
            path = self._download_pretrained()

        sd = torch.load(path, map_location=self.device, weights_only=False)
        self.tokenizer.load_state_dict(_extract_state_dict(sd, "tokenizer"))
        self.world_model.load_state_dict(_extract_state_dict(sd, "world_model"))

    def _download_pretrained(self) -> str:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id="vmicheli/delta-iris",
            filename="last.pt",
        )
        return path

    def _obs_to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        """Convert uint8 HWC to [0, 1] CHW tensor [1, 1, C, H, W]."""
        img = torch.from_numpy(obs.astype(np.float32)).to(self.device)
        if img.ndim == 2:
            img = img.unsqueeze(-1)
        img = img.permute(2, 0, 1)  # CHW
        if img.shape[1] != self.obs_size or img.shape[2] != self.obs_size:
            img = F.interpolate(
                img.unsqueeze(0), size=(self.obs_size, self.obs_size),
                mode="bilinear", align_corners=False,
            ).squeeze(0)
        img = img.div(255.0)
        return img.unsqueeze(0).unsqueeze(0)  # [1, 1, C, H, W]

    def _tensor_to_obs(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert [1, 1, C, H, W] tensor in [0,1] to uint8 HWC numpy."""
        from einops import rearrange

        img = rearrange(tensor, "1 1 c h w -> h w c")
        img = img.clamp(0, 1).mul(255).byte()
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
        self.obs = self._obs_to_tensor(initial_obs)  # [1, 1, C, H, W]
        self.x = None
        self.last_latent_token_emb = None

        # Initialize transformer KV cache
        self.world_model.transformer.reset_kv_cache(n=1)

        # Burn in with just the initial observation (no prior actions)
        act = torch.empty(1, 0, dtype=torch.long, device=self.device)
        latent_tokens = self.tokenizer.burn_in(self.obs, act)
        self.world_model.burn_in(self.obs, act, latent_tokens, use_kv_cache=True)

    @torch.no_grad()
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        if self.obs is None:
            raise RuntimeError("Must call reset() before step()")

        from einops import rearrange
        from utils import compute_softmax_over_buckets, symexp

        # Check KV cache capacity
        if self.world_model.transformer.num_blocks_left_in_kv_cache <= 1:
            self.world_model.transformer.reset_kv_cache(n=1)
            self.last_latent_token_emb = None

        action_t = torch.tensor([[action]], dtype=torch.long, device=self.device)
        a = self.world_model.act_emb(action_t)

        # Build input sequence for transformer
        if self.last_latent_token_emb is None:
            if self.x is None:
                outputs = self.world_model(a, use_kv_cache=True)
            else:
                outputs = self.world_model(torch.cat((self.x, a), dim=1), use_kv_cache=True)
        else:
            if self.x is None:
                outputs = self.world_model(
                    torch.cat((self.last_latent_token_emb, a), dim=1), use_kv_cache=True
                )
            else:
                outputs = self.world_model(
                    torch.cat((self.last_latent_token_emb, self.x, a), dim=1), use_kv_cache=True
                )

        # Predict reward
        if self.world_model.config.two_hot_rews:
            reward = float(symexp(compute_softmax_over_buckets(outputs.logits_rewards)).flatten().cpu())
        else:
            reward = float(Categorical(logits=outputs.logits_rewards).sample().flatten() - 1)

        done = bool(Categorical(logits=outputs.logits_ends).sample().flatten().cpu())

        # Autoregressively sample latent tokens
        latent_tokens = []
        latent_token = Categorical(logits=outputs.logits_latents).sample()
        latent_tokens.append(latent_token)

        for _ in range(self.tokenizer.config.num_tokens - 1):
            latent_token_emb = self.world_model.latents_emb(latent_token)
            outputs = self.world_model(latent_token_emb, use_kv_cache=True)
            latent_token = Categorical(logits=outputs.logits_latents).sample()
            latent_tokens.append(latent_token)

        self.last_latent_token_emb = self.world_model.latents_emb(latent_token)

        # Decode latent tokens to observation
        q = self.tokenizer.quantizer.embed_tokens(torch.stack(latent_tokens, dim=-1))
        self.obs = self.tokenizer.decode(
            self.obs,
            action_t,
            rearrange(
                q, "b t (h w) (k l e) -> b t e (h k) (w l)",
                h=self.tokenizer.tokens_grid_res,
                k=self.tokenizer.token_res,
                l=self.tokenizer.token_res,
            ),
            should_clamp=True,
        )

        self.x = rearrange(self.world_model.frame_cnn(self.obs), "b 1 k e -> b k e")

        obs_np = self._tensor_to_obs(self.obs)
        return obs_np, reward, done

    def get_latent(self) -> np.ndarray:
        if self.obs is None:
            raise RuntimeError("Must call reset() before get_latent()")
        return self.obs.flatten().cpu().numpy()
