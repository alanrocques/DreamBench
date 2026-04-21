"""DIAMOND world model adapter using eloialonso/diamond.

DIAMOND (DIffusion As a Model Of eNvironment Dreams) is a diffusion-based
world model that predicts next frames by denoising, conditioned on a window
of past observations and actions. Pretrained checkpoints for all 26 Atari
100k games are available on HuggingFace.
"""

import sys
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from dreambench.adapters.base import WorldModelAdapter

# Add diamond src to path
_DIAMOND_PATH = Path(__file__).parent.parent.parent / "third_party" / "diamond" / "src"
if str(_DIAMOND_PATH) not in sys.path:
    sys.path.insert(0, str(_DIAMOND_PATH))


def _extract_state_dict(state_dict: dict, module_name: str) -> OrderedDict:
    return OrderedDict(
        {k.split(".", 1)[1]: v for k, v in state_dict.items() if k.startswith(module_name)}
    )


class DIAMONDAdapter(WorldModelAdapter):
    """Adapter for DIAMOND pretrained Atari world models.

    Uses diffusion sampling conditioned on past frames to predict next observations.
    Downloads pretrained weights from HuggingFace if checkpoint_path is "pretrained".

    Args:
        checkpoint_path: Path to .pt file, or "pretrained" to auto-download.
        game: Atari game name (e.g. "Breakout"). Used for pretrained download.
        device: Torch device string.
        num_actions: Number of discrete actions for this game.
        obs_size: Observation resolution (default 64).
        num_steps_conditioning: Number of past frames the denoiser conditions on.
        num_steps_denoising: Diffusion sampling steps.
    """

    def __init__(
        self,
        checkpoint_path: str = "",
        game: str = "Breakout",
        device: str = "cpu",
        num_actions: int = 4,
        obs_size: int = 64,
        num_steps_conditioning: int = 4,
        num_steps_denoising: int = 3,
    ):
        self.device = torch.device(device)
        self.game = game
        self.num_actions = num_actions
        self.obs_size = obs_size
        self.num_steps_conditioning = num_steps_conditioning
        self._original_obs_shape: Optional[tuple] = None

        # Build model components
        from models.diffusion import Denoiser, DenoiserConfig, InnerModelConfig
        from models.diffusion import DiffusionSampler, DiffusionSamplerConfig
        from models.rew_end_model import RewEndModel, RewEndModelConfig

        denoiser_cfg = DenoiserConfig(
            inner_model=InnerModelConfig(
                img_channels=3,
                num_steps_conditioning=num_steps_conditioning,
                cond_channels=256,
                depths=[2, 2, 2, 2],
                channels=[64, 64, 64, 64],
                attn_depths=[0, 0, 0, 0],
                num_actions=num_actions,
            ),
            sigma_data=0.5,
            sigma_offset_noise=0.3,
        )
        self.denoiser = Denoiser(denoiser_cfg).to(self.device).eval()

        sampler_cfg = DiffusionSamplerConfig(
            num_steps_denoising=num_steps_denoising,
            sigma_min=2e-3,
            sigma_max=5.0,
            rho=7,
            order=1,
        )
        self.sampler = DiffusionSampler(self.denoiser, sampler_cfg)

        rew_end_cfg = RewEndModelConfig(
            lstm_dim=512,
            img_channels=3,
            img_size=obs_size,
            cond_channels=128,
            depths=[2, 2, 2, 2],
            channels=[32, 32, 32, 32],
            attn_depths=[0, 0, 0, 0],
            num_actions=num_actions,
        )
        self.rew_end_model = RewEndModel(rew_end_cfg).to(self.device).eval()

        # State buffers (initialized on reset)
        self.obs_buffer: Optional[torch.Tensor] = None  # [1, T, C, H, W]
        self.act_buffer: Optional[torch.Tensor] = None  # [1, T]
        self.hx_rew_end: Optional[torch.Tensor] = None
        self.cx_rew_end: Optional[torch.Tensor] = None

        # Load checkpoint
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)

    def _load_checkpoint(self, path: str) -> None:
        if path == "pretrained":
            path = self._download_pretrained()

        sd = torch.load(path, map_location=self.device, weights_only=False)
        self.denoiser.load_state_dict(_extract_state_dict(sd, "denoiser"))
        self.rew_end_model.load_state_dict(_extract_state_dict(sd, "rew_end_model"))

    def _download_pretrained(self) -> str:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id="eloialonso/diamond",
            filename=f"atari_100k/models/{self.game}.pt",
        )
        return path

    def _obs_to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        """Convert uint8 HWC observation to [-1, 1] CHW tensor."""
        img = torch.from_numpy(obs.astype(np.float32)).to(self.device)
        if img.ndim == 2:
            img = img.unsqueeze(-1)
        # HWC -> CHW
        img = img.permute(2, 0, 1)
        # Resize if needed
        if img.shape[1] != self.obs_size or img.shape[2] != self.obs_size:
            img = F.interpolate(
                img.unsqueeze(0), size=(self.obs_size, self.obs_size),
                mode="bilinear", align_corners=False,
            ).squeeze(0)
        # [0, 255] -> [-1, 1]
        img = img.div(255).mul(2).sub(1)
        return img

    def _tensor_to_obs(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert [-1, 1] CHW tensor back to uint8 HWC numpy."""
        # [-1, 1] -> [0, 255]
        img = tensor.add(1).div(2).mul(255).clamp(0, 255).byte()
        # CHW -> HWC
        img = img.permute(1, 2, 0)
        # Resize back to original shape if needed
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
        obs_t = self._obs_to_tensor(initial_obs)  # [C, H, W]

        # Fill conditioning buffer with repeated initial frame
        self.obs_buffer = obs_t.unsqueeze(0).unsqueeze(0).repeat(
            1, self.num_steps_conditioning, 1, 1, 1
        )  # [1, T, C, H, W]
        self.act_buffer = torch.zeros(
            1, self.num_steps_conditioning, dtype=torch.long, device=self.device
        )  # [1, T]

        # Initialize LSTM hidden state for reward/end model
        self.hx_rew_end = torch.zeros(1, 1, 512, device=self.device)
        self.cx_rew_end = torch.zeros(1, 1, 512, device=self.device)

    @torch.no_grad()
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        if self.obs_buffer is None:
            raise RuntimeError("Must call reset() before step()")

        # Set the last action in the buffer
        self.act_buffer[:, -1] = action

        # Predict next observation via diffusion
        next_obs, _ = self.sampler.sample(self.obs_buffer, self.act_buffer)
        # next_obs: [1, C, H, W]

        # Predict reward and end
        from torch.distributions.categorical import Categorical

        logits_rew, logits_end, (self.hx_rew_end, self.cx_rew_end) = (
            self.rew_end_model.predict_rew_end(
                self.obs_buffer[:, -1:],          # [1, 1, C, H, W]
                self.act_buffer[:, -1:],           # [1, 1]
                next_obs.unsqueeze(1),             # [1, 1, C, H, W]
                (self.hx_rew_end, self.cx_rew_end),
            )
        )
        reward = float(Categorical(logits=logits_rew).sample().squeeze() - 1)
        done = bool(Categorical(logits=logits_end).sample().squeeze())

        # Roll buffers forward
        self.obs_buffer = self.obs_buffer.roll(-1, dims=1)
        self.act_buffer = self.act_buffer.roll(-1, dims=1)
        self.obs_buffer[:, -1] = next_obs
        self.act_buffer[:, -1] = 0  # placeholder for next step

        # Convert to numpy
        obs_np = self._tensor_to_obs(next_obs.squeeze(0))
        return obs_np, reward, done

    def get_latent(self) -> np.ndarray:
        if self.obs_buffer is None:
            raise RuntimeError("Must call reset() before get_latent()")
        # Use flattened obs buffer as latent representation
        return self.obs_buffer.flatten().cpu().numpy()
