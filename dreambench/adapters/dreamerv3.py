"""DreamerV3 world model adapter using NM512/dreamerv3-torch."""

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from dreambench.adapters.base import WorldModelAdapter

# Add dreamerv3-torch to path
_D3_PATH = Path(__file__).parent.parent.parent / "third_party" / "dreamerv3-torch"
if str(_D3_PATH) not in sys.path:
    sys.path.insert(0, str(_D3_PATH))


class DreamerV3Adapter(WorldModelAdapter):
    """Adapter for DreamerV3 using NM512/dreamerv3-torch.

    Loads a trained checkpoint and runs open-loop imagination:
    - reset(obs): encodes one real frame into the RSSM latent state
    - step(action): predicts forward using the PRIOR (no real observations)
    - get_latent(): returns the concatenated (h, z) latent state

    Requires a checkpoint trained with dreamerv3-torch. See
    docs/training_dreamerv3.md for training instructions.
    """

    def __init__(
        self,
        checkpoint_path: str,
        obs_size: int = 64,
        action_size: int = 18,
        device: str = "cpu",
        grayscale: bool = False,
        dyn_deter: int = 512,
        dyn_stoch: int = 32,
        dyn_hidden: int = 512,
        dyn_discrete: int = 32,
        dyn_rec_depth: int = 1,
        cnn_depth: int = 32,
        kernel_size: int = 4,
        minres: int = 4,
        units: int = 512,
    ):
        self.obs_size = obs_size
        self.action_size = action_size
        self.device = torch.device(device)
        self.grayscale = grayscale
        self._original_obs_shape = None

        # Store architecture params
        self.dyn_deter = dyn_deter
        self.dyn_stoch = dyn_stoch
        self.dyn_hidden = dyn_hidden
        self.dyn_discrete = dyn_discrete
        self.dyn_rec_depth = dyn_rec_depth

        channels = 1 if grayscale else 3
        self.feat_size = dyn_stoch * dyn_discrete + dyn_deter

        # Build world model components
        from networks import ConvEncoder, ConvDecoder, RSSM, MLP

        self.encoder = ConvEncoder(
            input_shape=(obs_size, obs_size, channels),
            depth=cnn_depth, act="SiLU", norm=True,
            kernel_size=kernel_size, minres=minres,
        ).to(self.device)

        self.rssm = RSSM(
            stoch=dyn_stoch, deter=dyn_deter, hidden=dyn_hidden,
            rec_depth=dyn_rec_depth, discrete=dyn_discrete,
            act="SiLU", norm=True, mean_act="none", std_act="sigmoid2",
            min_std=0.1, unimix_ratio=0.01, initial="learned",
            num_actions=action_size, embed=self.encoder.outdim,
            device=str(self.device),
        ).to(self.device)

        self.decoder = ConvDecoder(
            feat_size=self.feat_size,
            shape=(channels, obs_size, obs_size),
            depth=cnn_depth, act="SiLU", norm=True,
            kernel_size=kernel_size, minres=minres,
        ).to(self.device)

        self.reward_head = MLP(
            inp_dim=self.feat_size, shape=(255,), layers=2, units=units,
            act="SiLU", norm=True, dist="symlog_disc", outscale=0.0,
            device=str(self.device), name="reward",
        ).to(self.device)

        self.cont_head = MLP(
            inp_dim=self.feat_size, shape=(), layers=2, units=units,
            act="SiLU", norm=True, dist="binary", outscale=1.0,
            device=str(self.device), name="cont",
        ).to(self.device)

        # Load checkpoint
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)

        # Set all to eval mode
        self.encoder.eval()
        self.rssm.eval()
        self.decoder.eval()
        self.reward_head.eval()
        self.cont_head.eval()

        # Latent state
        self._state = None

    def _load_checkpoint(self, path: str) -> None:
        """Load world model weights from a dreamerv3-torch checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        state_dict = checkpoint["agent_state_dict"]

        # Map checkpoint keys to our component names
        # dreamerv3-torch uses _wm.encoder, _wm.dynamics, _wm.heads.decoder, etc.
        key_mapping = {
            "_wm.encoder.": ("encoder", ""),
            "_wm.dynamics.": ("rssm", ""),
            "_wm.heads.decoder.": ("decoder", ""),
            "_wm.heads.reward.": ("reward_head", ""),
            "_wm.heads.cont.": ("cont_head", ""),
        }

        component_dicts = {name: {} for _, (name, _) in key_mapping.items()}

        for k, v in state_dict.items():
            for prefix, (component, new_prefix) in key_mapping.items():
                if k.startswith(prefix):
                    new_key = new_prefix + k[len(prefix):]
                    component_dicts[component][new_key] = v
                    break

        # Load into each component
        for name, sd in component_dicts.items():
            if sd:
                component = getattr(self, name)
                component.load_state_dict(sd, strict=False)

    def _preprocess(self, obs: np.ndarray) -> torch.Tensor:
        """Convert uint8 observation to model input: [1, 1, H, W, C] float in [0, 1]."""
        img = obs.astype(np.float32) / 255.0

        if self.grayscale and img.ndim == 3 and img.shape[2] == 3:
            img = np.mean(img, axis=2, keepdims=True)

        # Resize to model's expected resolution
        img_t = torch.from_numpy(img).to(self.device)
        if img_t.ndim == 2:
            img_t = img_t.unsqueeze(-1)
        # img_t is (H, W, C), need to resize
        if img_t.shape[0] != self.obs_size or img_t.shape[1] != self.obs_size:
            # Use F.interpolate: needs (N, C, H, W)
            img_t = img_t.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
            img_t = F.interpolate(
                img_t, size=(self.obs_size, self.obs_size),
                mode="bilinear", align_corners=False,
            )
            img_t = img_t.squeeze(0).permute(1, 2, 0)  # (H, W, C)

        # dreamerv3-torch expects [batch, time, H, W, C]
        return img_t.unsqueeze(0).unsqueeze(0)

    def _postprocess(self, predicted: torch.Tensor) -> np.ndarray:
        """Convert model output [1, 1, H, W, C] back to uint8 at original resolution."""
        # predicted is in roughly [0, 1] range (decoder adds 0.5)
        img = predicted.squeeze(0).squeeze(0)  # (H, W, C)

        if self._original_obs_shape is not None:
            h, w = self._original_obs_shape[:2]
            orig_c = self._original_obs_shape[2] if len(self._original_obs_shape) > 2 else 1
            # Resize back
            if img.shape[0] != h or img.shape[1] != w:
                img = img.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
                img = F.interpolate(
                    img, size=(h, w), mode="bilinear", align_corners=False,
                )
                img = img.squeeze(0).permute(1, 2, 0)  # (H, W, C)
            # If model is grayscale but original was RGB, repeat channels
            if self.grayscale and orig_c == 3:
                img = img.repeat(1, 1, 3)

        img = (img.clamp(0, 1) * 255).byte()
        return img.cpu().numpy()

    @torch.no_grad()
    def reset(self, initial_obs: np.ndarray) -> None:
        self._original_obs_shape = initial_obs.shape
        img = self._preprocess(initial_obs)  # [1, 1, H, W, C]

        # Encode observation
        embed = self.encoder(img)  # [1, 1, embed_size]

        # Initialize RSSM state
        state = self.rssm.initial(1)
        # Move initial state to device
        state = {k: v.to(self.device) for k, v in state.items()}

        # Run obs_step to get posterior (conditions on the real observation)
        action = torch.zeros(1, self.action_size, device=self.device)
        is_first = torch.ones(1, 1, device=self.device)
        post, _ = self.rssm.obs_step(state, action, embed[:, 0], is_first[:, 0])
        self._state = post

    @torch.no_grad()
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        if self._state is None:
            raise RuntimeError("Must call reset() before step()")

        # One-hot encode action
        action_oh = torch.zeros(1, self.action_size, device=self.device)
        action_oh[0, action] = 1.0

        # RSSM forward: predict next state using PRIOR (no real obs)
        self._state = self.rssm.img_step(self._state, action_oh, sample=True)

        # Get feature vector
        feat = self.rssm.get_feat(self._state)  # [1, feat_size]

        # Decode predicted observation
        feat_for_decoder = feat.unsqueeze(1)  # [1, 1, feat_size]
        predicted_img = self.decoder(feat_for_decoder)  # [1, 1, H, W, C]
        obs = self._postprocess(predicted_img)

        # Predict reward
        reward_dist = self.reward_head(feat)
        reward = float(reward_dist.mode().squeeze().cpu())

        # Predict continuation
        cont_dist = self.cont_head(feat)
        done = float(cont_dist.mode().squeeze().cpu()) < 0.5

        return obs, reward, done

    def get_latent(self) -> np.ndarray:
        if self._state is None:
            raise RuntimeError("Must call reset() before get_latent()")
        feat = self.rssm.get_feat(self._state)
        return feat.squeeze(0).cpu().numpy()
