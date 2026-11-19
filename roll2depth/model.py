"""The single-layer DINOv2-L / Roll2Depth architecture used by version45."""

import torch
from torch import nn
import torch.nn.functional as F
import timm
from timm.layers import resample_abs_pos_embed


def create_backbone(pretrained=True, weights=None):
    # Load the original patch14 weights before the version45 patch16 resampling.
    backbone = timm.create_model(
        "vit_large_patch14_dinov2", pretrained=pretrained and weights is None,
        dynamic_img_size=True,
    )
    if weights is not None:
        if str(weights).endswith(".safetensors"):
            from safetensors.torch import load_file
            state = load_file(str(weights))
        else:
            state = torch.load(weights, map_location="cpu", weights_only=True)
            state = state.get("state_dict", state)
        backbone.load_state_dict(state, strict=True)

    old = backbone.patch_embed.proj
    weight = F.interpolate(old.weight, size=(16, 16), mode="bicubic", align_corners=False)
    weight = weight * (old.kernel_size[0] / 16) * (old.kernel_size[1] / 16)
    backbone.patch_embed.proj = nn.Conv2d(3, 1024, 16, stride=16, bias=old.bias is not None)
    with torch.no_grad():
        backbone.patch_embed.proj.weight.copy_(weight)
        if old.bias is not None:
            backbone.patch_embed.proj.bias.copy_(old.bias)
    backbone.patch_embed.patch_size = (16, 16)
    backbone.patch_embed.img_size = (384, 384)
    backbone.patch_embed.grid_size = (24, 24)
    backbone.pos_embed = nn.Parameter(resample_abs_pos_embed(
        backbone.pos_embed, (24, 24), num_prefix_tokens=backbone.num_prefix_tokens,
    ))
    return backbone


def exp_attractor(dx, alpha=300, gamma=2):
    return torch.exp(-alpha * torch.abs(dx) ** gamma) * dx


def log_binom(n, k, eps=1e-7):
    n, k = n + eps, k + eps
    return n * torch.log(n) - k * torch.log(k) - (n - k) * torch.log(n - k + eps)


class LogBinomial(nn.Module):
    def __init__(self, n_classes=64):
        super().__init__()
        self.K = n_classes
        self.register_buffer("k_idx", torch.arange(n_classes).view(1, -1, 1, 1))
        self.register_buffer("K_minus_1", torch.tensor([n_classes - 1.0]).view(1, -1, 1, 1))

    def forward(self, x, t=1.0, eps=1e-4):
        if x.ndim == 3:
            x = x.unsqueeze(1)
        other = (1 - x).clamp(eps, 1)
        x = x.clamp(eps, 1)
        y = log_binom(self.K_minus_1, self.k_idx)
        y = y + self.k_idx * torch.log(x) + (self.K - 1 - self.k_idx) * torch.log(other)
        return torch.softmax(y / t, dim=1)


class AffineBins(nn.Module):
    def __init__(self, in_features=1024, n_bins=64, mlp_dim=256, embedding_dim=128):
        super().__init__()
        self.seed_bins = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim), nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, n_bins), nn.Softplus(),
        )
        self.projector = nn.Sequential(
            nn.Linear(in_features, mlp_dim), nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, embedding_dim),
        )
        # Keep the original spelling so existing weights load without renaming.
        self.attactor = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim), nn.ReLU(inplace=True), nn.Linear(mlp_dim, 1),
        )
        self.conf_head = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim), nn.GELU(),
            nn.Linear(mlp_dim, 1), nn.Softmax(dim=-2),
        )
        self.pt_pred = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim), nn.GELU(),
            nn.Linear(embedding_dim, 4), nn.Softplus(),
        )
        self.log_binomial_transform = LogBinomial(n_bins)

    def forward(self, features):
        cls = self.projector(features[:, :1])
        patches = self.projector(features[:, 1:])
        centers = self.seed_bins(cls)
        attractors = self.attactor(cls + patches)
        delta = torch.zeros_like(centers)
        # Preserve version45's accumulation order for checkpoint parity.
        for i in range(attractors.shape[1]):
            delta = delta + exp_attractor(attractors[:, i].unsqueeze(-1) - centers)
        centers = (centers + delta).squeeze(1).unsqueeze(-1).unsqueeze(-1)
        confidence = self.conf_head(patches)
        combined = (patches * confidence).sum(dim=1).unsqueeze(1)
        pt = self.pt_pred(torch.cat((combined, cls), dim=-1).squeeze(1))[:, :, None, None]
        p, t = pt[:, :2] + 1e-4, pt[:, 2:] + 1e-4
        p = p[:, 0] / (p[:, 0] + p[:, 1])
        t = (t[:, 0] / (t[:, 0] + t[:, 1])).unsqueeze(1)
        t = (50 - 1e-4) * t + 1e-4
        return (self.log_binomial_transform(p, t) * centers).sum(dim=1).reshape(-1)


class Roll2DepthModel(nn.Module):
    def __init__(self, pretrained=True, backbone_weights=None):
        super().__init__()
        self.input_projection = nn.Conv2d(4, 3, 1, bias=False)
        self.vit_backbone = create_backbone(pretrained, backbone_weights)
        self.scalebins = AffineBins()
        self.shiftbins = AffineBins()

    def forward(self, relative_depth, stripe):
        if relative_depth.ndim == 3:
            relative_depth = relative_depth.unsqueeze(1)
        if stripe.shape[1] != 3 or relative_depth.shape[1] != 1:
            raise ValueError("Expected a one-channel relative depth and a three-channel stripe cue.")
        if relative_depth.shape[-2:] != (384, 384) or stripe.shape[-2:] != (384, 384):
            raise ValueError("Resize both model inputs to 384 x 384.")
        x = self.input_projection(torch.cat((relative_depth, stripe), dim=1))
        features = self.vit_backbone.forward_features(x)
        values = torch.stack((self.scalebins(features), self.shiftbins(features)), dim=1)
        # Explicit batch dimensions avoid the legacy [B,B,H,W] broadcasting bug.
        depth = relative_depth * values[:, 0, None, None, None] + values[:, 1, None, None, None]
        return depth, values


def predict_depth(model, relative_depth, stripe):
    """Infer global parameters at 384px and apply them at the original resolution."""
    if relative_depth.ndim == 3:
        relative_depth = relative_depth.unsqueeze(1)
    rd = F.interpolate(relative_depth.float(), (384, 384), mode="bilinear", align_corners=True)
    cue = F.interpolate(stripe.float(), (384, 384), mode="bilinear", align_corners=True)
    _, values = model(rd, cue)
    depth = relative_depth * values[:, 0, None, None, None] + values[:, 1, None, None, None]
    return depth, values
