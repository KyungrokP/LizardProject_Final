from __future__ import annotations

import math
import site
import sys
import types
from pathlib import Path

import torch
from torch import nn


def load_tapnext_torch_model(
    checkpoint_path: str | Path,
    image_size: tuple[int, int] = (256, 256),
    device: str = "cpu",
    sketch_tokens: int | None = None,
    sketch_seed: int = 0,
) -> torch.nn.Module:
    """Loads TAPNext torch model while bypassing tapnet tensorflow import side effects."""
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"TAPNext checkpoint not found: {checkpoint}")

    if device != "cpu" and device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device but torch CUDA is not available")

    pkg_root = _find_tapnet_pkg_root()
    _install_tapnet_shim(pkg_root)
    tapnext_torch_mod = _load_tapnext_torch_module(pkg_root, device=device)

    from tapnet.tapnext.tapnext_torch_utils import restore_model_from_jax_checkpoint

    model = tapnext_torch_mod.TAPNext(image_size=image_size)
    model = restore_model_from_jax_checkpoint(model, str(checkpoint))
    if sketch_tokens is not None and sketch_tokens > 0:
        _enable_vit_token_sketching(
            model=model,
            sketch_tokens=int(sketch_tokens),
            sketch_seed=int(sketch_seed),
        )
    model = model.to(device)
    model.eval()
    return model


def _find_tapnet_pkg_root() -> Path:
    search_paths = list(site.getsitepackages())
    try:
        search_paths.append(site.getusersitepackages())
    except Exception:
        pass

    for base in search_paths:
        base_path = Path(base)
        candidate = base_path / "tapnet" / "tapnext" / "tapnext_torch.py"
        if candidate.exists():
            return base_path / "tapnet"
    raise FileNotFoundError("Could not locate installed tapnet package")


def _install_tapnet_shim(pkg_root: Path) -> None:
    existing = sys.modules.get("tapnet")
    pkg_root_s = str(pkg_root)
    if existing is None:
        shim = types.ModuleType("tapnet")
        shim.__path__ = [pkg_root_s]
        sys.modules["tapnet"] = shim
        return

    module_path = list(getattr(existing, "__path__", []))
    if pkg_root_s not in module_path:
        module_path.append(pkg_root_s)
    existing.__path__ = module_path


def _load_tapnext_torch_module(pkg_root: Path, device: str) -> types.ModuleType:
    module_name = f"tapnet.tapnext.tapnext_torch_{device}_patched"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached

    source_path = pkg_root / "tapnext" / "tapnext_torch.py"
    source = source_path.read_text()
    if device == "cpu":
        source = source.replace("device='cuda'", "device='cpu'")

    module = types.ModuleType(module_name)
    module.__file__ = str(source_path)
    module.__package__ = "tapnet.tapnext"
    sys.modules[module_name] = module
    exec(compile(source, str(source_path), "exec"), module.__dict__)
    return module


class _SketchedViTBlock(nn.Module):
    """Approximates full token attention by sketching tokens to a smaller set.

    This wraps a pretrained ViT encoder block and keeps its weights unchanged.
    """

    def __init__(self, base: nn.Module, sketch_tokens: int, seed: int) -> None:
        super().__init__()
        self.base = base
        self.sketch_tokens = sketch_tokens
        self.seed = seed
        self._sketch_cache: dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C]
        if x.ndim != 3:
            return self.base(x)
        n = int(x.shape[1])
        k = min(self.sketch_tokens, n)
        if k >= n or k <= 0:
            return self.base(x)

        sketch = self._get_sketch(n=n, k=k, device=x.device, dtype=x.dtype)
        x_sk = torch.einsum("bnc,nk->bkc", x, sketch)  # compress tokens
        y_sk = self.base(x_sk)
        y = torch.einsum("bkc,nk->bnc", y_sk, sketch)  # project back
        # Residual correction keeps identity behavior stable.
        x_hat = torch.einsum("bkc,nk->bnc", x_sk, sketch)
        return x + (y - x_hat)

    def _get_sketch(
        self,
        n: int,
        k: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (n, device, dtype)
        cached = self._sketch_cache.get(key)
        if cached is not None and cached.shape == (n, k):
            return cached

        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.seed + (n * 1009) + (k * 9176))
        signs = torch.randint(0, 2, (n, k), generator=gen, dtype=torch.int8).to(torch.float32)
        sketch = (signs * 2.0 - 1.0) / math.sqrt(float(k))
        sketch = sketch.to(device=device, dtype=dtype)
        self._sketch_cache[key] = sketch
        return sketch


def _enable_vit_token_sketching(
    model: nn.Module,
    sketch_tokens: int,
    sketch_seed: int,
) -> None:
    if not hasattr(model, "blocks"):
        return
    for idx, blk in enumerate(getattr(model, "blocks")):
        vit_block = getattr(blk, "vit_block", None)
        if vit_block is None:
            continue
        blk.vit_block = _SketchedViTBlock(
            base=vit_block,
            sketch_tokens=sketch_tokens,
            seed=sketch_seed + (idx * 100_003),
        )
