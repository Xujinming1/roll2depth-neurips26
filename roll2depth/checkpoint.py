"""Strict loading and privacy-preserving export of model tensor weights."""

from pathlib import Path

import torch


# Historical Lightning checkpoints pickle an empty __main__.Args container.
# Allow only this inert compatibility type, retaining torch's weights-only loader.
LegacyArgs = type("Args", (), {"__module__": "__main__"})
UNUSED_LEGACY_PREFIXES = ("decoder.", "reg_head.", "conf_head.")


def read_state(path, allow_legacy=False):
    kwargs = dict(map_location="cpu", weights_only=True, mmap=True)
    if allow_legacy:
        with torch.serialization.safe_globals([LegacyArgs]):
            checkpoint = torch.load(path, **kwargs)
    else:
        checkpoint = torch.load(path, **kwargs)
    state = checkpoint.get("state_dict", checkpoint)
    if state and all(key.startswith("model.") for key in state):
        state = {key[6:]: value for key, value in state.items()}
    return state, checkpoint


def load_weights(model, path, allow_legacy=False):
    state, checkpoint = read_state(path, allow_legacy)
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    extra = sorted(set(state) - set(expected))
    forbidden = [key for key in extra if not (allow_legacy and key.startswith(UNUSED_LEGACY_PREFIXES))]
    if missing or forbidden:
        raise ValueError(f"Checkpoint mismatch: missing={missing[:12]}, unexpected={forbidden[:12]}")
    model.load_state_dict({key: state[key] for key in expected}, strict=True)
    return {"loaded_tensors": len(expected), "discarded_inactive_tensors": len(extra)}


def export_legacy(source, destination):
    state, checkpoint = read_state(source, allow_legacy=True)
    kept = {key: value for key, value in state.items()
            if not key.startswith(UNUSED_LEGACY_PREFIXES)}
    if not kept or not all(isinstance(value, torch.Tensor) for value in kept.values()):
        raise ValueError("Expected a model state dictionary of tensors.")
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": 1,
        "architecture": "dinov2_large_patch16_roll2depth64",
        "epoch": int(checkpoint.get("epoch", -1)),
        "global_step": int(checkpoint.get("global_step", -1)),
        "state_dict": kept,
    }, destination)
    return {"exported_tensors": len(kept), "discarded_inactive_tensors": len(state) - len(kept)}
