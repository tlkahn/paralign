"""Configuration dataclasses for paralign."""

from __future__ import annotations

from dataclasses import dataclass, field

from paralign._dp import DPConfig


@dataclass(frozen=True)
class AlignConfig:
    """Top-level configuration for the align() function."""

    model: str | None = None
    device: str | None = None
    embed_batch_size: int = 64
    normalize: bool = True
    window: int = 50
    floor: float = 0.0
    dp: DPConfig = field(default_factory=DPConfig)
    include_similarity_matrix: bool = False
