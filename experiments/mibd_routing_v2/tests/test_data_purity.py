"""CPU-only tests for data-purity building blocks (no torch).

Covers the v4 fixes for the confounds the v3 audit surfaced:
* carrier-matched safe-image selection (per-sample carrier match, was 0.55)
* neutral pool generates carrier-encoded filenames the selector can match

These modules import only numpy + Pillow (+ stdlib), so unlike ``test_v2_cpu``
they collect and run on the CPU dev box.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiments.mibd_routing_v2.data.matched_safe_images import (
    load_category_safe_pool,
    select_matched_safe_image,
    _carrier_token,
)
from experiments.mibd_routing_v2.data.neutral_carrier_renderer import (
    NeutralPoolSpec,
    NeutralRenderConfig,
    generate_neutral_safe_pool,
    render_typographic_neutral,
)


class TestCarrierToken:
    def test_normalization(self) -> None:
        assert _carrier_token("figstep") == "figstep"
        assert _carrier_token("FigStep") == "figstep"
        assert _carrier_token("typographic") == "typographic"
        assert _carrier_token("TYPOGRAPHIC") == "typographic"


class TestNeutralPoolAndSelection:
    def _build_pool(self, tmp_path: Path):
        spec = NeutralPoolSpec(
            categories=["01", "02"],
            per_category=4,  # alternates typographic/figstep -> 2 of each
            carriers=("typographic", "figstep"),
        )
        generate_neutral_safe_pool(tmp_path, spec)
        return load_category_safe_pool(tmp_path)

    def test_pool_has_both_carriers(self, tmp_path) -> None:
        pool = self._build_pool(tmp_path)
        names = [Path(p).name for p in pool.by_category["01"]]
        assert any("neutral_typographic_" in n for n in names)
        assert any("neutral_figstep_" in n for n in names)

    def test_selection_is_carrier_matched(self, tmp_path) -> None:
        pool = self._build_pool(tmp_path)
        img_t, mode_t = select_matched_safe_image(pool, category="01", index=0, carrier="typographic")
        img_f, mode_f = select_matched_safe_image(pool, category="01", index=0, carrier="figstep")
        assert mode_t == "carrier_matched"
        assert mode_f == "carrier_matched"
        assert "neutral_typographic_" in Path(img_t).name
        assert "neutral_figstep_" in Path(img_f).name

    def test_selection_deterministic(self, tmp_path) -> None:
        pool = self._build_pool(tmp_path)
        a = select_matched_safe_image(pool, category="02", index=3, carrier="figstep")
        b = select_matched_safe_image(pool, category="02", index=3, carrier="figstep")
        assert a == b

    def test_backward_compatible_without_carrier(self, tmp_path) -> None:
        pool = self._build_pool(tmp_path)
        img, mode = select_matched_safe_image(pool, category="01", index=0)
        assert mode == "category_matched"
        assert img is not None

    def test_empty_pool_returns_none(self) -> None:
        pool = load_category_safe_pool(None)
        img, mode = select_matched_safe_image(pool, category="01", index=0, carrier="figstep")
        assert img is None
        assert mode == "none"

    def test_full_match_rate_when_carrier_aware(self, tmp_path) -> None:
        """Simulate the builder loop: carrier-aware selection => 100% match."""
        pool = self._build_pool(tmp_path)
        risk_carriers = ["typographic", "figstep"] * 10
        matched = 0
        for i, carrier in enumerate(risk_carriers):
            img, mode = select_matched_safe_image(pool, category="01", index=i, carrier=carrier)
            token = _carrier_token(carrier)
            if f"neutral_{token}_" in Path(img).name:
                matched += 1
        assert matched == len(risk_carriers)  # was ~0.55 in v3


class TestTextDensity:
    """text_repeat raises pixel variance to match dense risk carriers (std ~73)."""

    def _std(self, image) -> float:
        return float(np.asarray(image.resize((336, 336)), dtype=float).std())

    def test_repeat_raises_pixel_std(self) -> None:
        cfg_sparse = NeutralRenderConfig(size=(500, 500), font_size=40, text_repeat=1)
        cfg_dense = NeutralRenderConfig(size=(500, 500), font_size=40, text_repeat=5)
        sparse = self._std(render_typographic_neutral("Steps to water a plant", cfg_sparse))
        dense = self._std(render_typographic_neutral("Steps to water a plant", cfg_dense))
        assert dense > sparse
        # dense should land in the risk-contrast band (well above sparse ~23)
        assert dense > 45.0

    def test_repeat_one_is_backward_compatible(self) -> None:
        cfg = NeutralRenderConfig(size=(500, 500), font_size=40, text_repeat=1)
        img = render_typographic_neutral("Steps to water a plant", cfg)
        assert img.size == (500, 500)
