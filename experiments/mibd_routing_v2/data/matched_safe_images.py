"""Category-matched safe-image selection (data purity upgrade).

The original ``experiments.mibd_routing.data.convert_phase2a`` picks a safe
counterpart image by ``index % len(pool)``, i.e. independent of the risk
item's category. That weakens the paired triplet: a clean ``(q, v_safe,
v_risk)`` pair should differ *only* in whether risk evidence is present, while
staying matched on scene/category, otherwise a probe can exploit the
category/source as a confound.

This v2 helper organizes a safe-image directory **by category** (one
subdirectory per category) and selects a same-category benign image for each
risk item, falling back to a global pool only when the category is missing.

CPU-only, standard library + pathlib. No model, no GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass
class CategorySafeImagePool:
    """A safe-image pool indexed by category subdirectory name.

    Directory layout expected::

        safe_image_dir/
            weapons/   img001.png ...
            cyber/     img002.png ...
            ...

    Images placed directly under ``safe_image_dir`` (no category subdir) form a
    global fallback pool used when a requested category has no images.
    """

    by_category: dict[str, list[str]]
    global_pool: list[str]

    @property
    def total(self) -> int:
        return sum(len(v) for v in self.by_category.values()) + len(self.global_pool)

    def is_empty(self) -> bool:
        return self.total == 0


def load_category_safe_pool(safe_image_dir: str | Path | None) -> CategorySafeImagePool:
    """Scan ``safe_image_dir`` into a category-indexed pool."""
    by_category: dict[str, list[str]] = {}
    global_pool: list[str] = []
    if safe_image_dir is None:
        return CategorySafeImagePool(by_category=by_category, global_pool=global_pool)
    base = Path(safe_image_dir)
    if not base.exists():
        return CategorySafeImagePool(by_category=by_category, global_pool=global_pool)

    for path in sorted(base.iterdir()):
        if path.is_dir():
            images = [
                str(p)
                for p in sorted(path.rglob("*"))
                if p.suffix.lower() in _IMAGE_SUFFIXES
            ]
            if images:
                by_category[path.name] = images
        elif path.suffix.lower() in _IMAGE_SUFFIXES:
            global_pool.append(str(path))
    return CategorySafeImagePool(by_category=by_category, global_pool=global_pool)


def _carrier_token(carrier: str) -> str:
    """Normalize a carrier label to the token used in neutral filenames."""
    c = carrier.lower().replace("-", "_")
    # CarrierType values: "typographic", "figstep". Filenames: neutral_<token>_NN.png
    if "figstep" in c:
        return "figstep"
    if "typograph" in c:
        return "typographic"
    return c


def _filter_by_carrier(images: list[str], carrier: str | None) -> list[str]:
    """Keep only images whose filename encodes the requested carrier.

    Neutral safe images are named ``neutral_<carrier>_NN.png``; when ``carrier``
    is given we restrict to the matching subset so the safe control shares the
    per-sample carrier FORMAT of its paired risk image. Returns the full list
    unchanged when ``carrier`` is None or no image matches (caller then handles
    the fallback / mismatch).
    """
    if carrier is None:
        return images
    token = _carrier_token(carrier)
    matched = [p for p in images if f"neutral_{token}_" in Path(p).name.lower()]
    return matched if matched else images


def select_matched_safe_image(
    pool: CategorySafeImagePool,
    category: str,
    index: int,
    carrier: str | None = None,
) -> tuple[str | None, str]:
    """Pick a category-matched (and optionally carrier-matched) safe image.

    Returns ``(image_path_or_None, mode)`` where mode is one of:
      * ``"carrier_matched"``  -- same-category AND same-carrier image found.
      * ``"category_matched"`` -- same-category image found (carrier not matched
        or not requested).
      * ``"global_fallback"``  -- no category match, used the global pool.
      * ``"none"``             -- pool empty; caller should generate a placeholder.

    Deterministic given (category, index, carrier): selection is index-based, so
    the builder stays reproducible under a fixed seed/order.
    """
    category_images = pool.by_category.get(category)
    if category_images:
        if carrier is not None:
            carrier_images = _filter_by_carrier(category_images, carrier)
            if carrier_images is not category_images and carrier_images:
                return carrier_images[index % len(carrier_images)], "carrier_matched"
        return category_images[index % len(category_images)], "category_matched"
    if pool.global_pool:
        return pool.global_pool[index % len(pool.global_pool)], "global_fallback"
    return None, "none"
