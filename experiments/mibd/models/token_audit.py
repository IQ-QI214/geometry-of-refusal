"""Token-position auditing helpers.

The Phase 1 claims depend on mapping relative positions like -5/-1 to real
chat-template tokens. This module records those mappings explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class TokenAudit:
    sequence_length: int
    tokens: tuple[str, ...]
    relative_to_absolute: dict[int, int]
    user_end_pos: int | None
    assistant_start_pos: int | None


def audit_token_positions(
    input_ids: Sequence[int],
    tokenizer,
    relative_positions: Sequence[int],
    user_end_markers: Sequence[str] = (),
    assistant_start_markers: Sequence[str] = (),
) -> TokenAudit:
    ids = [int(v) for v in input_ids]
    tokens = tuple(str(t) for t in tokenizer.convert_ids_to_tokens(ids))
    seq_len = len(tokens)
    rel_map = {int(pos): _relative_to_absolute(int(pos), seq_len) for pos in relative_positions}
    return TokenAudit(
        sequence_length=seq_len,
        tokens=tokens,
        relative_to_absolute=rel_map,
        user_end_pos=_find_last_marker(tokens, user_end_markers),
        assistant_start_pos=_find_first_marker(tokens, assistant_start_markers),
    )


def _relative_to_absolute(pos: int, seq_len: int) -> int:
    absolute = seq_len + pos if pos < 0 else pos
    if absolute < 0 or absolute >= seq_len:
        raise IndexError(f"Position {pos} is outside sequence length {seq_len}")
    return absolute


def _find_last_marker(tokens: tuple[str, ...], markers: Sequence[str]) -> int | None:
    marker_set = set(markers)
    for idx in range(len(tokens) - 1, -1, -1):
        if tokens[idx] in marker_set:
            return idx
    return None


def _find_first_marker(tokens: tuple[str, ...], markers: Sequence[str]) -> int | None:
    marker_set = set(markers)
    for idx, token in enumerate(tokens):
        if token in marker_set:
            return idx
    return None

