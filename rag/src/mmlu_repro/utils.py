import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def maybe_json_loads(x: Any) -> Any:
    if isinstance(x, str):
        return json.loads(x)
    return x


def get_query(ex: Dict[str, Any]) -> str:
    # [FIX] raw_query / query 둘 다 없거나 빈 문자열이면 명시적 에러
    result = ex.get("raw_query") or ex.get("query")
    if not result:
        raise ValueError(
            f"Example has no usable query field. Keys present: {list(ex.keys())}"
        )
    return result


def get_ctxs(ex: Dict[str, Any]) -> List[Dict[str, Any]]:
    ctxs = maybe_json_loads(ex["ctxs"])
    return [c for c in ctxs if isinstance(c, dict) and c.get("retrieval text")]


def write_jsonl_line(f, obj: Dict[str, Any]) -> None:
    f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def simple_tokenize(text: str) -> List[str]:
    return text.lower().replace("\n", " ").split()


def to_cpu_float_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x.detach()
    elif isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    elif isinstance(x, list):
        t = torch.tensor(x)
    else:
        raise TypeError(f"Unsupported embedding type: {type(x)}")
    t = t.float().cpu()
    if t.ndim == 1:
        t = t.unsqueeze(0)
    return t


def last_token_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = hidden_states.shape[0]
    return hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]


class NumpyDiskCache:
    """Lightweight embedding cache using one .npy file per key.

    [FIX] float32로 저장 (기존 float16은 dot-product 수치 오차 유발).
    """

    def __init__(self, cache_dir: Optional[str]):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        assert self.cache_dir is not None
        return self.cache_dir / f"{key}.npy"

    def enabled(self) -> bool:
        return self.cache_dir is not None

    def get(self, key: str) -> Optional[torch.Tensor]:
        if not self.enabled():
            return None
        path = self._path(key)
        if not path.exists():
            return None
        arr = np.load(path)
        return torch.from_numpy(arr).float()

    def set(self, key: str, value: torch.Tensor) -> None:
        if not self.enabled():
            return
        path = self._path(key)
        if path.exists():
            return
        # [FIX] float32로 저장 (float16 → dot-product 수치 오차 제거)
        arr = value.detach().cpu().numpy().astype(np.float32)
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, arr)
        os.replace(tmp, path)


# ---------------------------------------------------------------------------
# BM25 tokenization in-process cache
# ---------------------------------------------------------------------------

_TOKENIZE_CACHE: Dict[str, List[str]] = {}


def simple_tokenize_cached(text: str) -> List[str]:
    """simple_tokenize with in-process dict cache.

    MMLU candidate pool에는 쿼리 간 중복 passage가 많아
    동일 텍스트에 대한 반복 tokenization을 피하기 위해 캐싱.
    메모리 상한 200k 엔트리 초과 시 절반 드롭.
    """
    tokens = _TOKENIZE_CACHE.get(text)
    if tokens is None:
        tokens = simple_tokenize(text)
        if len(_TOKENIZE_CACHE) >= 200_000:
            to_drop = list(_TOKENIZE_CACHE.keys())[:100_000]
            for k in to_drop:
                del _TOKENIZE_CACHE[k]
        _TOKENIZE_CACHE[text] = tokens
    return tokens
