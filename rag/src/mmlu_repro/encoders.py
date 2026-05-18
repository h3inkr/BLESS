from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from .utils import last_token_pool, to_cpu_float_tensor


@dataclass
class EncodeConfig:
    backend: str
    model_name_or_path: str
    peft_adapter_path: Optional[str] = None
    device: str = "cuda"
    # [NEW] device_map 지원: None(단일 GPU), "auto"(멀티 GPU/CPU offload)
    device_map: Optional[str] = None
    torch_dtype: str = "auto"
    max_length: int = 2048
    query_prefix: str = ""
    passage_prefix: str = ""
    append_eos: bool = False
    normalize: bool = False
    query_instruction: str = ""
    doc_instruction: str = ""
    trust_remote_code: bool = True


class BaseEncoder:
    def encode_queries(self, texts: List[str], batch_size: int) -> torch.Tensor:
        raise NotImplementedError

    def encode_docs(self, texts: List[str], batch_size: int) -> torch.Tensor:
        raise NotImplementedError


class ReasonIREncoder(BaseEncoder):
    def __init__(self, cfg: EncodeConfig):
        self.cfg = cfg
        self.model = AutoModel.from_pretrained(
            cfg.model_name_or_path,
            torch_dtype=cfg.torch_dtype,
            trust_remote_code=cfg.trust_remote_code,
        ).to(cfg.device)
        self.model.eval()

    @torch.inference_mode()
    def _encode(self, texts: List[str], instruction: str, batch_size: int) -> torch.Tensor:
        outs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            emb = self.model.encode(batch, instruction=instruction)
            emb = to_cpu_float_tensor(emb)
            if self.cfg.normalize:
                emb = F.normalize(emb, p=2, dim=-1)
            outs.append(emb)
        return torch.cat(outs, dim=0)

    def encode_queries(self, texts: List[str], batch_size: int) -> torch.Tensor:
        return self._encode(texts, self.cfg.query_instruction, batch_size)

    def encode_docs(self, texts: List[str], batch_size: int) -> torch.Tensor:
        return self._encode(texts, self.cfg.doc_instruction, batch_size)


class SentenceTransformerEncoder(BaseEncoder):
    def __init__(self, cfg: EncodeConfig):
        from sentence_transformers import SentenceTransformer
        self.cfg = cfg
        model_kwargs = {}
        if cfg.torch_dtype == "auto":
            model_kwargs["torch_dtype"] = "auto"
        self.model = SentenceTransformer(
            cfg.model_name_or_path,
            trust_remote_code=cfg.trust_remote_code,
            model_kwargs=model_kwargs,
            device=cfg.device,
        )
        try:
            self.model.set_pooling_include_prompt(include_prompt=False)
        except Exception:
            pass

    @torch.inference_mode()
    def _encode(self, texts: List[str], instruction: str, batch_size: int) -> torch.Tensor:
        try:
            emb = self.model.encode(
                texts,
                batch_size=batch_size,
                instruction=instruction,
                normalize_embeddings=self.cfg.normalize,
                convert_to_tensor=True,
                show_progress_bar=False,
            )
        except TypeError:
            # instruction 파라미터 미지원 구버전 fallback
            # [FIX] separator 추가 (기존: instruction + text, 수정: instruction + " " + text)
            if instruction:
                sep = " " if not instruction.endswith((" ", "\n")) else ""
                texts = [instruction + sep + t for t in texts]
            emb = self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=self.cfg.normalize,
                convert_to_tensor=True,
                show_progress_bar=False,
            )
        return to_cpu_float_tensor(emb)

    def encode_queries(self, texts: List[str], batch_size: int) -> torch.Tensor:
        return self._encode(texts, self.cfg.query_instruction, batch_size)

    def encode_docs(self, texts: List[str], batch_size: int) -> torch.Tensor:
        return self._encode(texts, self.cfg.doc_instruction, batch_size)


class HFCausalEOSEncoder(BaseEncoder):
    """
    Generic Tevatron/Qwen-style embedding backend.

    - Adds query/passage prefixes.
    - Optionally appends EOS token.
    - Uses last non-padding token hidden state as embedding.
    - Optionally L2-normalizes embeddings.

    [FIX] device_map 지원 추가: None(단일 GPU), "auto"(멀티 GPU / CPU offload).
    """

    def __init__(self, cfg: EncodeConfig):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name_or_path,
            trust_remote_code=cfg.trust_remote_code,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs: dict = dict(
            torch_dtype=cfg.torch_dtype,
            trust_remote_code=cfg.trust_remote_code,
        )
        # [FIX] device_map 지원: "auto" 지정 시 accelerate가 GPU/CPU 배분
        if cfg.device_map is not None:
            load_kwargs["device_map"] = cfg.device_map
        else:
            load_kwargs["device_map"] = None

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name_or_path,
            **load_kwargs,
        )
        if cfg.peft_adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, cfg.peft_adapter_path)

        # device_map="auto" 사용 시 .to(device) 불필요 (accelerate가 처리)
        if cfg.device_map is None:
            self.model.to(cfg.device)
        self.model.eval()

        # 실제 forward device (device_map="auto"면 첫 레이어 device)
        self._device = next(self.model.parameters()).device

    def _format(self, texts: List[str], prefix: str) -> List[str]:
        out = []
        eos = self.tokenizer.eos_token or ""
        for text in texts:
            s = prefix + text
            if self.cfg.append_eos and eos and not s.endswith(eos):
                s = s + eos
            out.append(s)
        return out

    @torch.inference_mode()
    def _encode(self, texts: List[str], prefix: str, batch_size: int) -> torch.Tensor:
        outs = []
        formatted = self._format(texts, prefix)
        for i in range(0, len(formatted), batch_size):
            batch = formatted[i:i + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.cfg.max_length,
                return_tensors="pt",
            ).to(self._device)

            outputs = self.model.model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                output_hidden_states=False,
                return_dict=True,
            )
            emb = last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
            emb = emb.float()
            if self.cfg.normalize:
                emb = F.normalize(emb, p=2, dim=-1)
            outs.append(emb.detach().cpu())
        return torch.cat(outs, dim=0)

    def encode_queries(self, texts: List[str], batch_size: int) -> torch.Tensor:
        return self._encode(texts, self.cfg.query_prefix, batch_size)

    def encode_docs(self, texts: List[str], batch_size: int) -> torch.Tensor:
        return self._encode(texts, self.cfg.passage_prefix, batch_size)


def build_encoder(cfg: EncodeConfig) -> BaseEncoder:
    backend = cfg.backend.lower()
    if backend == "reasonir":
        return ReasonIREncoder(cfg)
    if backend in {"sentence_transformer", "st"}:
        return SentenceTransformerEncoder(cfg)
    if backend in {"hf_causal_eos", "tevatron", "qwen_eos"}:
        return HFCausalEOSEncoder(cfg)
    raise ValueError(f"Unknown backend: {cfg.backend}")
