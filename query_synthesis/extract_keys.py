import torch
from vllm import LLM, SamplingParams
import argparse
import json
from tqdm import tqdm
from datasets import load_dataset


# ─────────────────────────────────────────────
# Prompt templates
# ─────────────────────────────────────────────

SYSTEM_PROMPT = {
    "math": (
        "You are a keyphrase extraction engine.\n"
        "Do not answer questions. Do not explain.\n"
        "Your only job is to output 1 to 3 key phrases capturing the document's core topic.\n"
        "\n"
        "Hard rules:\n"
        "- Output EXACTLY one line: \"Document Key Topics\": <comma-separated key phrases>\n"
        "- 1 to 3 phrases only.\n"
        "- Each phrase should be a specific retrieval-friendly noun phrase.\n"
        "- Prefer theorem/definition names, formula names, named concepts, or distinctive expressions.\n"
        "- If a crucial formula appears, you may include it in compact form (e.g., \"1/2 ab sin C\", \"s = rθ\").\n"
        "- IGNORE metadata/noise: Tags, Category lines, ids, timestamps, wiki markup, file names.\n"
        "- AVOID overly generic words like: mathematics, geometry, Euclidean geometry, triangles, circles, definition, theorem, proof.\n"
        "- Do NOT output those generic words unless they are part of a specific named concept.\n"
    ),
    "bright": (
        "You are a helpful assistant. "
        "Do not answer any question. Do not add explanations. "
        "Only extract the document's core topics and key phrases."
    ),
}

USER_PROMPT_TEMPLATE = {
    "math": (
        "Task: Extract the document's core topic key phrases.\n\n"
        "Document:\n{document}\n\n"
        "Instructions:\n"
        "- Return 1 to 3 key phrases for retrieval.\n"
        "- Make them as specific as possible (prefer formula/concept names over broad categories).\n"
        "- If the document is a definition, include the defined concept and its defining setting (e.g., \"cosine in third quadrant\").\n"
        "- Output EXACTLY one line in the format:\n"
        "\"Document Key Topics\": <comma-separated key phrases>"
    ),
    "bright": (
        "Task: Extract the document's core topics.\n\n"
        "Document:\n{document}\n\n"
        "Instructions:\n"
        "- Identify the main subject(s) and central themes of the document.\n"
        "- Extract 1 to 3 key phrases that best represent the core topics.\n"
        "- Prefer proper nouns (people, organizations, places, titles) when relevant.\n"
        "- If the document contains multiple sections, reflect the overall topics.\n"
        "- Do NOT summarize the document.\n"
        "- Output must be EXACTLY one line in the following format:\n"
        "\"Document Key Topics\": <comma-separated key phrases>"
    ),
}

SAMPLING_PARAMS = {
    "math": SamplingParams(
        temperature=0.2,
        repetition_penalty=1.1,
        max_tokens=80,
        stop=["\n"],
    ),
    "bright": SamplingParams(
        temperature=0.2,
        repetition_penalty=1.1,
        max_tokens=150,
    ),
}


# ─────────────────────────────────────────────
# Core functions
# ─────────────────────────────────────────────

def extract_keywords(document: str, llm: LLM, source: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT[source]},
        {"role": "user",   "content": USER_PROMPT_TEMPLATE[source].format(document=document)},
    ]

    prompt = llm.get_tokenizer().apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    output = llm.generate(prompts=[prompt], sampling_params=SAMPLING_PARAMS[source])
    return output[0].outputs[0].text.replace("\n", " ").strip()


def get_query_and_pos(example: dict, source: str) -> tuple[str, str]:
    """source에 따라 (query, positive passage text) 쌍을 반환."""
    if source == "math":
        query = example["query"]
        pos = _get_first_positive_text_math(example)
    else:  # bright
        query = example["query"][1]
        pos = example["pos"][0][1]
    return query, pos


def _get_first_positive_text_math(example: dict) -> str:
    for k in ["positive_passages", "positive passages", "positives", "positive"]:
        if k in example:
            pos_obj = example[k]
            break
    else:
        raise KeyError(
            f"Cannot find positive passages key. Available keys: {list(example.keys())}"
        )

    if isinstance(pos_obj, list):
        if len(pos_obj) == 0:
            raise ValueError("positive passages list is empty")
        pos0 = pos_obj[0]
    else:
        pos0 = pos_obj

    if isinstance(pos0, dict):
        if "text" not in pos0:
            raise KeyError(f"positive passage dict has no 'text' field. keys={list(pos0.keys())}")
        return pos0["text"]
    if isinstance(pos0, (list, tuple)) and len(pos0) >= 2:
        return pos0[1]
    if isinstance(pos0, str):
        return pos0

    raise TypeError(f"Unsupported positive passage type: {type(pos0)}")


def load_data(args):
    """source에 따라 데이터 이터레이터를 반환."""
    if args.source == "math":
        ds = load_dataset(
            "Raderspace/MATH_NuminaMath_allquerytypes",
            split=args.split,
            streaming=args.streaming,
        )
        return ds
    else:  # bright
        with open(args.data_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]


# ─────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────

def parse_argument():
    parser = argparse.ArgumentParser(
        description="Unified keyword extraction for Math (HF RaDeR dataset) and Bright (local JSONL)."
    )
    parser.add_argument(
        "--source", type=str, required=True, choices=["math", "bright"],
        help="Data source: 'math' for HuggingFace dataset, 'bright' for local JSONL file.",
    )
    parser.add_argument("--output_path", "-op", type=str, required=True)

    # math-only options
    parser.add_argument("--split",       type=str,  default="train")
    parser.add_argument("--max_samples", type=int,  default=None,  help="(math) 앞에서 N개만 처리")
    parser.add_argument("--streaming",   action="store_true",      help="(math) 대용량 스트리밍 모드")

    # bright-only options
    parser.add_argument("--data_path", "-dp", type=str, default=None, help="(bright) 로컬 JSONL 경로")

    return parser.parse_args()


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_argument()

    if args.source == "bright" and not args.data_path:
        raise ValueError("--data_path is required when --source=bright")

    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        tensor_parallel_size=1,
        max_model_len=4096,
        dtype="half",
    )

    data = load_data(args)

    with open(args.output_path, "w", encoding="utf-8") as save_file:
        for i, example in enumerate(tqdm(data, desc="Analyzing...")):
            if args.source == "math" and args.max_samples is not None and i >= args.max_samples:
                break

            query, pos = get_query_and_pos(example, args.source)
            keywords   = extract_keywords(pos, llm, args.source)

            json.dump({"query": query, "pos": pos, "keywords": keywords},
                      save_file, ensure_ascii=False)
            save_file.write("\n")