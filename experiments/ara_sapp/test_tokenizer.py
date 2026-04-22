import json, shutil
from pathlib import Path
from transformers import AutoTokenizer

src = "/inspire/hdd/global_user/wenming-253108090054/models/gemma-4-heretic"
dst = Path("/inspire/hdd/global_user/wenming-253108090054/zhujiaqi/geometry-of-refusal/results/ara_sapp/_tok_tmp")
dst.mkdir(parents=True, exist_ok=True)

for f in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "config.json"]:
    p = Path(src) / f
    if p.exists():
        shutil.copy2(p, dst / f)

cfg = json.load(open(dst / "tokenizer_config.json"))
if isinstance(cfg.get("extra_special_tokens"), list):
    cfg["extra_special_tokens"] = {}
json.dump(cfg, open(dst / "tokenizer_config.json", "w"))

tok = AutoTokenizer.from_pretrained(str(dst), local_files_only=True)
out = tok.apply_chat_template(
    [{"role": "user", "content": "hello"}],
    add_generation_prompt=True,
    tokenize=False,
)
print("OK:", repr(out[:150]))
