"""
计算 SRR 附带指标并写 summary.md。
Run: python experiments/repro_arditi_wollschlager/compute_summary.py
"""
import sys, os, json

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from common.stealth_analysis import compute_srr

EVAL_JSON  = "results/repro_arditi_wollschlager/evaluation.json"
SUMMARY_MD = "results/repro_arditi_wollschlager/summary.md"


def fmt(v):
    if v is None or v == -1.0:
        return "  N/A  "
    return f"{v*100:5.1f}%"


def fmt_srr(v):
    if v is None:
        return "  N/A  "
    return f"{v*100:+5.1f}pp"


def main():
    if not os.path.exists(EVAL_JSON):
        print(f"[ERROR] {EVAL_JSON} not found. Run run_evaluate.py first.")
        raise SystemExit(1)

    data = json.load(open(EVAL_JSON))
    lines = ["# ASR Summary — Repro Arditi-Wollschläger", "",
             f"Generated from: {EVAL_JSON}", ""]

    for model_key, configs in data.items():
        lines.append(f"## {model_key}")
        lines.append("")
        header = f"{'Config':<20} | {'ASR_kw':>7} | {'ASR_LG3':>7} | {'ASR_SR':>7} | {'SRR':>8} | n"
        lines.append(header)
        lines.append("-" * len(header))
        best_srr = None
        best_config = None
        for config, m in configs.items():
            asr_kw  = m.get("asr_kw")
            asr_lg3 = m.get("asr_lg3")
            asr_sr  = m.get("asr_sr")
            srr = compute_srr(asr_kw, asr_lg3) if (asr_kw is not None and asr_lg3 is not None and asr_lg3 >= 0) else None
            n = m.get("n", "?")
            row = f"{config:<20} | {fmt(asr_kw)} | {fmt(asr_lg3)} | {fmt(asr_sr)} | {fmt_srr(srr)} | {n}"
            lines.append(row)
            if srr is not None and (best_srr is None or srr > best_srr):
                best_srr = srr
                best_config = config
        lines.append("")
        if best_srr is not None:
            lines.append(f"**Best SRR: {best_srr*100:+.1f}pp @ {best_config}**")
        lines.append("")

    lines += ["---", "",
              "## 附带观测：LLM 上的 SRR", "",
              "注：VLM 也有内生对齐训练，本节仅作对照记录，不作 narrative 分叉依据。", ""]
    for model_key, configs in data.items():
        srrs = {}
        for config, m in configs.items():
            asr_kw  = m.get("asr_kw")
            asr_lg3 = m.get("asr_lg3")
            if asr_kw is not None and asr_lg3 is not None and asr_lg3 >= 0:
                srrs[config] = compute_srr(asr_kw, asr_lg3)
        if srrs:
            best_c = max(srrs, key=lambda c: srrs[c])
            lines.append(f"- **{model_key}**: best SRR = {srrs[best_c]*100:+.1f}pp @ `{best_c}`")
    lines.append("")

    os.makedirs(os.path.dirname(SUMMARY_MD), exist_ok=True)
    with open(SUMMARY_MD, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Written: {SUMMARY_MD}")
    for line in lines[:30]:
        print(line)


if __name__ == "__main__":
    main()
