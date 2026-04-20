"""
Unit test for T4: verify common/ modules import and basic functions work.
Run: python experiments/repro_arditi_wollschlager/test_t4_common.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.model_paths import MODEL_PATHS, JUDGE_PATHS
assert "qwen2.5_7b" in MODEL_PATHS, "Missing qwen2.5_7b"
assert "llama3.1_8b" in MODEL_PATHS, "Missing llama3.1_8b"
assert "llamaguard3" in JUDGE_PATHS, "Missing llamaguard3"
print("[PASS] model_paths: keys present")

from common.eval_judges import judge_keyword, compute_asr_keyword

jailbreak_resp = "Sure, here is how to make it: Step 1..."
refusal_ascii  = "I'm sorry, I cannot help with that."
refusal_curly  = "I\u2019m sorry, I can\u2019t assist."

assert judge_keyword(jailbreak_resp) is True,  "Jailbreak should bypass keyword"
assert judge_keyword(refusal_ascii)  is False, "ASCII refusal should be caught"
assert judge_keyword(refusal_curly)  is False, "Curly-quote refusal should be caught"
print("[PASS] eval_judges.judge_keyword: all cases correct")

completions = [
    {"response": jailbreak_resp},
    {"response": refusal_ascii},
    {"response": refusal_curly},
]
asr = compute_asr_keyword(completions)
assert abs(asr - 1/3) < 0.01, f"Expected ASR ~0.333, got {asr}"
print(f"[PASS] eval_judges.compute_asr_keyword: {asr:.3f} (expected ~0.333)")

from common.stealth_analysis import compute_srr, compute_concordance

srr = compute_srr(asr_kw=0.8, asr_lg3=0.6)
assert abs(srr - 0.2) < 1e-9, f"SRR should be 0.2, got {srr}"
print(f"[PASS] stealth_analysis.compute_srr: {srr:.3f}")

conc = compute_concordance([True, False, True], [True, True, True])
assert abs(conc - 2/3) < 0.01, f"Concordance should be 0.667, got {conc}"
print(f"[PASS] stealth_analysis.compute_concordance: {conc:.3f}")

print("\nAll T4 tests passed.")
