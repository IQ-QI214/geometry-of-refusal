# A1 Cross-Validation Analysis: Qwen3Guard vs LLaMA-Guard

> Cross-validation to rule out Qwen3Guard in-family bias on Qwen-generated responses.


## Concordance Table

| Model | Type | Role | Config | n_disputed | concordance_rate | discord_rate | adjusted_FHCR |
|-------|------|------|--------|-----------|-----------------|--------------|---------------|
| qwen2vl_7b | II | Primary | ablation_nw_vmm | 311 | 0.868 | 0.132 | 0.185 |
| qwen2vl_7b | II | Primary | ablation_all_vmm | 476 | 0.887 | 0.113 | 0.227 |
| qwen2vl_7b | II | Primary | ablation_nw_vtext | 308 | 0.886 | 0.114 | 0.170 |
| qwen2vl_32b | II | Primary | ablation_nw_vmm | 483 | 0.886 | 0.114 | 0.194 |
| qwen2vl_32b | II | Primary | ablation_all_vmm | 524 | 0.905 | 0.095 | 0.170 |
| qwen2vl_32b | II | Primary | ablation_nw_vtext | 503 | 0.926 | 0.074 | 0.138 |
| llava_7b | I | Negative control | ablation_nw_vmm | 79 | 0.810 | 0.190 | 0.771 |
| llava_7b | I | Negative control | ablation_all_vmm | 103 | 0.767 | 0.233 | 0.661 |
| llava_7b | I | Negative control | ablation_nw_vtext | 71 | 0.761 | 0.239 | 0.755 |
| llava_13b | I | Negative control | ablation_nw_vmm | 91 | 0.791 | 0.209 | 0.788 |
| llava_13b | I | Negative control | ablation_all_vmm | 128 | 0.836 | 0.164 | 0.708 |
| llava_13b | I | Negative control | ablation_nw_vtext | 130 | 0.831 | 0.169 | 0.699 |

## Interpretation

- qwen2vl_7b (ablation_all_vmm): **Partial bias possible** (concordance=88.7%): Stealth Refusal is majority phenomenon but Qwen3Guard may be slightly more lenient.
- qwen2vl_32b (ablation_all_vmm): **Stealth Refusal CONFIRMED** (concordance=90.5%): LLaMA-Guard agrees with Qwen3Guard on >90% of disputed cases. No significant in-family bias.

- llava_7b (ablation_all_vmm): **Negative control NOTE** (concordance=76.7%): Unexpected LLaMA-Guard / Qwen3Guard disagreement on LLaVA responses.
- llava_13b (ablation_all_vmm): **Negative control NOTE** (concordance=83.6%): Unexpected LLaMA-Guard / Qwen3Guard disagreement on LLaVA responses.

Cross-Validation 分析总结


  1. 核心结论：Qwen 的 Stealth Refusal 被第二个 judge 确认                                                         
                  
  Qwen-32B (all_vmm)：concordance = 90.5%，LLaMA-Guard 与 Qwen3Guard 高度一致。隐性拒绝是真实现象，非 judge 偏见。 
                                                                                                                   
  Qwen-7B (all_vmm)：concordance = 88.7%，接近 90% 门槛，同样可信。                                                
                                                                                                                   
  2. 意外发现：LLaMA-Guard 比 Qwen3Guard 更严格                                                                    
                                                                                                                   
  这是交叉验证最有价值的新信息：                                                                                   
                                                                                                                   
  ┌───────────┬──────────┬──────────┬─────────────┬────────┐                                                       
  │   模型    │ 最优攻击 │ FHCR_q3g │ FHCR_lg_adj │  差值  │                                                       
  ├───────────┼──────────┼──────────┼─────────────┼────────┤                                                       
  │ LLaVA-7B  │ nw_vmm   │ 74.5%    │ 77.1%       │ +2.6pp │                                                       
  ├───────────┼──────────┼──────────┼─────────────┼────────┤                                                       
  │ LLaVA-13B │ nw_vmm   │ 75.5%    │ 78.8%       │ +3.3pp │                                                       
  ├───────────┼──────────┼──────────┼─────────────┼────────┤                                                       
  │ Qwen-7B   │ all_vmm  │ 13.3%    │ 22.7%       │ +9.4pp │                                                       
  ├───────────┼──────────┼──────────┼─────────────┼────────┤                                                       
  │ Qwen-32B  │ all_vmm  │ 8.2%     │ 17.0%       │ +8.7pp │                                                       
  └───────────┴──────────┴──────────┴─────────────┴────────┘                                                       
                                                                                                                   
  两个关键模式：                                                                                                   
                                                                                                                   
  a) LLaVA 上差异小（+2~3pp）：两个 judge 基本一致，LLaVA 的响应是"明确有害"或"明确安全"，边界模糊的少。           
                                                                                                                   
  b) Qwen 上差异大（+8~9pp）：LLaMA-Guard 把部分 Qwen3Guard 判 safe 的隐性拒绝改判为 unsafe。这说明 Qwen 的 stealth
   refusal 中有一部分处于灰色地带——伦理反驳中夹杂了足够多的有害信息片段，LLaMA-Guard 认为是 unsafe。
                                                                                                                   
  3. 对核心叙事的影响

  即使用更严格的 LLaMA-Guard 修正后：                                                                              
   
  ┌──────────────────────┬──────────┬───────────┬──────────────────────────┐                                       
  │                      │ Q3G 原值 │ LG 修正后 │        结论变化？        │
  ├──────────────────────┼──────────┼───────────┼──────────────────────────┤
  │ LLaVA-7B 危险        │ 74.5%    │ 77.1%     │ 不变，更危险             │
  ├──────────────────────┼──────────┼───────────┼──────────────────────────┤
  │ Qwen-7B 相对安全     │ 13.3%    │ 22.7%     │ 仍然安全，但差距缩小     │
  ├──────────────────────┼──────────┼───────────┼──────────────────────────┤                                       
  │ Type I vs II 差距    │ 5.6×     │ 3.4×      │ 差距缩小但仍显著         │
  ├──────────────────────┼──────────┼───────────┼──────────────────────────┤                                       
  │ Stealth refusal 存在 │ 83.2%    │ 73.8%     │ 缩小约10pp，仍为主要现象 │
  └──────────────────────┴──────────┴───────────┴──────────────────────────┘                                       
   
  核心叙事依然成立：Type I 对几何攻击脆弱（>75%），Type II 有冗余防线（<23%）。但定量上需要报告双 judge            
  范围而非单一数值。

  4. LLaVA 的 concordance 偏低（76~83%）值得关注                                                                   
   
  LLaVA disputed cases 只有 71~130 个（vs Qwen 的 300~524），但 discord rate 较高（16~24%）。Discord 方向是 LG 判  
  unsafe、Q3G 判 safe。这意味着：
                                                                                                                   
  - Qwen3Guard 对 LLaVA 的某些输出"过于宽容"
  - 或者 LLaMA-Guard 对这类输出"过于严格"
  - 需要人工抽样检查 discordant cases 来判断谁更准确
                                                                                                                   
  5. 论文中的推荐报告方式
                                                                                                                   
  建议用 FHCR 范围报告：

  ▎ LLaVA-7B (nw_vmm): FHCR = 74.5%–77.1% (Qwen3Guard / LLaMA-Guard)
  ▎ Qwen-32B (all_vmm): FHCR = 8.2%–17.0%
                                          