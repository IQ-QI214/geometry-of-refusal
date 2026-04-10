Exp 3C 完整结果分析                                                                                           
                                                                                           
  先整理数据总表（full_harmful_completion_rate）：                                                              
                                                                                                                
  ┌──────────────┬────────────────────┬───────────────┬─────────────┬────────┬─────────┬──────────┐             
  │     模型     │       Group        │ baseline_text │ baseline_mm │ nw_vmm │ all_vmm │ nw_vtext │             
  ├──────────────┼────────────────────┼───────────────┼─────────────┼────────┼─────────┼──────────┤             
  │ LLaVA        │ A (CLIP+reversal)  │ 0.125         │ 0.625       │ 0.875  │ 0.875   │ 0.875    │             
  ├──────────────┼────────────────────┼───────────────┼─────────────┼────────┼─────────┼──────────┤             
  │ Qwen2.5-VL   │ B (Custom+uniform) │ 0.000         │ 0.000       │ 0.625  │ 1.000   │ 0.625    │             
  ├──────────────┼────────────────────┼───────────────┼─────────────┼────────┼─────────┼──────────┤             
  │ InternVL2    │ B (Custom+uniform) │ 0.000         │ 0.125       │ 0.125  │ 0.250   │ 0.125    │             
  ├──────────────┼────────────────────┼───────────────┼─────────────┼────────┼─────────┼──────────┤             
  │ InstructBLIP │ A (CLIP+reversal)  │ 0.750         │ 1.000       │ 1.000  │ 1.000   │ 1.000    │             
  └──────────────┴────────────────────┴───────────────┴─────────────┴────────┴─────────┴──────────┘             
                                                                                                                
  ---                                                                                                           
  Insight 1（最核心）：Amplitude Reversal 行为预测最优 Ablation 策略                                            
                                                                                                                
  这是最有价值的新发现，将 3A 和 3C 的结论打通：            
                                                                                                                
  ┌─────────────────────────────┬────────────────────────────┬───────────────┬────────────────────────────┐     
  │            Group            │       Amplitude 行为       │    NW vs      │            含义            │     
  │                             │                            │   All-layer   │                            │     
  ├─────────────────────────────┼────────────────────────────┼───────────────┼────────────────────────────┤ 
  │ Group A (LLaVA,             │ 浅层压制 + 深层放大 →      │ NW ≥          │ Refusal 集中在特定瓶颈层   │ 
  │ InstructBLIP)               │ reversal                   │ all-layer     │                            │     
  ├─────────────────────────────┼────────────────────────────┼───────────────┼────────────────────────────┤ 
  │ Group B (Qwen2.5-VL,        │ 全层均匀压制 → no reversal │ all-layer >   │ Refusal                    │     
  │ InternVL2)                  │                            │ NW            │ 分布在所有层，无瓶颈       │     
  └─────────────────────────────┴────────────────────────────┴───────────────┴────────────────────────────┘
                                                                                                                
  Qwen2.5-VL 数据最强：NW=62.5% vs all=100%，差距 37.5pp。                                                      
  
  Paper claim 潜力："The presence or absence of amplitude reversal predicts the optimal ablation strategy —     
  CLIP-based architectures exhibit a narrow waist bottleneck enabling single-layer attacks, while custom ViT
  architectures require distributed ablation across all layers."                                                
                                                            
  这将 3A 和 3C 结合成一个统一的 mechanistic 框架，而不是两个独立的发现。                                       
  
  ---                                                                                                           
  Insight 2：Qwen2.5-VL — 最强 Baseline Safety，但最完全的对齐崩溃
                                                                                                                
  Qwen2.5-VL 是最有趣的：
  - baseline_text = 0%，baseline_mm = 0%（8/8 全部拒绝，是唯一 baseline 完全拒绝的模型）                        
  - all-layer ablation = 100%（8/8 全部 FULL_HARMFUL，0 self-correction）                                       
                                                                                                                
  这说明 Qwen2.5-VL 的 safety 机制完全依赖于 refusal direction 的完整性，没有冗余备份。一旦方向被全局消除，安全 
  机制完全失效且没有自我恢复能力（self_corr=0）。相比之下，InternVL2 在全层 ablation 后仍有 75% 的抵抗力。      
                                                                                                                
  Problem/Question：高 safety baseline 为什么对应低冗余？这是 alignment 训练的一个潜在                          
  trade-off——越精准的对齐越脆弱？                           
                                                                                                                
  ---                                                       
  Insight 3：InstructBLIP — Blank Image 已足够，Ablation 无增量效果
                                                                                                                
  InstructBLIP baseline_mm = 100%（不需要任何 ablation，只需要 blank image），而 baseline_text = 75%。这意味着
  Q-Former 的 32-token 信息瓶颈本身就构成了对 safety 机制的完全破坏，而非只是削弱。                             
                                                            
  这与 3A 的发现形成因果链：cos(v_text, v_mm) = 0.467（最低，Q-Former 导致方向偏移大）→ visual modality         
  彻底改变了 hidden state 分布 → safety 机制失效。          
                                                                                                                
  Paper claim 潜力："Q-Former architectures are trivially bypassed by visual input alone, prior to any          
  adversarial manipulation — the bottleneck that was designed for efficiency becomes a security vulnerability."
                                                                                                                
  ---                                                       
  Insight 4：v_text ≈ v_mm 对于 Ablation 效果相同（所有模型）
                                                                                                                
  ┌──────────────┬───────────────────┬────────┬──────────┬─────────────────┐
  │     模型     │ cos(v_text, v_mm) │ nw_vmm │ nw_vtext │      差距       │                                    
  ├──────────────┼───────────────────┼────────┼──────────┼─────────────────┤                                    
  │ LLaVA        │ 0.917             │ 0.875  │ 0.875    │ 0               │                                    
  ├──────────────┼───────────────────┼────────┼──────────┼─────────────────┤                                    
  │ Qwen2.5-VL   │ 0.961             │ 0.625  │ 0.625    │ 0               │                                    
  ├──────────────┼───────────────────┼────────┼──────────┼─────────────────┤                                    
  │ InternVL2    │ 0.919             │ 0.125  │ 0.125    │ 0               │                                    
  ├──────────────┼───────────────────┼────────┼──────────┼─────────────────┤                                    
  │ InstructBLIP │ 0.467             │ 1.000  │ 1.000    │ 0（at ceiling） │
  └──────────────┴───────────────────┴────────┴──────────┴─────────────────┘                                    
                                                            
  用 text-only 数据提取的方向，在 MM                                                                            
  攻击场景下效果完全相同。实践意义：攻击者只需要纯文本数据（更容易获取），不需要构建 harmful 的 image-text
  配对数据集来计算攻击方向。                                                                                    
                                                            
  ---
  Insight 5（Problem）：InternVL2 对 Ablation 的强抗性无法解释
                                                                                                                
  InternVL2 all-layer ablation 仅 25%，远低于其他模型。可能原因：
  1. 深层 NW（88% 深度）的特殊性：ablation 在接近输出层时可能已经来不及抑制中间层积累的 refusal signal          
  2. InternLM2 的 alignment 训练具有冗余机制：safety 不完全依赖单一方向                                         
  3. Mean-diff direction 在 InternVL2 上可能不是最优的 refusal direction 表示：3A 中 v_text 和 v_mm 的 cos      
  很高（0.919），但方向本身可能噪声较大                                                                         
                                                                                                                
  这是一个需要深入研究的 open problem：为什么具有相似 cos 值（0.919）的 InternVL2 和 LLaVA（0.917）在 ablation  
  效果上差距如此之大？                                                                                          
                                                            
  ---                                                                                                           
  对 Paper Narrative 的建议修正                             
                                                                                                                
  原来预期的 narrative（"narrow waist 是通用的"）已经被数据否定。新的、更有力的 narrative：
                                                                                                                
  ▎ "Visual encoder 架构类型通过 amplitude reversal 机制决定 refusal 的空间分布——CLIP                           
  系架构产生层级瓶颈（可被单层精准攻击），自研 ViT 架构产生均匀分布（需要全层攻击）。这一发现揭示了 visual      
  encoder 选择对 VLM safety robustness 的根本性影响，并为针对性攻击策略的设计提供了机制指导。"                  
                                                            
  这比原来的"narrow waist 通用性"要更有 novelty，因为它提出了一个从架构特性预测安全脆弱性的框架。