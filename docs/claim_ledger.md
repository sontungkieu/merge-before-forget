# SLAO reproduction claim ledger

| ID | Paper claim/setting | Extracted value | Evidence status | Reproduction decision |
|---|---|---:|---|---|
| C01 | SLAO update | Algorithm 1, QR A init; prior fine-tuned B init; time-aware B merge | specified | Implement literally and unit-test each transition. |
| C02 | Time coefficient | `lambda(i)=1/sqrt(i)` | specified | No tuned coefficient in primary run. |
| C03 | LoRA placement | query and value attention projections | specified | PEFT targets `q_proj`, `v_proj`. |
| C04 | LoRA rank | 8 | specified | Rank 8 for paper runs. |
| C05 | Qwen model | Qwen2.5-3B | partial | Pin public `Qwen/Qwen2.5-3B` revision `3aab1f...`; paper gives no repository/revision or Base-vs-Instruct explanation. |
| C06 | SuperNI order O1 | 15-task sequence in Appendix Table 17 | specified | Match exact order and SAPT task names. |
| C07 | SuperNI samples | Paper says 1,000 train and 100 validation/testing per task | primary-source conflict | Pinned SAPT has only 160/20/20 for task1572, 338/43/43 for task181, 142/18/18 for task639, 126/16/16 for task1590, and 975 train for task073. Use every available pinned split row without duplication and classify the table attempt as partial. |
| C08 | SuperNI optimization | LR `5e-5`, 5 epochs, batch 2, grad accumulation 4 | specified for Llama, not explicitly Qwen | Apply to Qwen and label assumption. |
| C09 | LoRA alpha/dropout | not reported | missing | Use alpha 32/dropout 0.1 inherited from official O-LoRA Llama code; sensitivity remains open. |
| C10 | Sequence lengths | not reported | missing | Use SAPT's 1024 source/50 target convention. |
| C11 | Seeds | three random seeds; identities not reported | missing | Pre-registered 42/43/44 are complete for SLAO and SeqLoRA; report their mean and sample SD while retaining the partial label for the undisclosed paper factors. |
| C12 | Qwen SuperNI target | SLAO O1 37.8, O2 32.4, avg 35.1 | specified | Compare AA in percentage points and relative delta. |
| C13 | SeqLoRA target on Qwen | not reported | missing | Run matched baseline; do not claim paper replication for its value. |
| C14 | Llama-3.2-3B smallest Standard-CL cell | O1 74.3 | specified but checkpoint ambiguous/gated | Available HF credential returned HTTP 403 for Base and Instruct on 2026-07-22; do not call substitutes exact. |
| C15 | Standard CL task count | prose lists five datasets including Yelp; Table 17 orders contain four and omit Yelp | conflict | Table-cell reproduction follows published order, and the conflict remains explicit. |
| C16 | BWT | `mean_{i<T}(a[i,T]-a[i,i])` | specified | Implement denominator `T-1`. |
| C17 | MOPD/AOPD | max/mean per-task range across orders | specified | Unit-test; unavailable for one order. |
| C18 | Hardware | A100; appendix says four A100s overall and one cell can run on one A100 | partial | Record exact accelerator and classify non-A100 results environment-sensitive. |

## Result-state vocabulary

`pending` means no terminal verified artifact; `failed` means a terminal run did
not meet its execution contract; `development` is a non-paper gate; `partial`
matches a paper cell with disclosed missing factors; `approximate` changes a
specified factor; and `exact` requires all factors and three-seed aggregation.
