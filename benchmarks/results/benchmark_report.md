# IntelliForge benchmark report

This file records one workstation run of the local mock. It is not a vLLM serving result and it is not a Qwen 2.5 3B measurement.

| Item | Value | Class |
| --- | --- | --- |
| Deployment decision | NO-GO | GATE |
| Real equipment records | 0 | SPEC |
| Machine | Windows 11 AMD64 processor=Intel64 Family 6 Model 140 Stepping 1, GenuineIntel cpu_count=8 python=3.12.10 | SPEC |
| LoRA held-out accuracy | 1.0 | MEASURED NumPy trainer |
| Baseline held-out accuracy | 0.333333 | MEASURED unadapted head |
| LoRA prototype accuracy | 1.0 | MEASURED |
| Baseline prototype accuracy | 0.333333 | MEASURED |
| Train loss start / end | 1.098612 / 0.000206 | MEASURED |
| Quality cases | 4 | MEASURED |
| LoRA-route task completion | 1.0 | MEASURED |
| Baseline-route task completion | 0.25 | MEASURED |
| LoRA-route fault accuracy | 1.0 | MEASURED |
| Baseline-route fault accuracy | 0.25 | MEASURED |
| Local latency iterations | 100 | MEASURED |
| Local latency P95 | 2.667 ms | MEASURED |
| Whitespace tokens per second | 73436.525774 | MEASURED on the mock answers |
| Int8 max absolute error | 0.00665971 | MEASURED on the LoRA matrices |
| Derived local cost, mean USD | 1.56e-05 | DERIVED |
| Local price assumption | 0.00015 USD per 1,000 whitespace tokens, as of 2026-10-21 | ASSUMPTION |
| grok-4.7 list price, prompts under 200k tokens | input 2.0 USD and output 6.0 USD per 1,000,000 tokens | SPEC from the vendor page on 2026-10-21 |
| Qwen / vLLM | not-run | GATE |
| Hugging Face PEFT | not-run | GATE |
| bitsandbytes | not-run | GATE |
| GPU memory | not-run | GATE |
| Neo4j | not-run | GATE |
| pgvector | not-run | GATE |
| Hosted commercial call | not-run | GATE |
| MLflow client | not-run | GATE |

The LoRA numbers come from `training/lora_train.py`. The task scores come from the synthetic cases under `datasets/synthetic_equipment/`. Latency is the in-process investigation, including retrieval, the local head, and the SciPy tool.
