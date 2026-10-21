# Benchmark methodology

The committed numbers come from one workstation run. Regenerate them with the commands below. Do not treat the tables in a slide as a substitute for `benchmarks/results/`.

## What is measured

| Measurement | How |
| --- | --- |
| LoRA quality | `training/lora_train.py` trains rank 2, alpha 4, seed 7, for 400 full-batch steps. The baseline head is zeros, so argmax is class `normal`. Held-out data uses seed 8 |
| Prototype accuracy | Three clean encoder vectors, one per class |
| Task completion | Four qualification cases. A pass needs the expected fault, the expected evidence id, the expected equipment ids, a completed status, and the sentence that the setpoint was not applied |
| Fault accuracy | Predicted supported fault equals the case label. A stopped run has a null fault and counts as a miss |
| Latency P95 | 100 diagnose calls on `SYN-CASE-001` through the quality route. Nearest-rank P95. The clock includes retrieval, the head, and SciPy |
| Tokens per second | Whitespace tokens in the prompt and answer, divided by the wall clock of those 100 calls. This is not a model tokenizer and not a vLLM throughput number |
| Int8 error | Max absolute error after quantizing and restoring the LoRA matrices |
| Cost | Whitespace tokens times `0.00015` USD per 1,000 tokens. The price is an assumption dated 2026-10-21. The product is derived |

The sensor encoder in `featurize` is synthetic. It places a 1 on the axis for `normal`, `cooling_flow_low`, or `pressure_leak`. The LoRA head learns that map. It is not a fine-tune of Qwen 2.5 3B.

## What stays not-run

Qwen weight loading, vLLM, Hugging Face PEFT, bitsandbytes, GPU memory, Neo4j, pgvector, the MLflow client, and any HTTP call to `https://api.x.ai/v1`. The grok-4.7 input and output prices in the report are the vendor list price for prompts under 200k tokens, copied from `https://docs.x.ai/docs/models` on 2026-10-21. `calls` is 0.

## Machine line

`platform.system()`, `platform.release()`, `platform.machine()`, `platform.processor()`, `os.cpu_count()`, and `platform.python_version()`. The host name is omitted.

## How to repeat

```bash
python -m pip install -e ".[dev]"
python benchmarks/run_quality_eval.py
python benchmarks/run_latency_eval.py
python -m pytest
```

`run_latency_eval.py` also refreshes the quality file, `benchmark_report.json`, `benchmark_report.md`, and `tracking_run.json`. The tests check that the markdown equals `render_markdown` of the JSON, that the LoRA route completed every qualification case, and that the baseline route did not.
