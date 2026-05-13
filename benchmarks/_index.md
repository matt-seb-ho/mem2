# Benchmarks Index

| Benchmark | Wired? | Conditions Runnable | Last Run | Notes |
|---|---|---|---|---|
| arc_agi | yes | ARC smoke plus current memory axes | see outputs/ and case_studies/ | Active benchmark for all current ports. |
| aime | no | none | none | Data exists at `data/aime_1983_2025/problems.jsonl`; needs math task/eval adapter wiring. |
| livecodebench | partial data only | none | none | Data and concept memory artifacts exist at `data/livecodebench_v56/`; needs active benchmark config. |
| gpqa | no | none | none | Data exists at `data/gpqa_diamond/gpqa_diamond.csv`; needs multiple-choice scoring adapter. |
| math | partial data only | none | none | Competition math data exists at `data/competition_math_*`; needs unified math benchmark adapter. |
| bfcl | no | none | none | BFCL v4 data exists at `data/bfcl_v4/`; needs function-call evaluator. |
| omni_math | no | none | none | Data exists at `data/omni_math/problems.jsonl`; needs math scoring adapter. |
| episodic_streams | planned | none | none | Synthetic or derived sequential task stream for memory growth studies. |
| continual_learning | planned | none | none | Future forgetting and transfer harness. |
| streaming_online | planned | none | none | Future online, time-budgeted benchmark mode. |
