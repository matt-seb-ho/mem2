# Case-Study Modes for LM Inference Research

Case studies in `mem2` should be more than run summaries. A run trace tells us what happened; mode-specific reports ask different questions of the same trace so that a reader can see failures, comparisons, counterfactual openings, provenance, and stochastic behavior from separate angles.

The modes below are scaffolds for offline inspection. They do not call a model. If a later mode needs an LLM judge or a regenerated completion, that should be an explicit downstream step that reads these reports and writes a new derived artifact.

## Literature Anchors

- Behavioral and perturbation testing: Ribeiro et al., "Beyond Accuracy: Behavioral Testing of NLP Models with CheckList", ACL 2020. https://aclanthology.org/2020.acl-main.442/
- Contrast sets and counterfactual data edits: Gardner et al., "Evaluating Models' Local Decision Boundaries via Contrast Sets", Findings of EMNLP 2020. https://aclanthology.org/2020.findings-emnlp.117/
- Retrieval-augmented generation as a setting where retrieved evidence affects outputs: Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", NeurIPS 2020. https://arxiv.org/abs/2005.11401
- RAG evaluation dimensions such as faithfulness and context use: Es et al., "RAGAS: Automated Evaluation of Retrieval Augmented Generation", 2023. https://arxiv.org/abs/2309.15217
- Mechanistic causal tracing and model internals: Meng et al., "Locating and Editing Factual Associations in GPT", NeurIPS 2022. https://arxiv.org/abs/2202.05262
- Self-consistency and repeated-sample variability: Wang et al., "Self-Consistency Improves Chain of Thought Reasoning in Language Models", 2022. https://arxiv.org/abs/2203.11171
- Holistic model evaluation framing: Liang et al., "Holistic Evaluation of Language Models", 2022. https://arxiv.org/abs/2211.09110

## Mode Index

| Mode | What It Surfaces | When To Use | Inputs Needed | Script |
|---|---|---|---|---|
| Error analysis | Failed tasks, incorrect parsed outputs, missing retrieval context, and recurring failure labels. | First pass after every case-study run. | Single run. | `case_studies/scripts/modes/error_analysis.py` |
| Comparative | Same task under two or more ports, showing retrieval, prompt, response, and evaluation deltas. | When a method beats or loses to a baseline and we need a concrete explanation. | Pair or small set of runs with overlapping task IDs. | `case_studies/scripts/modes/comparative.py` |
| Counterfactual | Candidate bundle edits that would test whether one retrieved item or missing item was load-bearing. | When retrieval looks suspicious or a failed trace appears one concept away from success. | Single run, optional dry-run bundle edits. | `case_studies/scripts/modes/counterfactual.py` |
| Adversarial perturbation | Whether small input-grid edits or distractors break the method differently from the baseline. | After a promising case looks brittle or too benchmark-specific. | Single run plus original problem grids. | `case_studies/scripts/modes/adversarial.py` |
| Mechanistic attribution | Candidate links between retrieved concepts, prompt spans, response spans, and final predictions. | When we need to distinguish "retrieval present" from "retrieval used". | Single run, response text, retrieval bundle, optional future token attribution. | `case_studies/scripts/modes/mechanistic_attribution.py` |
| Provenance load-bearing | Which retrieved concepts appear necessary for a correct answer, and where those concepts came from. | For correct cases that should become paper evidence. | Single run plus concept provenance metadata if available. | `case_studies/scripts/modes/provenance_load_bearing.py` |
| Phase-shift envelope | Stability across repeated runs: output variance, retrieval variance, and correctness variance. | When temperature, sampling, or retry policy may dominate the result. | Multiple runs with the same port, seed family, or task set. | `case_studies/scripts/modes/phase_shift_envelope.py` |

## Operating Rule

Each report should be written under `case_studies/runs/<run_id>/analyses/<mode>.md` and linked from that run's `summary.md`. Derived reports should point back to raw traces rather than copying whole prompts and responses into many places.
