# Binary Security Patch Recognition

## Task

Use an agent to optimize a detector that predicts whether a binary diff is a
security patch.

`solution.py` must implement:

- `predict(samples: list[dict]) -> list[float]` (required)
- `train(samples: list[dict], labels: list[int]) -> None` (optional)

The grader trains on a dev split, then evaluates on a hidden split.

## Scoring

Composite score:

- `0.55 * F1@0.5`
- `0.25 * Recall@FPR<=0.10`
- `0.15 * ROC-AUC`
- `0.05 * latency_score`

Invalid outputs (non-list, wrong length, NaN, non-[0,1]) fail evaluation.

## Run

```bash
uv run coral start -c examples/binary_patch_recognition/task.yaml
```

## Files

```
examples/binary_patch_recognition/
├── README.md
├── task.yaml
├── seed/
│   └── solution.py
└── eval/
    ├── grader.py
    └── data/
        ├── dev.jsonl
        └── hidden.jsonl
```
