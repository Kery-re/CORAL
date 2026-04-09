"""Grader for Binary Security Patch Recognition task."""

from __future__ import annotations

import math
import textwrap
from pathlib import Path

from coral.grader import TaskGrader
from coral.types import ScoreBundle


class Grader(TaskGrader):
    """Evaluate hidden-set binary patch detection quality."""

    def evaluate(self) -> float | ScoreBundle:
        program_file = self.args.get("program_file", "solution.py")
        fpr_target = float(self.args.get("fpr_target", 0.10))
        latency_target_ms = float(self.args.get("latency_target_ms", 15.0))

        program_path = Path(self.codebase_path) / program_file
        if not program_path.exists():
            return self.fail(f"Program file ({program_file}) not found")

        dev_path = self.read_eval_path("data/dev.jsonl")
        hidden_path = self.read_eval_path("data/hidden.jsonl")

        script = textwrap.dedent(
            f"""
            import importlib.util
            import json
            import math
            import time

            program_path = {str(program_path)!r}
            dev_path = {str(dev_path)!r}
            hidden_path = {str(hidden_path)!r}
            fpr_target = {fpr_target}
            latency_target_ms = {latency_target_ms}

            def _load_jsonl(path):
                rows = []
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        rows.append(json.loads(line))
                return rows

            def _validate_scores(scores, expected_n):
                if not isinstance(scores, list):
                    return "predict() must return a list"
                if len(scores) != expected_n:
                    return f"predict() returned {{len(scores)}} scores, expected {{expected_n}}"
                for i, x in enumerate(scores):
                    try:
                        v = float(x)
                    except Exception:
                        return f"prediction at index {{i}} is not numeric"
                    if not math.isfinite(v):
                        return f"prediction at index {{i}} is not finite"
                    if v < 0.0 or v > 1.0:
                        return f"prediction at index {{i}} is out of [0,1]: {{v}}"
                return None

            def _binary_stats(labels, scores, threshold=0.5):
                tp = fp = tn = fn = 0
                for y, s in zip(labels, scores, strict=False):
                    pred = 1 if s >= threshold else 0
                    if y == 1 and pred == 1:
                        tp += 1
                    elif y == 0 and pred == 1:
                        fp += 1
                    elif y == 0 and pred == 0:
                        tn += 1
                    else:
                        fn += 1
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                return precision, recall, f1, fpr

            def _roc_auc(labels, scores):
                pos = [s for y, s in zip(labels, scores, strict=False) if y == 1]
                neg = [s for y, s in zip(labels, scores, strict=False) if y == 0]
                if not pos or not neg:
                    return 0.5
                wins = 0.0
                total = len(pos) * len(neg)
                for ps in pos:
                    for ns in neg:
                        if ps > ns:
                            wins += 1.0
                        elif ps == ns:
                            wins += 0.5
                return wins / total

            def _recall_at_fpr(labels, scores, target_fpr):
                thresholds = sorted(set(scores), reverse=True)
                best = 0.0
                for t in thresholds:
                    _, recall, _, fpr = _binary_stats(labels, scores, threshold=t)
                    if fpr <= target_fpr and recall > best:
                        best = recall
                return best

            dev_rows = _load_jsonl(dev_path)
            hidden_rows = _load_jsonl(hidden_path)
            if not dev_rows or not hidden_rows:
                print(json.dumps({{"error": "dev/hidden dataset is empty"}}))
                raise SystemExit(0)

            dev_samples = [{{k: v for k, v in r.items() if k != "label"}} for r in dev_rows]
            dev_labels = [int(r["label"]) for r in dev_rows]
            hidden_samples = [{{k: v for k, v in r.items() if k != "label"}} for r in hidden_rows]
            hidden_labels = [int(r["label"]) for r in hidden_rows]

            spec = importlib.util.spec_from_file_location("agent_solution", program_path)
            if spec is None or spec.loader is None:
                print(json.dumps({{"error": "failed to load solution module"}}))
                raise SystemExit(0)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if hasattr(module, "train"):
                try:
                    module.train(dev_samples, dev_labels)
                except Exception as e:
                    print(json.dumps({{"error": f"train() failed: {{e}}"}}))
                    raise SystemExit(0)

            if not hasattr(module, "predict"):
                print(json.dumps({{"error": "solution.py must define predict(samples)"}}))
                raise SystemExit(0)

            t0 = time.perf_counter()
            try:
                scores = module.predict(hidden_samples)
            except Exception as e:
                print(json.dumps({{"error": f"predict() failed: {{e}}"}}))
                raise SystemExit(0)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            err = _validate_scores(scores, len(hidden_samples))
            if err is not None:
                print(json.dumps({{"error": err}}))
                raise SystemExit(0)

            scores = [float(x) for x in scores]
            precision, recall, f1, fpr = _binary_stats(hidden_labels, scores, threshold=0.5)
            auc = _roc_auc(hidden_labels, scores)
            recall_low_fpr = _recall_at_fpr(hidden_labels, scores, fpr_target)

            per_sample_ms = elapsed_ms / max(1, len(hidden_samples))
            latency_score = max(0.0, min(1.0, 1.0 - (per_sample_ms / max(1e-6, latency_target_ms))))

            final_score = (
                0.55 * f1
                + 0.25 * recall_low_fpr
                + 0.15 * auc
                + 0.05 * latency_score
            )

            print(json.dumps({{
                "score": round(final_score, 6),
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
                "fpr_at_05": round(fpr, 6),
                "auc": round(auc, 6),
                "recall_at_low_fpr": round(recall_low_fpr, 6),
                "latency_per_sample_ms": round(per_sample_ms, 6),
                "latency_score": round(latency_score, 6),
                "n_hidden": len(hidden_samples),
            }}))
            """
        )

        try:
            result = self.run_script_json(script, timeout=self.timeout or 120)
        except Exception as e:
            return self.fail(f"Evaluation failed: {e}")

        if "error" in result:
            return self.fail(f"Error: {result['error']}")

        score = float(result["score"])
        explanation = (
            f"Score={score:.4f} | F1={result['f1']:.4f} | "
            f"Recall@FPR<={fpr_target:.2f}={result['recall_at_low_fpr']:.4f} | "
            f"AUC={result['auc']:.4f} | "
            f"Latency/sample={result['latency_per_sample_ms']:.3f}ms"
        )
        feedback = (
            f"Hidden samples: {result['n_hidden']}\n"
            f"Precision@0.5: {result['precision']:.4f}\n"
            f"Recall@0.5: {result['recall']:.4f}\n"
            f"FPR@0.5: {result['fpr_at_05']:.4f}\n"
            f"F1@0.5: {result['f1']:.4f}\n"
            f"ROC-AUC: {result['auc']:.4f}\n"
            f"Recall@FPR<={fpr_target:.2f}: {result['recall_at_low_fpr']:.4f}\n"
            f"Latency per sample (ms): {result['latency_per_sample_ms']:.3f}\n"
            f"Latency score: {result['latency_score']:.4f}"
        )
        return self.score(score, explanation, feedback=feedback)
