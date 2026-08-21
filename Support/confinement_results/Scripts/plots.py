import argparse
import csv
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import to_rgba

HAS_MPL = True

MODEL_COLORS = ["#0052C5", "#ff5500", "#007e76", "#ad0a87", "#000000"]  # one color per model
BOX_OUTLINE_COLOR = "#000000"  # subtle gray outline like default boxplot styling
OBJECTIVE_ORDER = ["base", "seg", "conf", "full", "nnlandmark"]
OBJECTIVE_LABELS = {
    "base": "base",
    "seg": "seg",
    "conf": "conf",
    "full": "full",
    "nnlandmark": "nnLandmark",
}
OBJECTIVE_COLORS = {
    "base": MODEL_COLORS[0],
    "seg": MODEL_COLORS[1],
    "conf": MODEL_COLORS[2],
    "full": MODEL_COLORS[3],
    "nnlandmark": MODEL_COLORS[4],
}


@dataclass
class ModelErrors:
    label: str
    path: str
    distances: np.ndarray  # shape: [num_samples, num_landmarks]
    objective: Optional[str] = None
    radius_px: Optional[float] = None
    dataset: Optional[str] = None
    segmentation_metrics: Optional[Dict[str, float]] = None


def _parse_model_arg(model_arg: str) -> Tuple[str, str]:
    if "=" in model_arg:
        label, path = model_arg.split("=", 1)
        return label.strip(), path.strip()

    path = model_arg.strip()
    parent = os.path.basename(os.path.dirname(path))
    label = parent if parent else os.path.basename(path)
    return label, path


def _load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def _normalize_objective_name(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    v = value.strip().lower()
    aliases = {
        "det": "base",
        "base": "base",
        "baseline": "base",
        "seg": "seg",
        "segmentation": "seg",
        "con": "conf",
        "conf": "conf",
        "confinement": "conf",
        "both": "full",
        "full": "full",
        "nnlandmark": "nnlandmark",
        "nn_landmark": "nnlandmark",
        "nn-landmark": "nnlandmark",
        "nn": "nnlandmark",
    }
    return aliases.get(v, v)


def _infer_objective_from_text(text: str) -> Optional[str]:
    if re.search(r"(?:^|[^a-z0-9])(?:nnlandmark|nn_landmark|nn-landmark)", text.lower()):
        return "nnlandmark"
    tokens = re.split(r"[^a-z0-9]+", text.lower())
    for token in tokens:
        mapped = _normalize_objective_name(token)
        if mapped in OBJECTIVE_ORDER:
            return mapped
    return None


def _to_float_or_none(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    if isinstance(value, (list, tuple)) and value:
        return _to_float_or_none(value[0])
    return None


def _normalize_dataset_name(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    v = str(value).strip()
    low = v.lower()
    if "imagetbad" in low:
        return "ImageTBAD"
    if "ctpels" in low:
        return "CTPelS"
    if "ctpel" in low:
        return "CTPel"
    if "hiea" in low:
        return "HIEA"
    if "_r" in v:
        v = v.split("_r", 1)[0]
    return v


def _infer_dataset_from_text(text: str) -> Optional[str]:
    low = text.lower()
    if "imagetbad" in low:
        return "ImageTBAD"
    if "ctpels" in low:
        return "CTPelS"
    if "ctpel" in low:
        return "CTPel"
    if "hiea" in low:
        return "HIEA"
    return None


def _infer_radius_from_text(text: str) -> Optional[float]:
    patterns = [
        r"(?:^|[^a-z0-9])(?:radius|rad|r)[_-]?(\d+(?:\.\d+)?)",
        r"(?:^|[^a-z0-9])(?:nnlandmark|nn_landmark|nn-landmark)[_-]?(\d+(?:\.\d+)?)",
        r"(?:^|[^a-z0-9])(?:base|det|seg|conf|con|full)[_-]?(\d+(?:\.\d+)?)",
        r"[_-](\d+(?:\.\d+)?)(?:\.[^.]+)?$",
    ]
    low = text.lower()
    for pattern in patterns:
        match = re.search(pattern, low)
        if match:
            return float(match.group(1))
    return None


def _extract_model_metadata(path: str, label: str) -> Tuple[Optional[str], Optional[float], Optional[str]]:
    """Infer (objective, radius_px, dataset) from train_overview-style JSON and label/filename fallback."""
    metadata_text = f"{label} {os.path.basename(path)}"
    objective = _infer_objective_from_text(metadata_text)
    radius_px: Optional[float] = _infer_radius_from_text(metadata_text)
    dataset = _infer_dataset_from_text(metadata_text)

    if os.path.splitext(path)[1].lower() == ".json":
        try:
            data = _load_json(path)
        except Exception:
            data = None
        if isinstance(data, dict):
            model_info = data.get("model", {}) if isinstance(data.get("model", {}), dict) else {}
            training_info = data.get("training", {}) if isinstance(data.get("training", {}), dict) else {}
            general_info = data.get("general", {}) if isinstance(data.get("general", {}), dict) else {}

            radius_from_model = _to_float_or_none(model_info.get("radius"))
            if radius_from_model is not None:
                radius_px = radius_from_model
            dataset_from_general = _normalize_dataset_name(general_info.get("dataset_name"))
            if dataset_from_general:
                dataset = dataset_from_general

            if "detailed_results" not in data:
                has_seg = False
                out_channels_seg = model_info.get("out_channels_seg")
                if isinstance(out_channels_seg, list):
                    has_seg = any(isinstance(v, (int, float)) and v >= 0 for v in out_channels_seg)

                has_conf = False
                conf_info = training_info.get("geometric_confinement")
                if isinstance(conf_info, dict):
                    has_conf = len(conf_info) > 0
                elif isinstance(conf_info, list):
                    has_conf = len(conf_info) > 0
                else:
                    has_conf = bool(conf_info)

                inferred = "full" if has_seg and has_conf else "seg" if has_seg else "conf" if has_conf else "det"
                objective = _normalize_objective_name(objective) if objective else inferred

    objective = _normalize_objective_name(objective)
    if objective == "nnlandmark":
        radius_px = None
    dataset = _normalize_dataset_name(dataset)
    return objective, radius_px, dataset


def _maybe_scale(coords: np.ndarray, resolution, units: str) -> np.ndarray:
    arr = np.asarray(coords)
    if units == "px":
        return arr
    return arr * np.asarray(resolution)


def _extract_pred_gt(path: str, units: str) -> Tuple[np.ndarray, np.ndarray]:
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npz":
        data = np.load(path)
        if "pred" in data and "gt" in data:
            return np.asarray(data["pred"]), np.asarray(data["gt"])
        if "predictions" in data and "ground_truth" in data:
            return np.asarray(data["predictions"]), np.asarray(data["ground_truth"])
        raise ValueError(f"{path}: .npz file must contain 'pred'/'gt' or 'predictions'/'ground_truth'.")

    data = _load_json(path)

    # train_overview.json format
    if isinstance(data, dict) and "general" in data and "raw_results" in data["general"]:
        pred, gt = data["general"]["raw_results"]
        resolution = data["model"]["image_resolution"]
        return _maybe_scale(pred, resolution, units), _maybe_scale(gt, resolution, units)

    # generic {"raw_results": [pred, gt]}
    if isinstance(data, dict) and "raw_results" in data:
        pred, gt = data["raw_results"]
        resolution = data["image_resolution"]
        return _maybe_scale(pred, resolution, units), _maybe_scale(gt, resolution, units)

    # generic {"pred": ..., "gt": ...}
    if isinstance(data, dict) and "pred" in data and "gt" in data:
        return np.asarray(data["pred"]), np.asarray(data["gt"])

    # generic {"predictions": ..., "ground_truth": ...}
    if isinstance(data, dict) and "predictions" in data and "ground_truth" in data:
        return np.asarray(data["predictions"]), np.asarray(data["ground_truth"])

    # raw list [pred, gt]
    if isinstance(data, list) and len(data) == 2:
        return np.asarray(data[0]), np.asarray(data[1])

    raise ValueError(
        f"{path}: unsupported format. Expected train_overview raw_results, "
        "'pred'/'gt', 'predictions'/'ground_truth', or [pred, gt]."
    )


def _landmark_sort_key(name: str) -> Tuple[int, object]:
    match = re.search(r"(\d+)$", str(name))
    if match:
        return 0, int(match.group(1))
    return 1, str(name)


def _extract_detailed_result_distances(path: str) -> Optional[np.ndarray]:
    if os.path.splitext(path)[1].lower() != ".json":
        return None

    data = _load_json(path)
    if not isinstance(data, dict):
        return None

    detailed_results = data.get("detailed_results")
    if not isinstance(detailed_results, dict) or not detailed_results:
        return None

    landmark_names = sorted(
        {
            landmark
            for case_results in detailed_results.values()
            if isinstance(case_results, dict)
            for landmark in case_results.keys()
        },
        key=_landmark_sort_key,
    )
    if not landmark_names:
        raise ValueError(f"{path}: detailed_results does not contain landmark errors.")

    rows: List[List[float]] = []
    for case_id in sorted(detailed_results.keys()):
        case_results = detailed_results[case_id]
        if not isinstance(case_results, dict):
            raise ValueError(f"{path}: detailed_results[{case_id!r}] must be an object.")
        row: List[float] = []
        for landmark in landmark_names:
            if landmark not in case_results:
                raise ValueError(f"{path}: detailed_results[{case_id!r}] is missing {landmark!r}.")
            value = case_results[landmark]
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(
                    f"{path}: detailed_results[{case_id!r}][{landmark!r}] must be a finite number."
                )
            row.append(float(value))
        rows.append(row)

    return np.asarray(rows, dtype=np.float64)


def _metric_value_to_float(value) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, np.number):
        return float(value)
    if isinstance(value, (list, tuple)) and value:
        return _metric_value_to_float(value[0])
    return None


def _extract_segmentation_metrics(path: str) -> Optional[Dict[str, float]]:
    """Return saved test Dice metrics.

    Convention in the training outputs:
    - scalar or first list element: detection/landmark-segmentation metric
    - second list element, when present: structure-segmentation metric
    """
    if os.path.splitext(path)[1].lower() != ".json":
        return None

    data = _load_json(path)
    if not isinstance(data, dict):
        return None

    general_info = data.get("general", {})
    if not isinstance(general_info, dict):
        return None

    metric = general_info.get("test_metrics", general_info.get("test_metric"))
    if metric is None:
        return None

    metrics: Dict[str, float] = {}
    if isinstance(metric, (list, tuple)):
        detection = _metric_value_to_float(metric[0]) if len(metric) >= 1 else None
        structure = _metric_value_to_float(metric[1]) if len(metric) >= 2 else None
        if detection is not None and math.isfinite(detection):
            metrics["detection"] = detection
        if structure is not None and math.isfinite(structure):
            metrics["structure"] = structure
    else:
        detection = _metric_value_to_float(metric)
        if detection is not None and math.isfinite(detection):
            metrics["detection"] = detection

    return metrics if metrics else None


def _normalize_shape(arr: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim < 2:
        raise ValueError(f"{name} must have at least 2 dims, got shape {arr.shape}.")

    # Flatten all leading dimensions into samples and keep [landmarks, dims].
    if arr.ndim == 2:
        arr = arr[:, np.newaxis, :]
    elif arr.ndim > 3:
        arr = arr.reshape(int(np.prod(arr.shape[:-2])), arr.shape[-2], arr.shape[-1])

    if arr.ndim != 3:
        raise ValueError(f"{name} must be [samples, landmarks, dims], got shape {arr.shape}.")
    if arr.shape[-1] < 2:
        raise ValueError(f"{name} last dimension should be coordinate dims (>=2), got {arr.shape[-1]}.")
    return arr


def _compute_distances(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pred_n = _normalize_shape(pred, "pred")
    gt_n = _normalize_shape(gt, "gt")

    if pred_n.shape != gt_n.shape:
        raise ValueError(f"pred and gt shape mismatch: {pred_n.shape} vs {gt_n.shape}.")

    diff = pred_n - gt_n
    return np.linalg.norm(diff, axis=-1)


def _median_iqr(values: np.ndarray) -> Tuple[float, float, float, float]:
    median = float(np.median(values))
    q1 = float(np.percentile(values, 25))
    q3 = float(np.percentile(values, 75))
    return median, q1, q3, q3 - q1


def compute_overall_median_iqr(models: List[ModelErrors]) -> List[dict]:
    """Return overall median and IQR stats for each model.

    Output format:
    [
      {"model": <label>, "median": <float>, "q1": <float>, "q3": <float>, "iqr": <float>},
      ...
    ]
    """
    stats: List[dict] = []
    for model in models:
        med, q1, q3, iqr = _median_iqr(model.distances.reshape(-1))
        stats.append(
            {
                "model": model.label,
                "median": med,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
            }
        )
    return stats


def compute_global_median_iqr(models: List[ModelErrors]) -> dict:
    """Return one overall median/IQR computed from all model distances together."""
    if not models:
        raise ValueError("No models provided.")

    all_values = np.concatenate([m.distances.reshape(-1) for m in models], axis=0)
    med, q1, q3, iqr = _median_iqr(all_values)
    return {
        "median": med,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "n_values": int(all_values.size),
    }


def _print_stats(models: List[ModelErrors]) -> None:
    global_stats = compute_global_median_iqr(models)
    print("\nGlobal overall error statistics (all models + landmarks + samples)")
    print("median\tq1\tq3\tiqr\tn")
    print(
        f"{global_stats['median']:.4f}\t{global_stats['q1']:.4f}\t"
        f"{global_stats['q3']:.4f}\t{global_stats['iqr']:.4f}\t{global_stats['n_values']}"
    )

    print("\nOverall error statistics")
    print("model\tmedian\tq1\tq3\tiqr")
    for row in compute_overall_median_iqr(models):
        print(f"{row['model']}\t{row['median']:.4f}\t{row['q1']:.4f}\t{row['q3']:.4f}\t{row['iqr']:.4f}")

    print("\nPer-landmark error statistics")
    for model in models:
        print(f"\n[{model.label}]")
        print("lm\tmedian\tq1\tq3\tiqr")
        for idx in range(model.distances.shape[1]):
            med, q1, q3, iqr = _median_iqr(model.distances[:, idx])
            print(f"{idx + 1}\t{med:.4f}\t{q1:.4f}\t{q3:.4f}\t{iqr:.4f}")


def _aggregate_cases(distances: np.ndarray, method: str) -> np.ndarray:
    d = np.asarray(distances, dtype=np.float64)
    if d.ndim == 1:
        return d
    if method == "mean":
        return np.mean(d, axis=1)
    if method == "median":
        return np.median(d, axis=1)
    raise ValueError(f"Unsupported case aggregation: {method}")


def _format_threshold_name(threshold: float) -> str:
    return f"{threshold:g}".replace(".", "p")


def _model_group_key(model: ModelErrors) -> Tuple[str, str]:
    dataset = model.dataset if model.dataset else "Unknown"
    if model.radius_px is None:
        radius = "unknown"
    elif float(model.radius_px).is_integer():
        radius = str(int(model.radius_px))
    else:
        radius = f"{model.radius_px:g}"
    return dataset, radius


def _format_dataset_suptitle(models: Sequence[ModelErrors], title: str) -> str:
    datasets: List[str] = []
    for model in models:
        dataset = model.dataset if model.dataset else "Unknown"
        if dataset not in datasets:
            datasets.append(dataset)
    return f"{', '.join(datasets)}: {title}"


def _compute_metric_rows(
    models: List[ModelErrors],
    thresholds: Sequence[float],
    failure_threshold: float,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for model in models:
        values = np.asarray(model.distances, dtype=np.float64).reshape(-1)
        med, q1, q3, iqr = _median_iqr(values)
        row: Dict[str, object] = {
            "dataset": model.dataset if model.dataset else "Unknown",
            "radius_px": "" if model.radius_px is None else f"{model.radius_px:g}",
            "objective": model.objective if model.objective else "",
            "model": model.label,
            "n_cases": int(np.asarray(model.distances).shape[0]),
            "n_landmarks": int(np.asarray(model.distances).shape[1]) if np.asarray(model.distances).ndim > 1 else 1,
            "n_points": int(values.size),
            "median": med,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "catastrophic_failure_rate": float(np.mean(values > failure_threshold)),
            "catastrophic_failures": int(np.sum(values > failure_threshold)),
        }
        for threshold in thresholds:
            row[f"sdr_le_{_format_threshold_name(threshold)}"] = 100.0 * float(np.mean(values <= threshold))
        rows.append(row)
    return rows


def _binom_two_sided_pvalue(successes: int, trials: int) -> float:
    if trials == 0:
        return float("nan")
    observed = min(successes, trials - successes)
    p = 0.0
    for k in range(0, observed + 1):
        p += math.comb(trials, k) * (0.5 ** trials)
    return float(min(1.0, 2.0 * p))


def _paired_sign_test(diff: np.ndarray) -> Tuple[int, int, float]:
    nonzero = diff[np.abs(diff) > 0]
    n = int(nonzero.size)
    wins = int(np.sum(nonzero < 0))
    p_value = _binom_two_sided_pvalue(wins, n)
    return wins, n, p_value


def _paired_sign_flip_permutation_pvalue(diff: np.ndarray, max_exact_n: int = 20) -> Tuple[float, str]:
    nonzero = diff[np.abs(diff) > 0]
    n = int(nonzero.size)
    if n == 0:
        return float("nan"), "all_ties"
    observed = abs(float(np.mean(nonzero)))
    if n <= max_exact_n:
        total = 1 << n
        extreme = 0
        abs_diff = np.abs(nonzero)
        for mask in range(total):
            signs = np.ones(n, dtype=np.float64)
            for bit in range(n):
                if (mask >> bit) & 1:
                    signs[bit] = -1.0
            statistic = abs(float(np.mean(signs * abs_diff)))
            if statistic >= observed - 1e-12:
                extreme += 1
        return float(extreme / total), "exact_sign_flip"

    rng = np.random.default_rng(23)
    n_permutations = 100000
    abs_diff = np.abs(nonzero)
    signs = rng.choice([-1.0, 1.0], size=(n_permutations, n))
    statistics = np.abs(np.mean(signs * abs_diff, axis=1))
    p_value = (float(np.sum(statistics >= observed - 1e-12)) + 1.0) / (n_permutations + 1.0)
    return p_value, f"monte_carlo_sign_flip_{n_permutations}"


def _select_reference_model(group: List[ModelErrors], reference: str) -> Optional[ModelErrors]:
    ref_norm = _normalize_objective_name(reference)
    for model in group:
        if model.label == reference:
            return model
    for model in group:
        if model.objective == ref_norm:
            return model
    return None


def _compute_test_rows(
    models: List[ModelErrors],
    reference: str,
    case_aggregate: str,
) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, str], List[ModelErrors]] = {}
    for model in models:
        groups.setdefault(_model_group_key(model), []).append(model)

    rows: List[Dict[str, object]] = []
    for (dataset, radius), group in groups.items():
        reference_model = _select_reference_model(group, reference)
        if reference_model is None:
            continue
        ref_cases = _aggregate_cases(reference_model.distances, case_aggregate)
        for model in group:
            if model is reference_model:
                continue
            cases = _aggregate_cases(model.distances, case_aggregate)
            if cases.shape != ref_cases.shape:
                rows.append(
                    {
                        "dataset": dataset,
                        "radius_px": radius,
                        "reference_model": reference_model.label,
                        "model": model.label,
                        "case_aggregate": case_aggregate,
                        "n_cases": "",
                        "median_delta": "",
                        "mean_delta": "",
                        "wins": "",
                        "non_ties": "",
                        "sign_test_p": "",
                        "permutation_p": "",
                        "permutation_method": "skipped_shape_mismatch",
                    }
                )
                continue
            diff = cases - ref_cases
            wins, non_ties, sign_p = _paired_sign_test(diff)
            perm_p, perm_method = _paired_sign_flip_permutation_pvalue(diff)
            rows.append(
                {
                    "dataset": dataset,
                    "radius_px": radius,
                    "reference_model": reference_model.label,
                    "model": model.label,
                    "case_aggregate": case_aggregate,
                    "n_cases": int(diff.size),
                    "median_delta": float(np.median(diff)),
                    "mean_delta": float(np.mean(diff)),
                    "wins": wins,
                    "non_ties": non_ties,
                    "sign_test_p": sign_p,
                    "permutation_p": perm_p,
                    "permutation_method": perm_method,
                }
            )
    return rows


def _write_tsv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _format_table_value(value: object) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.4f}"
    return str(value)


def _write_latex_table(path: str, rows: List[Dict[str, object]], caption: str, label: str) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w") as f:
        f.write("\\begin{table}[h]\n\\centering\n\\small\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write(f"\\label{{{label}}}\n")
        f.write("\\begin{tabular}{" + "l" * len(fieldnames) + "}\n")
        f.write("\\hline\n")
        f.write(" & ".join(fieldnames).replace("_", "\\_") + " \\\\\n")
        f.write("\\hline\n")
        for row in rows:
            f.write(" & ".join(_format_table_value(row.get(field, "")) for field in fieldnames) + " \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")


def _write_statistics_outputs(
    models: List[ModelErrors],
    thresholds: Sequence[float],
    failure_threshold: float,
    output_dir: str,
    reference: str,
    case_aggregate: str,
    write_latex: bool,
) -> Tuple[List[str], List[Dict[str, object]], List[Dict[str, object]]]:
    metric_rows = _compute_metric_rows(models, thresholds, failure_threshold)
    test_rows = _compute_test_rows(models, reference, case_aggregate)

    written: List[str] = []
    summary_path = os.path.join(output_dir, "stats_summary.tsv")
    _write_tsv(summary_path, metric_rows)
    written.append(summary_path)

    tests_path = os.path.join(output_dir, "stats_tests.tsv")
    _write_tsv(tests_path, test_rows)
    written.append(tests_path)

    assumptions_path = os.path.join(output_dir, "stats_assumptions.txt")
    with open(assumptions_path, "w") as f:
        f.write("Statistics are computed from train_overview raw_results predictions and ground truth.\n")
        f.write("Paired tests assume that models use the same test cases in the same order within each dataset/radius group.\n")
        f.write(f"SDR thresholds: {', '.join(f'{t:g}' for t in thresholds)} in selected CLI units.\n")
        f.write(f"Catastrophic failure threshold: > {failure_threshold:g} in selected CLI units.\n")
        f.write(f"Per-case test aggregation: {case_aggregate} across landmarks.\n")
        f.write(f"Reference selector: {reference} (matched first by label, then by objective).\n")
    written.append(assumptions_path)

    if write_latex:
        summary_tex = os.path.join(output_dir, "stats_summary.tex")
        _write_latex_table(
            summary_tex,
            metric_rows,
            "Localization statistics including median, IQR, SDR, and catastrophic failure rate.",
            "tab:stats_summary",
        )
        written.append(summary_tex)

        tests_tex = os.path.join(output_dir, "stats_tests.tex")
        _write_latex_table(
            tests_tex,
            test_rows,
            "Paired statistical tests against the selected reference model.",
            "tab:stats_tests",
        )
        written.append(tests_tex)

    return written, metric_rows, test_rows


def _plot_boxplot(models: List[ModelErrors], output_dir: str, dpi: int) -> None:
    labels = [m.label for m in models]
    values = [m.distances.reshape(-1) for m in models]

    n_models = len(models)
    box_width = 0.7

    fig, ax = plt.subplots(figsize=(min(8, int(1.7 * n_models)), 6))
    bp = ax.boxplot(values, tick_labels=labels, widths=box_width, showfliers=True, patch_artist=True)
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(_model_color(i))
        box.set_alpha(0.7)
    for median in bp["medians"]:
        median.set_color(BOX_OUTLINE_COLOR)
        median.set_linewidth(1.4)
    for i, flier in enumerate(bp["fliers"]):
        flier.set_markerfacecolor(_model_color(i))
        flier.set_alpha(0.7)
    ax.set_title("Euclidean Error Distribution Per Model")
    ax.set_ylabel("Euclidean distance (px)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "boxplot_overall.png"), dpi=dpi)
    plt.close(fig)


def _safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "model"


def _model_color(index: int) -> str:
    return MODEL_COLORS[index % len(MODEL_COLORS)]


def _plot_per_model_landmark_boxplots(
    models: List[ModelErrors],
    output_dir: str,
    landmark_names: Optional[List[str]],
    dpi: int,
) -> None:
    box_width = 0.7
    for model_idx, model in enumerate(models):
        n_landmarks = model.distances.shape[1]
        if landmark_names is not None and len(landmark_names) == n_landmarks:
            x_labels = landmark_names
        else:
            x_labels = [str(i + 1) for i in range(n_landmarks)]
        values = [model.distances[:, idx] for idx in range(n_landmarks)]
        fig, ax = plt.subplots(figsize=(min(8, int(1.7 * n_landmarks)), 6))
        color = _model_color(model_idx)
        bp = ax.boxplot(values, tick_labels=x_labels, showfliers=True, widths=box_width, patch_artist=True)
        for box in bp["boxes"]:
            box.set_facecolor(color)
            box.set_alpha(0.7)
        for median in bp["medians"]:
            median.set_color(BOX_OUTLINE_COLOR)
            median.set_linewidth(1.4)
        for flier in bp["fliers"]:
            flier.set_markerfacecolor(color)
            flier.set_alpha(0.7)
        ax.set_title(f"{model.label}: Per-Landmark Error Distribution")
        ax.set_xlabel("Landmark")
        ax.set_ylabel("Euclidean distance (px)")
        ax.grid(axis="y", alpha=0.3)
        if landmark_names is not None:
            ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(
            os.path.join(output_dir, f"boxplot_per_landmark_{_safe_filename(model.label)}.png"),
            dpi=dpi,
        )
        plt.close(fig)


def _plot_grouped_landmark_model_boxplots(
    models: List[ModelErrors],
    output_dir: str,
    landmark_names: Optional[List[str]],
    dpi: int,
) -> None:
    n_models = len(models)
    n_landmarks = min(int(np.asarray(m.distances).shape[1]) for m in models)
    if n_landmarks <= 0:
        raise ValueError("No shared landmarks available for grouped landmark/model boxplot.")
    # Keep landmark groups compact while still separating model boxes inside each group.
    group_spacing = 0.6
    group_centers = np.arange(n_landmarks) * group_spacing
    offsets = np.linspace(-0.2, 0.2, n_models) if n_models > 1 else np.array([0.0])
    box_width = 0.45 / n_models

    colors = [_model_color(i) for i in range(n_models)]

    fig, ax = plt.subplots(figsize=(min(9, int(1.7 * n_landmarks)), 6))

    # Plot by landmark first, then split each landmark group by model.
    for lm_idx in range(n_landmarks):
        for model_idx, model in enumerate(models):
            values = model.distances[:, lm_idx]
            pos = group_centers[lm_idx] + offsets[model_idx]
            bp = ax.boxplot(
                [values],
                positions=[pos],
                widths=box_width,
                patch_artist=True,
                showfliers=True,
                manage_ticks=False,
            )
            for box in bp["boxes"]:
                box.set_facecolor(colors[model_idx])
                box.set_alpha(0.7)
            for median in bp["medians"]:
                median.set_color(BOX_OUTLINE_COLOR)
                median.set_linewidth(1.4)
            for flier in bp["fliers"]:
                flier.set_markerfacecolor(colors[model_idx])
                flier.set_alpha(0.7)

    ax.set_title("Per-Landmark Errors Grouped By Landmark, Split By Model")
    ax.set_xlabel("Landmark")
    ax.set_ylabel("Euclidean distance (px)")
    ax.set_xticks(group_centers)
    if landmark_names is None:
        ax.set_xticklabels([str(i + 1) for i in range(n_landmarks)])
    else:
        ax.set_xticklabels(landmark_names, rotation=30, ha="right")
    for center in group_centers[:-1]:
        ax.axvline(center + (group_spacing / 2), color="gray", alpha=0.4, linewidth=1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(
        handles=[
            Patch(facecolor=colors[i], edgecolor="black", alpha=0.7, label=models[i].label)
            for i in range(n_models)
        ],
        title="Model",
    )
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "boxplot_grouped_landmark_model.png"), dpi=dpi)
    plt.close(fig)


def _plot_sample_bar_comparison_per_landmark(
    models: List[ModelErrors],
    output_dir: str,
    landmark_names: Optional[List[str]],
    dpi: int,
    filename_prefix: str = "sample_bar_comparison",
) -> List[str]:
    """Plot per-sample model comparison bars, one figure per landmark.

    - One figure per landmark.
    - One subplot per sample.
    - In each subplot: one bar per model (method), colored by model color.
    """
    if not models:
        return []

    dist_arrays = []
    sample_counts = []
    landmark_counts = []
    for m in models:
        d = np.asarray(m.distances)
        if d.ndim == 1:
            d = d[:, np.newaxis]
        elif d.ndim > 2:
            d = d.reshape(d.shape[0], -1)
        dist_arrays.append(d)
        sample_counts.append(int(d.shape[0]))
        landmark_counts.append(int(d.shape[1]))

    if len(set(sample_counts)) != 1:
        raise ValueError(
            "Sample counts must match across models for per-sample bar plots. "
            f"Got counts: {sample_counts}"
        )
    if len(set(landmark_counts)) != 1:
        raise ValueError(
            "Landmark counts must match across models for per-sample bar plots. "
            f"Got counts: {landmark_counts}"
        )

    n_samples = sample_counts[0]
    n_landmarks = landmark_counts[0]
    if n_samples <= 0 or n_landmarks <= 0:
        return []

    model_labels = [m.label for m in models]
    model_colors = [_model_color(i) for i in range(len(models))]
    model_x = np.arange(len(models))

    # Choose a compact grid that scales to typical sample counts.
    n_cols = max(1, min(6, int(np.ceil(np.sqrt(n_samples)))))
    n_rows = int(np.ceil(n_samples / n_cols))
    saved_files: List[str] = []

    y_max = max(float(np.max(d)) for d in dist_arrays) * 1.08
    y_max = max(y_max, 1e-6)

    for lm_idx in range(n_landmarks):
        lm_name = landmark_names[lm_idx] if landmark_names is not None and lm_idx < len(landmark_names) else f"lm_{lm_idx + 1}"
        safe_lm = _safe_filename(lm_name)

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(max(10, 2.4 * n_cols), max(4, 2.1 * n_rows)),
            sharey=True,
            squeeze=False,
        )
        axes_flat = axes.ravel()

        for sample_idx in range(n_samples):
            ax = axes_flat[sample_idx]
            vals = [float(d[sample_idx, lm_idx]) for d in dist_arrays]
            ax.bar(model_x, vals, color=model_colors, edgecolor=BOX_OUTLINE_COLOR, linewidth=0.6, alpha=0.85)
            ax.set_title(f"Sample {sample_idx + 1}", fontsize=8)
            ax.set_ylim(0, y_max)
            ax.grid(axis="y", alpha=0.2)
            ax.set_xticks(model_x)
            if sample_idx // n_cols == n_rows - 1:
                ax.set_xticklabels(model_labels, rotation=35, ha="right", fontsize=7)
            else:
                ax.set_xticklabels([])

        for ax in axes_flat[n_samples:]:
            ax.axis("off")

        axes_flat[0].set_ylabel("Euclidean distance (px)")
        fig.suptitle(f"Per-Sample Model Comparison - {lm_name}", y=0.98)
        fig.tight_layout(rect=(0, 0, 1, 0.95))

        out_name = f"{filename_prefix}_{safe_lm}.png"
        fig.savefig(os.path.join(output_dir, out_name), dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved_files.append(out_name)

    return saved_files


def _plot_median_iqr(
    models: List[ModelErrors],
    output_dir: str,
    landmark_names: Optional[List[str]],
    dpi: int,
) -> None:
    normalized_distances = []
    landmark_counts = []
    for model in models:
        d = np.asarray(model.distances)
        if d.ndim == 1:
            d = d[:, np.newaxis]
        elif d.ndim > 2:
            d = d.reshape(d.shape[0], -1)
        normalized_distances.append(d)
        landmark_counts.append(int(d.shape[1]))

    n_landmarks = min(landmark_counts)
    if n_landmarks <= 0:
        raise ValueError("No landmarks available for median/IQR plot.")
    x = np.arange(n_landmarks)

    fig, ax = plt.subplots(figsize=(min(8, int(1.7 * n_landmarks)), 6))
    offsets = np.linspace(-0.25, 0.25, num=len(models)) if len(models) > 1 else np.array([0.0])

    for i, model in enumerate(models):
        color = _model_color(i)
        d = normalized_distances[i][:, :n_landmarks]
        med = np.median(d, axis=0)
        q1 = np.percentile(d, 25, axis=0)
        q3 = np.percentile(d, 75, axis=0)
        err_low = med - q1
        err_high = q3 - med

        ax.errorbar(
            x + offsets[i],
            med,
            yerr=[err_low, err_high],
            fmt="o-",
            capsize=4,
            linewidth=1.7,
            color=color,
            label=model.label,
        )

    ax.set_title("Per-Landmark Median Error With IQR")
    ax.set_xlabel("Landmark")
    ax.set_ylabel("Euclidean distance (px)")
    ax.grid(axis="y", alpha=0.3)
    ax.set_xticks(x)
    if landmark_names is None:
        ax.set_xticklabels([str(i + 1) for i in x])
    else:
        ax.set_xticklabels(landmark_names[:n_landmarks], rotation=30, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "per_landmark_median_iqr.png"), dpi=dpi)
    plt.close(fig)


def _plot_threshold_groups(
    models: List[ModelErrors],
    thresholds: List[float],
    output_dir: str,
    landmark_names: Optional[List[str]],
    dpi: int,
) -> None:
    t1, t2, t3 = thresholds
    labels = [f"<= {t1:g}mm", f"{t1:g}mm - {t2:g}mm", f"{t2:g}mm - {t3:g}mm", f"> {t3:g}mm"]
    markers = ["o", "s", "^", "D"]

    n_models = len(models)
    n_landmarks = models[0].distances.shape[1]
    x_base = np.arange(n_landmarks)
    offsets = np.linspace(-0.28, 0.28, num=n_models) if n_models > 1 else np.array([0.0])
    box_width = 0.45 / max(1, n_models)
    jitter_width = 0.05
    rng = np.random.default_rng(23)

    fig, ax = plt.subplots(figsize=(min(10, int(1.7 * n_landmarks)), 6))

    for idx, model in enumerate(models):
        model_color = _model_color(idx)
        d = model.distances

        # Use real (unthresholded) values for the boxplots.
        bp = ax.boxplot(
            [d[:, lm_idx] for lm_idx in range(n_landmarks)],
            positions=x_base + offsets[idx],
            widths=box_width,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
        )
        for box in bp["boxes"]:
            box.set_facecolor(model_color)
            box.set_alpha(0.4)
            box.set_edgecolor(BOX_OUTLINE_COLOR)
            box.set_linewidth(1.1)
        for whisker in bp["whiskers"]:
            whisker.set_color(BOX_OUTLINE_COLOR)
            whisker.set_linewidth(1.1)
        for cap in bp["caps"]:
            cap.set_color(BOX_OUTLINE_COLOR)
            cap.set_linewidth(1.1)
        for median in bp["medians"]:
            median.set_color(BOX_OUTLINE_COLOR)
            median.set_linewidth(1.3)

        for lm_idx in range(n_landmarks):
            lm_values = d[:, lm_idx]
            lm_bins = np.digitize(lm_values, bins=[t1, t2, t3], right=True)
            for b in range(4):
                group_values = lm_values[lm_bins == b]
                if group_values.size == 0:
                    continue
                x_scatter = np.full(group_values.size, x_base[lm_idx] + offsets[idx], dtype=np.float64)
                x_scatter = x_scatter + rng.uniform(-jitter_width, jitter_width, size=group_values.size)
                y_scatter = np.full(group_values.size, 1.05 * t3, dtype=np.float64) if b == 3 else group_values
                ax.scatter(
                    x_scatter,
                    y_scatter,
                    s=16,
                    alpha=0.55,
                    c=model_color,
                    marker=markers[b],
                    edgecolors="none",
                    zorder=4,
                )

    for t in thresholds:
        ax.axhline(t, color="gray", linestyle="--", linewidth=1, alpha=0.6)

    ax.set_xlabel("Landmark")
    ax.set_ylabel("Euclidean distance (px)")
    ax.set_xticks(x_base)
    if landmark_names is None:
        ax.set_xticklabels([str(i + 1) for i in range(n_landmarks)])
    else:
        ax.set_xticklabels(landmark_names, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.2)
    ax.yaxis.set_minor_locator(MultipleLocator(1.0))
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax.grid(axis="y", which="minor", alpha=0.4, linestyle=":")
    ax.set_yticks([t1, t2, t3])
    y_max_real = max(float(np.max(m.distances)) for m in models)
    ax.set_ylim((0, t3 * 1.5))

    model_handles = [
        Patch(
            facecolor=_model_color(i),
            edgecolor=BOX_OUTLINE_COLOR,
            alpha=0.4,
            label=models[i].label,
        )
        for i in range(n_models)
    ]
    threshold_handles = [
        Line2D(
            [0],
            [0],
            marker=markers[i],
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.2,
            color="black",
            markersize=7,
            label=labels[i],
        )
        for i in range(4)
    ]
    fig.suptitle(" ", y=0.9)
    model_legend = fig.legend(
        handles=model_handles,
        title="Model",
        frameon=True,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=min(4, max(1, n_models)),
        fontsize=9,
        title_fontsize=10,
        markerscale=1.1,
        borderpad=0.45,
        labelspacing=0.45,
        handletextpad=0.6,
        columnspacing=1.0,
    )
    fig.add_artist(model_legend)
    # fig.legend(
    #     handles=threshold_handles,
    #     title="Threshold group",
    #     frameon=True,
    #     loc="upper left",
    #     bbox_to_anchor=(0.40, 0.955, 0.52, 0.0),
    #     mode="expand",
    #     ncol=4,
    #     fontsize=9,
    #     title_fontsize=10,
    #     markerscale=1.1,
    #     borderpad=0.45,
    #     labelspacing=0.45,
    #     handletextpad=0.6,
    #     columnspacing=1.0,
    # )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(os.path.join(output_dir, "per_landmark_threshold_groups.png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_stacked_landmark_sdr_bars_by_threshold(
    models: List[ModelErrors],
    thresholds: List[float],
    output_dir: str,
    dpi: int,
    units: str,
    landmark_names: Optional[List[str]],
    filename: str = "stacked_landmark_sdr_bars_by_threshold.png",
) -> str:
    """Plot mean SDR bars stacked by per-landmark SDR contribution."""
    valid_models = [m for m in models if m.objective in OBJECTIVE_ORDER]
    if not valid_models:
        raise ValueError("No models with recognized objectives found for stacked landmark SDR plot.")

    radius_models = [m for m in valid_models if m.objective != "nnlandmark"]
    nnlandmark_models = [m for m in valid_models if m.objective == "nnlandmark"]
    if not radius_models:
        raise ValueError("No radius/objective models found for stacked landmark SDR plot.")

    nnlandmark_by_dataset: Dict[str, List[ModelErrors]] = {}
    for model in nnlandmark_models:
        dataset = model.dataset if model.dataset else "Unknown"
        nnlandmark_by_dataset.setdefault(dataset, []).append(model)

    objectives = [
        obj
        for obj in OBJECTIVE_ORDER
        if obj != "nnlandmark" and any(m.objective == obj for m in radius_models)
    ]

    group_order: List[Tuple[str, str]] = []
    grouped: Dict[Tuple[str, str], List[ModelErrors]] = {}
    for model in radius_models:
        key = _model_group_key(model)
        if key not in grouped:
            grouped[key] = []
            group_order.append(key)
        grouped[key].append(model)

    grouped_dataset_names = {dataset for dataset, _ in group_order}
    nnlandmark_plot_models: List[ModelErrors] = []
    for dataset in grouped_dataset_names:
        nnlandmark_plot_models.extend(nnlandmark_by_dataset.get(dataset, []))
    if not nnlandmark_plot_models and len(grouped_dataset_names) == 1:
        nnlandmark_plot_models.extend(nnlandmark_by_dataset.get("Unknown", []))
    has_nnlandmark_plot = bool(nnlandmark_plot_models)

    n_groups = len(group_order)
    if has_nnlandmark_plot:
        n_cols = n_groups + 1
        fig_width = 5.0 * n_groups + 2.2
        fig_height = 5.0
        fig = plt.figure(figsize=(fig_width, fig_height))
        grid = fig.add_gridspec(1, n_cols, width_ratios=[1.0] * n_groups + [0.44], wspace=0.03)
        axes_flat = np.asarray([fig.add_subplot(grid[0, idx]) for idx in range(n_groups)], dtype=object)
        nnlandmark_ax = fig.add_subplot(grid[0, n_groups], sharey=axes_flat[0])
    else:
        n_cols = min(2, n_groups)
        n_rows = int(np.ceil(n_groups / n_cols))
        fig_width = 5.0 * n_cols
        fig_height = 4.8 * n_rows
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False, sharey=True)
        axes_flat = axes.ravel()
        nnlandmark_ax = None

    threshold_x = np.arange(len(thresholds))
    total_width = 0.78
    bar_width = total_width / max(1, len(objectives))
    offsets = (
        np.linspace(-total_width / 2 + bar_width / 2, total_width / 2 - bar_width / 2, len(objectives))
        if len(objectives) > 1
        else np.array([0.0])
    )
    landmark_hatches = ["///", ""]

    def add_bar_label(ax, positions: list, matching: list, thresholds: list) -> None:
        matching = [round(100.0 * float(np.mean(np.asarray(m.distances) <= threshold))) for m in matching for threshold in thresholds]
        print("\t".join([str(value) for value in matching]))
        for x_pos, value in zip(positions, matching):
            if value < 0.05:
                y_pos = value + 4
            else:
                y_pos = value + 3
            ax.text(
                x_pos,
                y_pos,
                value,
                ha="center",
                va="top",
                fontsize=8,
                color="black",
                clip_on=True,
                zorder=6,
            )

    def _draw_stacked_sdr_bars(
        ax,
        matching: List[ModelErrors],
        positions: np.ndarray,
        width: float,
        color: str,
        label: str,
    ) -> None:
        arrays = [np.asarray(m.distances, dtype=np.float64) for m in matching]
        n_landmarks = min(arr.shape[1] for arr in arrays)
        values = np.concatenate([arr[:, :n_landmarks] for arr in arrays], axis=0)
        bottoms = np.zeros(len(thresholds), dtype=np.float64)

        for lm_idx in range(n_landmarks):
            per_threshold_sdr = np.array(
                [100.0 * float(np.mean(values[:, lm_idx] <= threshold)) for threshold in thresholds],
                dtype=np.float64,
            )
            # Divide by the number of landmarks so the full stack is the mean SDR.
            heights = per_threshold_sdr / n_landmarks
            alpha = 0.4 + 0.25 * ((lm_idx + 1) / max(1, n_landmarks))
            ax.bar(
                positions,
                heights,
                bottom=bottoms,
                width=width,
                color=color,
                edgecolor=BOX_OUTLINE_COLOR,
                linewidth=0.7,
                alpha=alpha,
                hatch=landmark_hatches[lm_idx % len(landmark_hatches)],
                label=label if lm_idx == 0 else None,
            )
            bottoms += heights

    def _style_sdr_axis(ax, show_ylabel: bool) -> None:
        ax.set_xticks(threshold_x)
        ax.set_xticklabels([f"{threshold:g}" for threshold in thresholds])
        ax.grid(axis="y", alpha=0.2)
        ax.yaxis.set_minor_locator(MultipleLocator(10))
        ax.tick_params(axis="x", which="minor", bottom=False, top=False)
        ax.grid(axis="y", which="minor", alpha=0.4, linestyle=":")
        if show_ylabel:
            ax.set_ylabel("Success Detection Rate (%)", fontsize=13)
        ax.set_ylim(0.0, 105)
        ax.grid(axis="y", alpha=0.25)

    print("\nSuccess detection rate by threshold and objective:")
    print(f"model\t {'px\t'.join([str(int(value)) for value in thresholds])+'px'}")
    for ax_idx, key in enumerate(group_order):
        ax = axes_flat[ax_idx]
        dataset, radius = key
        group_models = grouped[key]

        for obj_idx, objective in enumerate(objectives):
            matching = [m for m in group_models if m.objective == objective]
            if not matching:
                continue
            color = OBJECTIVE_COLORS.get(objective, _model_color(obj_idx))
            positions = threshold_x + offsets[obj_idx]
            _draw_stacked_sdr_bars(
                ax,
                matching=matching,
                positions=positions,
                width=bar_width * 0.92,
                color=color,
                label=OBJECTIVE_LABELS.get(objective, objective),
            )
            print(f"{objective}{radius}\t", end=" ")
            add_bar_label(ax, positions, matching, thresholds)
            if ax_idx != 0:
                ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
        title = dataset if radius == "unknown" else f"Radius = {radius}px"
        ax.set_title(title, fontsize=13)
        _style_sdr_axis(ax, show_ylabel=(ax_idx % n_cols == 0))

    if has_nnlandmark_plot and nnlandmark_ax is not None:
        _draw_stacked_sdr_bars(
            nnlandmark_ax,
            matching=nnlandmark_plot_models,
            positions=threshold_x,
            width=0.54*total_width,
            color=OBJECTIVE_COLORS.get("nnLandmark", _model_color(obj_idx+1)),
            label="nnLandmark",
        )
        print("nnLM\t", end=" ")
        add_bar_label(nnlandmark_ax, threshold_x, nnlandmark_plot_models, thresholds)
        nnlandmark_ax.set_xlim(-0.5, 2.5)
        nnlandmark_ax.set_title("nnLandmark", fontsize=13)
        _style_sdr_axis(nnlandmark_ax, show_ylabel=False)
        nnlandmark_ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
        nnlandmark_ax.set_ylabel("")
    elif not has_nnlandmark_plot:
        for ax in axes_flat[n_groups:]:
            ax.axis("off")

    objective_handles = [
        Patch(
            facecolor=OBJECTIVE_COLORS.get(obj, _model_color(i)),
            edgecolor=BOX_OUTLINE_COLOR,
            alpha=0.6,
            label=OBJECTIVE_LABELS.get(obj, obj),
        )
        for i, obj in enumerate(objectives)
    ]
    max_landmarks = max(int(np.asarray(m.distances).shape[1]) for m in valid_models)

    # Keep threshold legend design consistent with the existing threshold-group plot style.
    # fig.legend(handles=threshold_handles, ...)

    fig.suptitle(
        _format_dataset_suptitle(valid_models, "Success Detection Rate by Threshold and Objective"),
        y=0.96,
        fontsize=16,
    )
    fig.supxlabel(f"Threshold ({units})", fontsize=13)
    fig.subplots_adjust(left=0.00, right=1.0, bottom=0.1, top=0.85, wspace=0.03)

    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_segmentation_metric_bars(
    models: List[ModelErrors],
    output_dir: str,
    dpi: int,
    filename: str = "segmentation_metrics_by_objective.png",
) -> str:
    """Plot saved test segmentation metrics by radius with SDR-style objective colors."""
    valid_models = [
        m
        for m in models
        if m.objective in OBJECTIVE_ORDER and m.radius_px is not None and m.segmentation_metrics
    ]
    if not valid_models:
        raise ValueError("No models with saved test segmentation metrics and radius metadata found.")

    objectives = [obj for obj in OBJECTIVE_ORDER if any(m.objective == obj for m in valid_models)]
    dataset_groups: Dict[str, List[ModelErrors]] = {}
    dataset_order: List[str] = []
    for model in valid_models:
        dataset = model.dataset if model.dataset else "Unknown"
        if dataset not in dataset_groups:
            dataset_groups[dataset] = []
            dataset_order.append(dataset)
        dataset_groups[dataset].append(model)

    n_datasets = len(dataset_order)
    fig_width = 10.0
    fig, axes = plt.subplots(1, n_datasets, figsize=(fig_width, 5), sharey=True, squeeze=False)
    axes = axes[0]

    group_spacing = 1.25
    objective_offsets = (
        np.linspace(-0.34, 0.34, num=len(objectives)) if len(objectives) > 1 else np.array([0.0])
    )
    objective_width = 0.8 / max(1, len(objectives))
    bar_width = objective_width * 0.9
    label_offset = 0.025

    def add_bar_label(ax, x_pos: float, value: float) -> None:
        if value < 0.05:
            y_pos = value + 0.04
        else:
            y_pos = value + 0.03
        ax.text(
            x_pos,
            y_pos,
            f".{round(100*value):02d}",
            ha="center",
            va="top",
            fontsize=6,
            color="black",
            clip_on=True,
            zorder=6,
        )

    for ds_idx, dataset in enumerate(dataset_order):
        ax = axes[ds_idx]
        ds_models = dataset_groups[dataset]
        radii = sorted({float(m.radius_px) for m in ds_models if m.radius_px is not None})
        x_base = np.arange(len(radii)) * group_spacing

        for obj_idx, objective in enumerate(objectives):
            color = OBJECTIVE_COLORS.get(objective, _model_color(obj_idx))
            for radius_idx, radius in enumerate(radii):
                matching = [
                    m
                    for m in ds_models
                    if m.objective == objective
                    and m.radius_px is not None
                    and float(m.radius_px) == radius
                    and m.segmentation_metrics
                ]
                if not matching:
                    continue

                detection_values = [
                    float(m.segmentation_metrics["detection"])
                    for m in matching
                    if m.segmentation_metrics and "detection" in m.segmentation_metrics
                ]
                structure_values = [
                    float(m.segmentation_metrics["structure"])
                    for m in matching
                    if m.segmentation_metrics and "structure" in m.segmentation_metrics
                ]

                objective_center = x_base[radius_idx] + objective_offsets[obj_idx]
                if structure_values:
                    structure_mean = float(np.mean(structure_values))
                    ax.bar(
                        objective_center,
                        structure_mean,
                        width=bar_width,
                        color=to_rgba(color, alpha=0.35),
                        edgecolor=to_rgba(BOX_OUTLINE_COLOR, alpha=0.35),
                        linewidth=0.7,
                    )
                    add_bar_label(ax, objective_center, structure_mean)
                if detection_values:
                    detection_mean = float(np.mean(detection_values))
                    ax.bar(
                        objective_center,
                        detection_mean,
                        width=bar_width,
                        color=to_rgba(color, alpha=0.6),
                        edgecolor=to_rgba(BOX_OUTLINE_COLOR, alpha=0.6),
                        linewidth=0.7,
                    )
                    add_bar_label(ax, objective_center, detection_mean)

        ax.set_title(dataset, fontsize=12)
        ax.set_xticks(x_base)
        ax.set_xticklabels([f"{int(r):d}" if float(r).is_integer() else f"{r:g}" for r in radii])
        if ds_idx == 0:
            ax.set_ylabel("Mean Dice", fontsize=12)
        else:
            ax.set_ylabel("")
        ax.set_ylim(0.0, 1.15)
        ax.grid(axis="y", alpha=0.25)
        metric_handles = [
            Patch(facecolor="#808080", edgecolor=BOX_OUTLINE_COLOR, alpha=0.8, label="Landmark"),
            Patch(facecolor="#808080", edgecolor=BOX_OUTLINE_COLOR, alpha=0.35, label="Structure"),
        ]
        ax.legend(
            handles=metric_handles,
            frameon=True,
            loc="upper left",
            fontsize=8,
            borderpad=0.25,
            labelspacing=0.35,
            handletextpad=0.5,
        )

    objective_handles = [
        Patch(
            facecolor=OBJECTIVE_COLORS.get(obj, _model_color(i)),
            edgecolor=BOX_OUTLINE_COLOR,
            alpha=0.6,
            label=OBJECTIVE_LABELS.get(obj, obj),
        )
        for i, obj in enumerate(objectives)
    ]

    fig.suptitle("Segmentation Performance by Radius and Objective",
        y=0.95,
        fontsize=14,
    )
    fig.supxlabel("Radius (px)", fontsize=12)
    objective_legend = fig.legend(
        handles=objective_handles,
        frameon=True,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.9),
        ncol=min(4, max(1, len(objectives))),
        fontsize=9,
        title_fontsize=10,
        borderpad=0.45,
        labelspacing=0.45,
        handletextpad=0.6,
        columnspacing=1.0,
    )
    fig.add_artist(objective_legend)
    fig.subplots_adjust(left=0.00, right=1.0, bottom=0.1, top=0.82, wspace=0.03)

    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_threshold_groups_by_radius_objective(
    models: List[ModelErrors],
    thresholds: List[float],
    output_dir: str,
    dpi: int,
    landmark_names: Optional[List[str]] = None,
    filename: str = "per_radius_threshold_groups.png",
) -> str:
    """Special threshold plot with one subplot per dataset.

    In each subplot:
    - x-axis: training radius (px)
    - colors: objective family (det/seg/conf/full)
    - y-values for boxplots: real/raw Euclidean errors (never threshold-clipped)
    - scatter markers: landmarks (values > t3 are visually grouped at top)
    """
    t1, t2, t3 = thresholds
    filled_markers = ["o", "s", "v", "P", "h", "<", ">"]
    outline_markers = ["x", "+", "*", "X"]


    valid_models = [
        m
        for m in models
        if m.objective in OBJECTIVE_ORDER and (m.radius_px is not None or m.objective == "nnlandmark")
    ]
    if not valid_models:
        raise ValueError(
            "No models with objective metadata found. Expected radius in train_overview model.radius "
            "for radius-based models, or nnLandmark in the label/filename for nnLandmark results."
        )

    objectives = [
        obj
        for obj in OBJECTIVE_ORDER
        if obj != "nnlandmark" and any(m.objective == obj for m in valid_models)
    ]
    dataset_groups = {}
    dataset_order: List[str] = []
    for m in valid_models:
        ds = m.dataset if m.dataset else "Unknown"
        if ds not in dataset_groups:
            dataset_groups[ds] = []
            dataset_order.append(ds)
        dataset_groups[ds].append(m)

    n_datasets = len(dataset_order)
    n_cols = 2
    fig_width = 10.0
    fig_height = 5.0
    fig = plt.figure(figsize=(fig_width, fig_height))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.15], wspace=0.03)
    fig_width = 10 #max(10, 4 * n_datasets)
    fig = plt.figure(figsize=(fig_width, 5.0))

    axes_pairs = []
    shared_y_ax = None
    for ds_idx in range(n_datasets):
        radius_ax = fig.add_subplot(grid[0, 2 * ds_idx], sharey=shared_y_ax)
        if shared_y_ax is None:
            shared_y_ax = radius_ax
        nn_ax = fig.add_subplot(grid[0, 2 * ds_idx + 1], sharey=shared_y_ax)
        axes_pairs.append((radius_ax, nn_ax))

    group_spacing = 1.25
    y_break_value = t3+0.5
    outside_scatter_y = t3 * 1.1
    outside_count_y = t3 * 1.13
    offsets = np.linspace(-0.34, 0.34, num=len(objectives)) if len(objectives) > 1 else np.array([0.0])
    objective_offset_by_name = {
        obj: float(offsets[idx]) for idx, obj in enumerate(objectives)
    }
    box_width = 0.6 / max(1, len(objectives))
    jitter_width = 0.07
    rng = np.random.default_rng(23)

    def _draw_y_axis_break(ax, break_y: float) -> None:
        slash_half_height = 0.0075 * t3
        slash_gap = 0.01
        for y_offset in (-slash_half_height * 1.8, slash_half_height * 1.8):
            ax.plot(
                [-slash_gap, slash_gap],
                [break_y + y_offset - slash_half_height, break_y + y_offset + slash_half_height],
                transform=ax.get_yaxis_transform(),
                color="black",
                linewidth=1.1,
                clip_on=False,
            )

    def _aggregate_landmark_values(
        ds_models: List[ModelErrors],
        objective: str,
        category_key: Optional[float],
    ) -> Optional[Dict[str, object]]:
        arrays = [
            np.asarray(m.distances, dtype=np.float64)
            for m in ds_models
            if m.objective == objective
            and (
                (category_key is None and m.objective == "nnlandmark")
                or (
                    category_key is not None
                    and m.radius_px is not None
                    and float(m.radius_px) == category_key
                )
            )
        ]
        if not arrays:
            return None
        n_landmarks = min(arr.shape[1] for arr in arrays)
        landmark_values = [
            np.concatenate([arr[:, lm_idx] for arr in arrays], axis=0)
            for lm_idx in range(n_landmarks)
        ]
        return {
            "all": np.concatenate(landmark_values, axis=0),
            "by_landmark": landmark_values,
        }

    def _draw_distribution(
        ax,
        values: np.ndarray,
        landmark_values: List[np.ndarray],
        pos: float,
        box_width: float,
        color: str,
    ) -> None:
        bp = ax.boxplot(
            [values],  # raw values only
            positions=[pos],
            widths=box_width,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
        )
        for box in bp["boxes"]:
            box.set_facecolor(to_rgba(color, alpha=0.2))
            box.set_edgecolor(to_rgba(BOX_OUTLINE_COLOR, alpha=0.4))
            box.set_linewidth(1.1)
        for whisker in bp["whiskers"]:
            whisker.set_color(BOX_OUTLINE_COLOR)
            whisker.set_linewidth(1.1)
        for cap in bp["caps"]:
            cap.set_color(BOX_OUTLINE_COLOR)
            cap.set_linewidth(1.1)
        for median in bp["medians"]:
            median.set_color(BOX_OUTLINE_COLOR)
            median.set_linewidth(1.3)

        if bp["boxes"]:
            box_vertices = bp["boxes"][0].get_path().vertices
            box_y_min = float(np.min(box_vertices[:, 1]))
            box_y_max = float(np.max(box_vertices[:, 1]))
        else:
            box_y_min, box_y_max = -np.inf, -np.inf
        if bp["whiskers"]:
            whisker_vals = np.concatenate([w.get_ydata() for w in bp["whiskers"]])
            whisk_y_min = float(np.min(whisker_vals))
            whisk_y_max = float(np.max(whisker_vals))
        else:
            whisk_y_min, whisk_y_max = -np.inf, -np.inf

        n_outside = 0
        for lm_idx, lm_values in enumerate(landmark_values):
            bins = np.digitize(lm_values, bins=[t1, t2, t3], right=True)
            for b in range(4):
                group_values = lm_values[bins == b]
                if group_values.size == 0:
                    continue
                x_scatter = np.full(group_values.size, pos, dtype=np.float64)
                x_scatter = x_scatter + rng.uniform(-jitter_width, jitter_width, size=group_values.size)
                y_scatter = (
                    np.full(group_values.size, outside_scatter_y, dtype=np.float64)
                    if b == 3
                    else group_values
                )
                marker = filled_markers[int(lm_idx/2) % len(filled_markers)] if b % 2 == 0 else outline_markers[int((lm_idx-1)/2) % len(outline_markers)]
                face_color = to_rgba(color, alpha=0.3) if b % 2 == 0 else to_rgba(color, alpha=1.0)
                edge_color = to_rgba(color, alpha=1.0) if b % 2 == 0 else None

                ax.scatter(
                    x_scatter,
                    y_scatter,
                    s=20,
                    alpha=0.55,
                    facecolors=face_color,
                    edgecolors=edge_color,
                    linewidths=1.0,
                    marker=marker,
                    zorder=4,
                )
                if b == 3:
                    n_outside += int(group_values.size)

        if n_outside:
            y_text = outside_count_y
            intersects_whiskers = whisk_y_min <= y_text <= whisk_y_max
            inside_box = box_y_min <= y_text <= box_y_max
            text_kwargs = {}
            if intersects_whiskers and not inside_box:
                text_kwargs["bbox"] = {
                    "boxstyle": "square,pad=0.1",
                    "facecolor": "white",
                    "edgecolor": "white",
                    "alpha": 1,
                }
            ax.text(
                pos,
                y_text,
                s=str(n_outside),
                fontsize=7,
                ha="center",
                va="bottom",
                zorder=6,
                clip_on=False,
                **text_kwargs,
            )

    def _style_threshold_axis(ax, show_ylabel: bool) -> None:
        for t in thresholds:
            ax.axhline(t, color="gray", linestyle="--", linewidth=1, alpha=0.6)
        ax.grid(axis="y", alpha=0.2)
        ax.yaxis.set_minor_locator(MultipleLocator(1.0))
        ax.tick_params(axis="x", which="minor", bottom=False, top=False)
        ax.grid(axis="y", which="minor", alpha=0.4, linestyle=":")
        ax.set_yticks([t1, t2, t3])
        if show_ylabel:
            ax.set_ylabel("Euclidean distance (px)", fontsize=12)
        ax.set_ylim(0, t3 * 1.19)

    for ds_idx, ds in enumerate(dataset_order):
        ax, nn_ax = axes_pairs[ds_idx]
        ds_models = dataset_groups[ds]
        radii = sorted({float(m.radius_px) for m in ds_models if m.radius_px is not None})
        has_nnlandmark = any(m.objective == "nnlandmark" for m in ds_models)
        x_positions = np.arange(len(radii), dtype=np.float64) * group_spacing

        for obj_idx, obj in enumerate(objectives):
            color = OBJECTIVE_COLORS.get(obj, _model_color(obj_idx))
            for radius_idx, radius in enumerate(radii):
                data = _aggregate_landmark_values(ds_models, obj, radius)
                if data is None:
                    continue
                values = data["all"]
                landmark_values = data["by_landmark"]
                if values.size == 0:
                    continue

                pos = x_positions[radius_idx] + objective_offset_by_name.get(obj, 0.0)
                _draw_distribution(ax, values, landmark_values, pos, box_width, color)

        if n_datasets > 1:
            ax.set_title(ds, fontsize=12)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f"{int(r):d}" if float(r).is_integer() else f"{r:g}" for r in radii])
        if x_positions.size:
            ax.set_xlim(float(x_positions[0]) - 0.55, float(x_positions[-1]) + 0.55)
        # _draw_y_axis_break(ax, y_break_value)
        _style_threshold_axis(ax, show_ylabel=ds_idx == 0)
        ax.set_xlabel("Radius (px)", fontsize=12)

        if has_nnlandmark:
            data = _aggregate_landmark_values(ds_models, "nnlandmark", None)
            if data is not None:
                values = data["all"]
                landmark_values = data["by_landmark"]
                if values.size:
                    _draw_distribution(nn_ax, values, landmark_values, 0.0, 0.235, "black")
            nn_ax.set_xlim(-0.55, 0.55)
            nn_ax.set_xticks([])
            nn_ax.set_title("nnLandmark", fontsize=12)
            _style_threshold_axis(nn_ax, show_ylabel=False)
            ax.spines["right"].set_visible(True)
        else:
            nn_ax.axis("off")

        nn_ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
        nn_ax.spines["left"].set_visible(True)
        nn_ax.set_ylabel("")

    objective_handles = [
        Patch(
            facecolor=OBJECTIVE_COLORS.get(obj, _model_color(i)),
            edgecolor=BOX_OUTLINE_COLOR,
            alpha=0.6,
            label=OBJECTIVE_LABELS.get(obj, obj),
        )
        for i, obj in enumerate(objectives)
    ]

    fig.suptitle(
        _format_dataset_suptitle(valid_models, "Aggregated Euclidean Errors by Radius and Objective"),
        y=0.95,
        fontsize=14,
    )
    # fig.supylabel("Euclidean distance (px)")
    objective_legend = fig.legend(
        handles=objective_handles,
        frameon=True,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.9),
        ncol=min(5, max(1, len(objectives))),
        fontsize=9,
        title_fontsize=10,
        borderpad=0.45,
        labelspacing=0.45,
        handletextpad=0.6,
        columnspacing=1.0,
    )
    fig.add_artist(objective_legend)

    fig.subplots_adjust(left=0.00, right=1.0, bottom=0.1, top=0.82, wspace=0.03)
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Euclidean landmark errors and create plots for one or multiple models.\n"
            "Use --model label=/path/to/train_overview.json (or any supported pred/gt/error file)."
        )
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help=(
            "Model input in format 'label=path'. If no label is provided, one is inferred. "
            "Path can be train_overview.json (.general.raw_results), json with pred/gt, "
            "nnLandmark-style detailed_results, or npz."
        ),
    )
    parser.add_argument(
        "--thresholds",
        nargs=3,
        type=float,
        default=[2.0, 5.0, 10.0],
        metavar=("T1", "T2", "T3"),
        help="Distance thresholds for grouping (default: 2 5 10). Must be increasing.",
    )
    parser.add_argument(
        "--landmark-names",
        type=str,
        default="",
        help="Optional comma-separated landmark names in index order.",
    )
    parser.add_argument("--output-dir", type=str, default="plots", help="Directory for output figures.")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI.")
    parser.add_argument(
        "--units",
        type=str,
        choices=["mm", "px"],
        default="px",
        help="Distance units for reported errors: 'mm' applies image resolution scaling, 'px' keeps voxel/pixel units.",
    )
    parser.add_argument(
        "--plot-radius-objective-trend",
        action="store_true",
        help=(
            "Generate the special aggregated threshold plot with objectives as colors and radius as x-axis "
            "(saved as per_radius_threshold_groups.png)."
        ),
    )
    parser.add_argument(
        "--radius-objective-plot-name",
        type=str,
        default="per_radius_threshold_groups.png",
        help="Filename for the special radius/objective aggregated threshold plot.",
    )
    parser.add_argument(
        "--plot-sdr-bars",
        action="store_true",
        help=(
            "Generate an SDR bar plot grouped by threshold, with base/seg/conf/full bars "
            "inside each threshold group."
        ),
    )
    parser.add_argument(
        "--sdr-bars-plot-name",
        type=str,
        default="sdr_bars_by_threshold.png",
        help="Filename for the SDR threshold bar plot.",
    )
    parser.add_argument(
        "--plot-stacked-landmark-sdr-bars",
        action="store_true",
        help=(
            "Generate an SDR bar plot grouped by threshold, with objectives next to each other "
            "and each bar stacked by per-landmark SDR contribution."
        ),
    )
    parser.add_argument(
        "--stacked-landmark-sdr-bars-plot-name",
        type=str,
        default="stacked_landmark_sdr_bars_by_threshold.png",
        help="Filename for the stacked per-landmark SDR threshold bar plot.",
    )
    parser.add_argument(
        "--plot-segmentation-bars",
        action="store_true",
        help=(
            "Generate a segmentation-performance bar plot with objective-colored bars. "
            "The first saved test metric is detection/landmark segmentation; the second, "
            "when present, is structure segmentation."
        ),
    )
    parser.add_argument(
        "--segmentation-bars-plot-name",
        type=str,
        default="segmentation_metrics_by_objective.png",
        help="Filename for the segmentation-performance bar plot.",
    )
    parser.add_argument(
        "--plot-sample-landmark-bars",
        action="store_true",
        help=(
            "Generate one figure per landmark with one subplot per sample and per-model bars "
            "(useful to spot samples that are good for some models and bad for others)."
        ),
    )
    parser.add_argument(
        "--sample-landmark-bars-prefix",
        type=str,
        default="sample_bar_comparison",
        help="Filename prefix for per-landmark sample comparison barplot figures.",
    )
    parser.add_argument(
        "--write-stats",
        action="store_true",
        help=(
            "Write stats_summary.tsv and stats_tests.tsv with SDR, median/IQR, catastrophic "
            "failure rate, and paired tests."
        ),
    )
    parser.add_argument(
        "--write-latex-stats",
        action="store_true",
        help="Also write LaTeX table fragments for the statistical outputs.",
    )
    parser.add_argument(
        "--failure-threshold",
        type=float,
        default=None,
        help=(
            "Threshold above which a point is counted as a catastrophic failure. "
            "Defaults to the largest value passed via --thresholds."
        ),
    )
    parser.add_argument(
        "--stat-reference",
        type=str,
        default="base",
        help=(
            "Reference for paired tests. Matched first by model label, then by objective "
            "(default: base)."
        ),
    )
    parser.add_argument(
        "--case-aggregate",
        type=str,
        choices=["mean", "median"],
        default="mean",
        help="Aggregate landmark errors per case before paired statistical tests (default: mean).",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip all matplotlib figure generation.",
    )

    args = parser.parse_args()

    thresholds = list(args.thresholds)
    if not (thresholds[0] < thresholds[1] < thresholds[2]):
        raise ValueError(f"Thresholds must be strictly increasing, got {thresholds}.")
    failure_threshold = args.failure_threshold if args.failure_threshold is not None else thresholds[-1]

    os.makedirs(args.output_dir, exist_ok=True)

    models: List[ModelErrors] = []
    for model_arg in args.model:
        label, path = _parse_model_arg(model_arg)
        distances = _extract_detailed_result_distances(path)
        if distances is None:
            pred, gt = _extract_pred_gt(path, units=args.units)
            distances = _compute_distances(pred, gt)
        objective, radius_px, dataset = _extract_model_metadata(path, label=label)
        segmentation_metrics = _extract_segmentation_metrics(path)
        models.append(
            ModelErrors(
                label=label,
                path=path,
                distances=distances,
                objective=objective,
                radius_px=radius_px,
                dataset=dataset,
                segmentation_metrics=segmentation_metrics,
            )
        )
    landmark_counts = [int(np.asarray(m.distances).shape[1]) for m in models]
    unique_counts = sorted(set(landmark_counts))
    same_landmark_count = len(unique_counts) == 1

    landmark_names = None
    if args.landmark_names.strip():
        landmark_names = [name.strip() for name in args.landmark_names.split(",")]
        if same_landmark_count:
            expected = unique_counts[0]
            if len(landmark_names) != expected:
                raise ValueError(
                    f"Expected {expected} landmark names, got {len(landmark_names)}."
                )
        elif len(landmark_names) not in unique_counts:
            raise ValueError(
                f"Models have different landmark counts {unique_counts}. "
                f"landmark-names length must match one of these counts, got {len(landmark_names)}."
            )

    _print_stats(models)
    if args.write_stats or args.write_latex_stats:
        written, metric_rows, test_rows = _write_statistics_outputs(
            models=models,
            thresholds=thresholds,
            failure_threshold=failure_threshold,
            output_dir=args.output_dir,
            reference=args.stat_reference,
            case_aggregate=args.case_aggregate,
            write_latex=args.write_latex_stats,
        )
        print(f"\nSaved statistical tables in: {os.path.abspath(args.output_dir)}")
        for path in written:
            print(f" - {os.path.basename(path)}")
        if not test_rows:
            print(
                "No paired test rows were written. Check that each dataset/radius group contains "
                f"a reference matching '{args.stat_reference}'."
            )

    if HAS_MPL and not args.skip_plots:
        generated_plots: List[str] = []
        _plot_boxplot(models, args.output_dir, dpi=args.dpi)
        generated_plots.append("boxplot_overall.png")
        _plot_per_model_landmark_boxplots(models, args.output_dir, landmark_names, dpi=args.dpi)
        for model in models:
            generated_plots.append(f"boxplot_per_landmark_{_safe_filename(model.label)}.png")

        if same_landmark_count:
            _plot_grouped_landmark_model_boxplots(models, args.output_dir, landmark_names, dpi=args.dpi)
            _plot_median_iqr(models, args.output_dir, landmark_names, dpi=args.dpi)
            _plot_threshold_groups(models, thresholds, args.output_dir, landmark_names, dpi=args.dpi)
            generated_plots.extend(
                [
                    "boxplot_grouped_landmark_model.png",
                    "per_landmark_median_iqr.png",
                    "per_landmark_threshold_groups.png",
                ]
            )
        else:
            counts_msg = ", ".join(
                f"{model.label}={count}" for model, count in zip(models, landmark_counts)
            )
            print(
                "\nSkipping comparative landmark-aligned plots because models have different landmark counts:"
            )
            print(f" {counts_msg}")
            print(
                " Skipped: boxplot_grouped_landmark_model.png, per_landmark_median_iqr.png, "
                "per_landmark_threshold_groups.png"
            )

        if args.plot_sample_landmark_bars:
            generated_plots.extend(
                _plot_sample_bar_comparison_per_landmark(
                    models=models,
                    output_dir=args.output_dir,
                    landmark_names=landmark_names,
                    dpi=args.dpi,
                    filename_prefix=args.sample_landmark_bars_prefix,
                )
            )

        if args.plot_radius_objective_trend:
            try:
                _plot_threshold_groups_by_radius_objective(
                    models=models,
                    thresholds=thresholds,
                    output_dir=args.output_dir,
                    dpi=args.dpi,
                    landmark_names=landmark_names,
                    filename=args.radius_objective_plot_name,
                )
                generated_plots.append(args.radius_objective_plot_name)
            except ValueError as exc:
                print(f"\nSkipping radius/objective trend plot: {exc}")

        if args.plot_sdr_bars:
            try:
                _plot_sdr_bars_by_threshold(
                    models=models,
                    thresholds=thresholds,
                    output_dir=args.output_dir,
                    dpi=args.dpi,
                    units=args.units,
                    filename=args.sdr_bars_plot_name,
                )
                generated_plots.append(args.sdr_bars_plot_name)
            except ValueError as exc:
                print(f"\nSkipping SDR bar plot: {exc}")

        if args.plot_stacked_landmark_sdr_bars:
            try:
                _plot_stacked_landmark_sdr_bars_by_threshold(
                    models=models,
                    thresholds=thresholds,
                    output_dir=args.output_dir,
                    dpi=args.dpi,
                    units=args.units,
                    landmark_names=landmark_names,
                    filename=args.stacked_landmark_sdr_bars_plot_name,
                )
                generated_plots.append(args.stacked_landmark_sdr_bars_plot_name)
            except ValueError as exc:
                print(f"\nSkipping stacked landmark SDR bar plot: {exc}")

        if args.plot_segmentation_bars:
            try:
                _plot_segmentation_metric_bars(
                    models=models,
                    output_dir=args.output_dir,
                    dpi=args.dpi,
                    filename=args.segmentation_bars_plot_name,
                )
                generated_plots.append(args.segmentation_bars_plot_name)
            except ValueError as exc:
                print(f"\nSkipping segmentation performance bar plot: {exc}")

        print(f"\nSaved plots in: {os.path.abspath(args.output_dir)}")
        for plot_name in generated_plots:
            print(f" - {plot_name}")
    elif args.skip_plots:
        print("\nSkipped figure generation because --skip-plots was set.")
    else:
        print("\nmatplotlib not available: skipped figure generation, statistics were still computed.")


if __name__ == "__main__":
    main()
