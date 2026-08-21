import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from scipy import stats as scipy_stats


OBJECTIVE_ORDER = ["base", "seg", "conf", "full", "nnlandmark"]


@dataclass
class ModelResult:
    label: str
    distances: List[List[float]]
    objective: Optional[str]
    radius_px: Optional[float]
    dataset: Optional[str]


def _parse_model_arg(model_arg: str) -> Tuple[str, str]:
    if "=" in model_arg:
        label, path = model_arg.split("=", 1)
        return label.strip(), path.strip()
    path = model_arg.strip()
    return os.path.basename(path), path


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
    for token in re.split(r"[^a-z0-9]+", text.lower()):
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
        return "AorticRoot"
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
        return "AorticRoot"
    if "ctpels" in low:
        return "CTPelS"
    if "ctpel" in low:
        return "CTPel"
    if "hiea" in low:
        return "HIEA"
    return None


def _extract_metadata(path: str, label: str) -> Tuple[Optional[str], Optional[float], Optional[str]]:
    data = _load_json(path)
    model_info = data.get("model", {}) if isinstance(data.get("model", {}), dict) else {}
    training_info = data.get("training", {}) if isinstance(data.get("training", {}), dict) else {}
    general_info = data.get("general", {}) if isinstance(data.get("general", {}), dict) else {}

    metadata_text = f"{label} {os.path.basename(path)}"
    objective = _infer_objective_from_text(metadata_text)
    radius_px = _to_float_or_none(model_info.get("radius"))
    dataset = _normalize_dataset_name(general_info.get("dataset_name")) or _infer_dataset_from_text(metadata_text)

    if "detailed_results" in data:
        objective = _normalize_objective_name(objective) or "nnlandmark"
        if objective == "nnlandmark":
            radius_px = None
        return objective, radius_px, _normalize_dataset_name(dataset)

    has_seg = False
    if model_info:
        out_channels_seg = model_info.get("out_channels_seg")
        if isinstance(out_channels_seg, list):
            has_seg = any(isinstance(v, (int, float)) and v >= 0 for v in out_channels_seg)

    conf_info = training_info.get("geometric_confinement")
    has_conf = bool(conf_info) if not isinstance(conf_info, (dict, list)) else len(conf_info) > 0
    inferred = "full" if has_seg and has_conf else "seg" if has_seg else "conf" if has_conf else "base"
    objective = _normalize_objective_name(objective) if objective else inferred
    if objective == "nnlandmark":
        radius_px = None
    return objective, radius_px, _normalize_dataset_name(dataset)


def _scale_point(point: Sequence[float], resolution: Sequence[float], units: str) -> List[float]:
    if units == "px":
        return [float(v) for v in point]
    return [float(v) * float(resolution[i]) for i, v in enumerate(point)]


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def _landmark_sort_key(name: str) -> Tuple[int, object]:
    match = re.search(r"(\d+)$", str(name))
    if match:
        return 0, int(match.group(1))
    return 1, str(name)


def _extract_detailed_result_distances(data: object, path: str) -> Optional[List[List[float]]]:
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

    distances: List[List[float]] = []
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
        distances.append(row)

    return distances


def _extract_distances(path: str, units: str) -> List[List[float]]:
    data = _load_json(path)
    detailed_distances = _extract_detailed_result_distances(data, path)
    if detailed_distances is not None:
        return detailed_distances

    pred, gt = data["general"]["raw_results"]
    resolution = data["model"]["image_resolution"]
    distances: List[List[float]] = []
    for sample_pred, sample_gt in zip(pred, gt):
        sample_distances = []
        for point_pred, point_gt in zip(sample_pred, sample_gt):
            p = _scale_point(point_pred, resolution, units)
            g = _scale_point(point_gt, resolution, units)
            sample_distances.append(_distance(p, g))
        distances.append(sample_distances)
    return distances


def _percentile(values: Sequence[float], percentile: float) -> float:
    sorted_values = sorted(float(v) for v in values)
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * percentile / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _median_iqr(values: Sequence[float]) -> Tuple[float, float, float, float]:
    median = _percentile(values, 50)
    q1 = _percentile(values, 25)
    q3 = _percentile(values, 75)
    return median, q1, q3, q3 - q1


def _flatten(values: Sequence[Sequence[float]]) -> List[float]:
    return [float(v) for row in values for v in row]


def _format_threshold_name(threshold: float) -> str:
    return f"{threshold:g}".replace(".", "p")


def _aggregate_cases(distances: Sequence[Sequence[float]], method: str) -> List[float]:
    aggregated = []
    for row in distances:
        if method == "mean":
            aggregated.append(sum(row) / len(row))
        elif method == "median":
            aggregated.append(_percentile(row, 50))
        else:
            raise ValueError(f"Unsupported case aggregation: {method}")
    return aggregated


def _model_group_key(model: ModelResult) -> Tuple[str, str]:
    dataset = model.dataset or "Unknown"
    if model.radius_px is None:
        radius = "unknown"
    elif float(model.radius_px).is_integer():
        radius = str(int(model.radius_px))
    else:
        radius = f"{model.radius_px:g}"
    return dataset, radius


def _is_nnlandmark(model: ModelResult) -> bool:
    return model.objective == "nnlandmark"


def _pairwise_groups(models: Sequence[ModelResult]) -> Dict[Tuple[str, str], List[ModelResult]]:
    groups: Dict[Tuple[str, str], List[ModelResult]] = {}
    nnlandmark_by_dataset: Dict[str, List[ModelResult]] = {}

    for model in models:
        dataset = model.dataset or "Unknown"
        if _is_nnlandmark(model):
            nnlandmark_by_dataset.setdefault(dataset, []).append(model)
        else:
            groups.setdefault(_model_group_key(model), []).append(model)

    for (dataset, _radius), group in list(groups.items()):
        for nnlandmark_model in nnlandmark_by_dataset.get(dataset, []):
            if nnlandmark_model not in group:
                group.append(nnlandmark_model)

    # Preserve nnLandmark-only inputs instead of silently dropping them.
    for dataset, nnlandmark_models in nnlandmark_by_dataset.items():
        if not any(group_dataset == dataset for group_dataset, _radius in groups):
            groups[(dataset, "unknown")] = list(nnlandmark_models)

    return groups


def _metric_rows(
    models: Sequence[ModelResult],
    thresholds: Sequence[float],
    failure_threshold: float,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for model in models:
        values = _flatten(model.distances)
        median, q1, q3, iqr = _median_iqr(values)
        row: Dict[str, object] = {
            "dataset": model.dataset or "Unknown",
            "radius_px": "" if model.radius_px is None else f"{model.radius_px:g}",
            "objective": model.objective or "",
            "model": model.label,
            "n_cases": len(model.distances),
            "n_landmarks": len(model.distances[0]) if model.distances else 0,
            "n_points": len(values),
            "median": median,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "catastrophic_failure_rate": sum(v > failure_threshold for v in values) / len(values),
            "catastrophic_failures": sum(v > failure_threshold for v in values),
        }
        for threshold in thresholds:
            row[f"sdr_le_{_format_threshold_name(threshold)}"] = sum(v <= threshold for v in values) / len(values)
        rows.append(row)
    return rows


def _binom_two_sided_pvalue(successes: int, trials: int) -> float:
    if trials == 0:
        return float("nan")
    observed = min(successes, trials - successes)
    p_value = 0.0
    for k in range(observed + 1):
        p_value += math.comb(trials, k) * (0.5 ** trials)
    return min(1.0, 2.0 * p_value)


def _paired_sign_flip_pvalue(diff: Sequence[float], max_exact_n: int = 20) -> Tuple[float, str]:
    nonzero = [float(v) for v in diff if abs(float(v)) > 0]
    n = len(nonzero)
    if n == 0:
        return float("nan"), "all_ties"
    observed = abs(sum(nonzero) / n)
    if n > max_exact_n:
        # Deterministic approximation without third-party dependencies.
        state = 23
        extreme = 0
        permutations = 100000
        abs_diff = [abs(v) for v in nonzero]
        for _ in range(permutations):
            signed_sum = 0.0
            for value in abs_diff:
                state = (1103515245 * state + 12345) & 0x7FFFFFFF
                signed_sum += value if state & 1 else -value
            if abs(signed_sum / n) >= observed - 1e-12:
                extreme += 1
        return (extreme + 1) / (permutations + 1), f"monte_carlo_sign_flip_{permutations}"

    total = 1 << n
    extreme = 0
    abs_diff = [abs(v) for v in nonzero]
    for mask in range(total):
        signed_sum = 0.0
        for bit, value in enumerate(abs_diff):
            signed_sum += -value if (mask >> bit) & 1 else value
        if abs(signed_sum / n) >= observed - 1e-12:
            extreme += 1
    return extreme / total, "exact_sign_flip"


def _select_reference(group: Sequence[ModelResult], reference: str) -> Optional[ModelResult]:
    ref_norm = _normalize_objective_name(reference)
    for model in group:
        if model.label == reference:
            return model
    for model in group:
        if model.objective == ref_norm:
            return model
    return None


def _test_rows(
    models: Sequence[ModelResult],
    reference: str,
    case_aggregate: str,
) -> List[Dict[str, object]]:
    groups = _pairwise_groups(models)

    rows: List[Dict[str, object]] = []
    for (dataset, radius), group in groups.items():
        ref = _select_reference(group, reference)
        if ref is None:
            continue
        ref_cases = _aggregate_cases(ref.distances, case_aggregate)
        for model in group:
            if model is ref:
                continue
            cases = _aggregate_cases(model.distances, case_aggregate)
            if len(cases) != len(ref_cases):
                rows.append(
                    {
                        "dataset": dataset,
                        "radius_px": radius,
                        "reference_model": ref.label,
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
            diff = [case - ref_case for case, ref_case in zip(cases, ref_cases)]
            nonzero = [v for v in diff if abs(v) > 0]
            wins = sum(v < 0 for v in nonzero)
            sign_p = _binom_two_sided_pvalue(wins, len(nonzero))
            perm_p, perm_method = _paired_sign_flip_pvalue(diff)
            rows.append(
                {
                    "dataset": dataset,
                    "radius_px": radius,
                    "reference_model": ref.label,
                    "model": model.label,
                    "case_aggregate": case_aggregate,
                    "n_cases": len(diff),
                    "median_delta": _percentile(diff, 50),
                    "mean_delta": sum(diff) / len(diff),
                    "wins": wins,
                    "non_ties": len(nonzero),
                    "sign_test_p": sign_p,
                    "permutation_p": perm_p,
                    "permutation_method": perm_method,
                }
            )
    return rows


def _flatten_errors_by_model(model: ModelResult) -> List[float]:
    return _flatten(model.distances)


def _failure_indicators_by_model(model: ModelResult, failure_threshold: float) -> List[int]:
    return [1 if value > failure_threshold else 0 for value in _flatten_errors_by_model(model)]


def _pairwise_distribution_test_rows(
    models: Sequence[ModelResult],
    alpha: float,
    test_mode: str,
) -> List[Dict[str, object]]:
    groups = _pairwise_groups(models)

    rows: List[Dict[str, object]] = []
    for (dataset, radius), group in groups.items():
        group_sorted = sorted(
            group,
            key=lambda model: (
                OBJECTIVE_ORDER.index(model.objective)
                if model.objective in OBJECTIVE_ORDER
                else len(OBJECTIVE_ORDER),
                model.label,
            ),
        )
        for idx_a in range(len(group_sorted)):
            for idx_b in range(idx_a + 1, len(group_sorted)):
                model_a = group_sorted[idx_a]
                model_b = group_sorted[idx_b]
                errors_a = _flatten_errors_by_model(model_a)
                errors_b = _flatten_errors_by_model(model_b)
                median_a, _q1_a, _q3_a, iqr_a = _median_iqr(errors_a)
                median_b, _q1_b, _q3_b, iqr_b = _median_iqr(errors_b)
                base_row: Dict[str, object] = {
                    "dataset": dataset,
                    "radius_px": radius,
                    "model_a": model_a.label,
                    "objective_a": model_a.objective or "",
                    "model_b": model_b.label,
                    "objective_b": model_b.objective or "",
                    "n_paired_errors": "",
                    "median_error_a": median_a,
                    "iqr_error_a": iqr_a,
                    "median_error_b": median_b,
                    "iqr_error_b": iqr_b,
                    "median_delta_b_minus_a": "",
                    "mean_delta_b_minus_a": "",
                    "shapiro_w": "",
                    "shapiro_p": "",
                    "normal_delta_at_alpha": "",
                    "pairwise_test_mode": test_mode,
                    "recommended_test": "",
                    "test_statistic": "",
                    "test_p": "",
                    "wilcoxon_usable": "",
                }
                if len(errors_a) != len(errors_b):
                    test = scipy_stats.mannwhitneyu(errors_b, errors_a, alternative="two-sided")
                    base_row.update(
                        {
                            "n_paired_errors": f"{len(errors_a)}/{len(errors_b)}",
                            "recommended_test": "mann_whitney_u_unpaired",
                            "test_statistic": float(test.statistic),
                            "test_p": float(test.pvalue),
                            "notes": "unpaired_length_mismatch",
                        }
                    )
                    rows.append(base_row)
                    continue

                deltas = [b - a for a, b in zip(errors_a, errors_b)]
                nonzero_deltas = [delta for delta in deltas if abs(delta) > 0]
                base_row.update(
                    {
                        "n_paired_errors": len(deltas),
                        "median_delta_b_minus_a": _percentile(deltas, 50),
                        "mean_delta_b_minus_a": sum(deltas) / len(deltas),
                    }
                )

                if len(deltas) < 3:
                    base_row["notes"] = "skipped_shapiro_requires_at_least_3_pairs"
                    rows.append(base_row)
                    continue

                shapiro = scipy_stats.shapiro(deltas)
                shapiro_p = float(shapiro.pvalue)
                normal_delta = bool(shapiro_p >= alpha)
                base_row.update(
                    {
                        "shapiro_w": float(shapiro.statistic),
                        "shapiro_p": shapiro_p,
                        "normal_delta_at_alpha": normal_delta,
                    }
                )

                if test_mode == "force-ttest":
                    test = scipy_stats.ttest_rel(errors_b, errors_a, nan_policy="omit")
                    base_row.update(
                        {
                            "recommended_test": "paired_t_test",
                            "test_statistic": float(test.statistic),
                            "test_p": float(test.pvalue),
                            "forced": True,
                        }
                    )
                elif len(nonzero_deltas) == 0:
                    base_row.update(
                        {
                            "recommended_test": "none",
                            "test_statistic": "",
                            "test_p": "",
                        }
                    )
                elif test_mode == "force-wilcoxon":
                    test = scipy_stats.wilcoxon(errors_b, errors_a, zero_method="wilcox", alternative="two-sided")
                    base_row.update(
                        {
                            "recommended_test": "wilcoxon_signed_rank",
                            "test_statistic": float(test.statistic),
                            "test_p": float(test.pvalue),
                            "forced": True,
                        }
                    )
                elif normal_delta:
                    test = scipy_stats.ttest_rel(errors_b, errors_a, nan_policy="omit")
                    base_row.update(
                        {
                            "recommended_test": "paired_t_test",
                            "test_statistic": float(test.statistic),
                            "test_p": float(test.pvalue),
                            "forced": False,
                        }
                    )
                else:
                    test = scipy_stats.wilcoxon(errors_b, errors_a, zero_method="wilcox", alternative="two-sided")
                    base_row.update(
                        {
                            "recommended_test": "wilcoxon_signed_rank",
                            "test_statistic": float(test.statistic),
                            "test_p": float(test.pvalue),
                            "forced": False,
                        }
                    )
                rows.append(base_row)
    return rows


def _pairwise_catastrophic_failure_test_rows(
    models: Sequence[ModelResult],
    failure_threshold: float,
    alpha: float,
) -> List[Dict[str, object]]:
    groups = _pairwise_groups(models)

    rows: List[Dict[str, object]] = []
    for (dataset, radius), group in groups.items():
        group_sorted = sorted(
            group,
            key=lambda model: (
                OBJECTIVE_ORDER.index(model.objective)
                if model.objective in OBJECTIVE_ORDER
                else len(OBJECTIVE_ORDER),
                model.label,
            ),
        )
        for idx_a in range(len(group_sorted)):
            for idx_b in range(idx_a + 1, len(group_sorted)):
                model_a = group_sorted[idx_a]
                model_b = group_sorted[idx_b]
                failures_a = _failure_indicators_by_model(model_a, failure_threshold)
                failures_b = _failure_indicators_by_model(model_b, failure_threshold)
                base_row: Dict[str, object] = {
                    "dataset": dataset,
                    "radius_px": radius,
                    "model_a": model_a.label,
                    "objective_a": model_a.objective or "",
                    "model_b": model_b.label,
                    "objective_b": model_b.objective or "",
                    "failure_threshold": failure_threshold,
                    "n_paired_points": "",
                    "failure_rate_a": "",
                    "failure_rate_b": "",
                    "failure_rate_delta_b_minus_a": "",
                    "shapiro_w": "",
                    "shapiro_p": "",
                    "normal_delta_at_alpha": "",
                    "test_mode": "force-wilcoxon",
                    "test": "",
                    "test_statistic": "",
                    "test_p": "",
                    "significant_at_alpha": "",
                    "notes": "",
                }
                if len(failures_a) != len(failures_b):
                    base_row["notes"] = "skipped_length_mismatch"
                    rows.append(base_row)
                    continue

                deltas = [b - a for a, b in zip(failures_a, failures_b)]
                nonzero_deltas = [delta for delta in deltas if delta != 0]
                n_pairs = len(deltas)
                rate_a = sum(failures_a) / n_pairs if n_pairs else float("nan")
                rate_b = sum(failures_b) / n_pairs if n_pairs else float("nan")
                base_row.update(
                    {
                        "n_paired_points": n_pairs,
                        "failure_rate_a": rate_a,
                        "failure_rate_b": rate_b,
                        "failure_rate_delta_b_minus_a": rate_b - rate_a,
                    }
                )

                if n_pairs < 3:
                    base_row["notes"] = "skipped_shapiro_requires_at_least_3_pairs"
                    rows.append(base_row)
                    continue

                shapiro = scipy_stats.shapiro(deltas)
                shapiro_p = float(shapiro.pvalue)
                normal_delta = bool(shapiro_p >= alpha)
                base_row.update(
                    {
                        "shapiro_w": float(shapiro.statistic),
                        "shapiro_p": shapiro_p,
                        "normal_delta_at_alpha": normal_delta,
                    }
                )

                if not nonzero_deltas:
                    base_row.update(
                        {
                            "test": "none",
                            "notes": "all_failure_indicators_tied",
                        }
                    )
                    rows.append(base_row)
                    continue

                test = scipy_stats.wilcoxon(
                    failures_b,
                    failures_a,
                    zero_method="wilcox",
                    alternative="two-sided",
                )
                p_value = float(test.pvalue)
                base_row.update(
                    {
                        "test": "wilcoxon_signed_rank",
                        "test_statistic": float(test.statistic),
                        "test_p": p_value,
                        "significant_at_alpha": bool(p_value < alpha),
                    }
                )
                rows.append(base_row)
    return rows


def _pairwise_decision_rule(test_mode: str) -> str:
    if test_mode == "force-wilcoxon":
        return "Always use Wilcoxon signed-rank; Shapiro-Wilk is reported but not used for test selection."
    if test_mode == "force-ttest":
        return "Always use paired t-test; Shapiro-Wilk is reported but not used for test selection."
    return (
        "Use paired t-test when Shapiro-Wilk p >= alpha; "
        "use Wilcoxon signed-rank when Shapiro-Wilk p < alpha."
    )

def _format_value(value: object) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.4f}"
    return str(value)


def _write_latex(path: str, rows: Sequence[Dict[str, object]], caption: str, label: str) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w") as f:
        f.write("\\begin{table}[h]\n\\centering\n\\small\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write(f"\\label{{{label}}}\n")
        f.write("\\begin{tabular}{" + "l" * len(fields) + "}\n")
        f.write("\\hline\n")
        f.write(" & ".join(field.replace("_", "\\_") for field in fields) + " \\\\\n")
        f.write("\\hline\n")
        for row in rows:
            f.write(" & ".join(_format_value(row.get(field, "")) for field in fields) + " \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")


def _latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def _format_median_iqr(median: object, iqr: object) -> str:
    if not isinstance(median, (int, float)) or not isinstance(iqr, (int, float)):
        return ""
    if math.isnan(float(median)) or math.isnan(float(iqr)):
        return ""
    return f"{float(median):.2f} ({float(iqr):.2f})"


def _format_p_value(value: object) -> str:
    if not isinstance(value, (int, float)) or math.isnan(float(value)):
        return ""
    value = float(value)
    if value < 0.001:
        return "$<0.001$"
    return f"{value:.3f}"


def _format_rate(value: object) -> str:
    if not isinstance(value, (int, float)) or math.isnan(float(value)):
        return ""
    return f"{100.0 * float(value):.1f}\\%"


def _write_pairwise_compact_latex(path: str, rows: Sequence[Dict[str, object]], caption: str, label: str) -> None:
    if not rows:
        return
    dataset_counts: Dict[str, int] = {}
    radius_counts: Dict[Tuple[str, str], int] = {}
    for row in rows:
        dataset = str(row.get("dataset", ""))
        radius = str(row.get("radius_px", ""))
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
        radius_counts[(dataset, radius)] = radius_counts.get((dataset, radius), 0) + 1

    seen_datasets = set()
    seen_radii = set()
    previous_dataset = None
    previous_radius = None

    with open(path, "w") as f:
        f.write("\\begin{table}[h]\n\\centering\n\\small\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write(f"\\label{{{label}}}\n")
        f.write("\\begin{tabular}{llllll}\n")
        f.write("\\hline\n")
        f.write("Dataset & Radius & Comparison & Model A & Model B & $p$-value \\\\\n")
        f.write("\\hline\n")
        for row in rows:
            dataset_raw = str(row.get("dataset", ""))
            radius_raw = str(row.get("radius_px", ""))

            if previous_dataset is not None and dataset_raw != previous_dataset:
                f.write("\\hline\n")
            elif previous_radius is not None and (dataset_raw, radius_raw) != previous_radius:
                f.write("\\cline{2-6}\n")

            if dataset_raw not in seen_datasets:
                dataset = f"\\multirow{{{dataset_counts[dataset_raw]}}}{{*}}{{{_latex_escape(dataset_raw)}}}"
                seen_datasets.add(dataset_raw)
            else:
                dataset = ""

            radius_key = (dataset_raw, radius_raw)
            if radius_key not in seen_radii:
                radius = f"\\multirow{{{radius_counts[radius_key]}}}{{*}}{{{_latex_escape(radius_raw)}}}"
                seen_radii.add(radius_key)
            else:
                radius = ""

            comparison = _latex_escape(f"{row.get('model_a', '')} vs {row.get('model_b', '')}")
            model_a = _format_median_iqr(row.get("median_error_a", ""), row.get("iqr_error_a", ""))
            model_b = _format_median_iqr(row.get("median_error_b", ""), row.get("iqr_error_b", ""))
            p_value = _format_p_value(row.get("test_p", ""))
            f.write(f"{dataset} & {radius} & {comparison} & {model_a} & {model_b} & {p_value} \\\\\n")

            previous_dataset = dataset_raw
            previous_radius = radius_key
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")


def _write_pairwise_failure_latex(path: str, rows: Sequence[Dict[str, object]], caption: str, label: str) -> None:
    if not rows:
        return
    dataset_counts: Dict[str, int] = {}
    radius_counts: Dict[Tuple[str, str], int] = {}
    for row in rows:
        dataset = str(row.get("dataset", ""))
        radius = str(row.get("radius_px", ""))
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
        radius_counts[(dataset, radius)] = radius_counts.get((dataset, radius), 0) + 1

    seen_datasets = set()
    seen_radii = set()
    previous_dataset = None
    previous_radius = None

    with open(path, "w") as f:
        f.write("\\begin{table}[h]\n\\centering\n\\small\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write(f"\\label{{{label}}}\n")
        f.write("\\begin{tabular}{llllllll}\n")
        f.write("\\hline\n")
        f.write("Dataset & Radius & Comparison & Model A CFR & Model B CFR & Shapiro $p$ & Wilcoxon $p$ & Significant \\\\\n")
        f.write("\\hline\n")
        for row in rows:
            dataset_raw = str(row.get("dataset", ""))
            radius_raw = str(row.get("radius_px", ""))

            if previous_dataset is not None and dataset_raw != previous_dataset:
                f.write("\\hline\n")
            elif previous_radius is not None and (dataset_raw, radius_raw) != previous_radius:
                f.write("\\cline{2-8}\n")

            if dataset_raw not in seen_datasets:
                dataset = f"\\multirow{{{dataset_counts[dataset_raw]}}}{{*}}{{{_latex_escape(dataset_raw)}}}"
                seen_datasets.add(dataset_raw)
            else:
                dataset = ""

            radius_key = (dataset_raw, radius_raw)
            if radius_key not in seen_radii:
                radius = f"\\multirow{{{radius_counts[radius_key]}}}{{*}}{{{_latex_escape(radius_raw)}}}"
                seen_radii.add(radius_key)
            else:
                radius = ""

            comparison = _latex_escape(f"{row.get('model_a', '')} vs {row.get('model_b', '')}")
            shapiro_p = _format_p_value(row.get("shapiro_p", ""))
            wilcoxon_p = _format_p_value(row.get("test_p", ""))
            significant = row.get("significant_at_alpha", "")
            significant_text = "Yes" if significant is True else "No" if significant is False else ""
            f.write(
                f"{dataset} & {radius} & {comparison} & "
                f"{_format_rate(row.get('failure_rate_a', ''))} & "
                f"{_format_rate(row.get('failure_rate_b', ''))} & "
                f"{shapiro_p} & {wilcoxon_p} & {significant_text} \\\\\n"
            )

            previous_dataset = dataset_raw
            previous_radius = radius_key
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")


def _json_safe(value):
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return value
    if isinstance(value, dict):
        return {key: _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _write_json(path: str, payload: object) -> None:
    with open(path, "w") as f:
        json.dump(_json_safe(payload), f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write statistical landmark-localization tables.")
    parser.add_argument("--model", action="append", required=True, help="Model input as label=path.")
    parser.add_argument("--output-dir", default="Stats", help="Output directory.")
    parser.add_argument("--units", choices=["px", "mm"], default="mm", help="Distance units.")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[2.0, 5.0, 10.0], help="SDR thresholds.")
    parser.add_argument("--failure-threshold", type=float, default=None, help="Catastrophic failure threshold.")
    parser.add_argument("--stat-reference", default="base", help="Reference model label or objective.")
    parser.add_argument("--case-aggregate", choices=["mean", "median"], default="mean", help="Per-case aggregation.")
    parser.add_argument("--write-latex", action="store_true", help="Write LaTeX table fragments too.")
    parser.add_argument(
        "--write-pairwise-distribution-tests",
        action="store_true",
        help=(
            "Write pairwise method comparisons per dataset/radius. Shapiro-Wilk is applied to "
            "paired error deltas; paired t-test is used for normal deltas and Wilcoxon signed-rank otherwise."
        ),
    )
    parser.add_argument(
        "--normality-alpha",
        type=float,
        default=0.05,
        help="Alpha level for Shapiro-Wilk normality decisions (default: 0.05).",
    )
    parser.add_argument(
        "--pairwise-test-mode",
        choices=["auto", "force-wilcoxon", "force-ttest"],
        default="auto",
        help=(
            "Pairwise distribution test selection. 'auto' uses Shapiro-Wilk to choose paired t-test "
            "or Wilcoxon; forced modes ignore Shapiro for selection but still report it."
        ),
    )
    parser.add_argument(
        "--write-pairwise-failure-tests",
        action="store_true",
        help=(
            "Write pairwise catastrophic failure-rate comparisons per dataset/radius. "
            "Shapiro-Wilk is reported for paired failure-indicator deltas; Wilcoxon signed-rank "
            "is always used for the significance test."
        ),
    )
    args = parser.parse_args()

    if len(args.thresholds) < 1 or args.thresholds != sorted(args.thresholds):
        raise ValueError(f"Thresholds must be sorted in increasing order, got {args.thresholds}.")
    failure_threshold = args.failure_threshold if args.failure_threshold is not None else args.thresholds[-1]

    os.makedirs(args.output_dir, exist_ok=True)
    models: List[ModelResult] = []
    for model_arg in args.model:
        label, path = _parse_model_arg(model_arg)
        objective, radius_px, dataset = _extract_metadata(path, label)
        distances = _extract_distances(path, units=args.units)
        models.append(ModelResult(label, distances, objective, radius_px, dataset))

    pairwise_rows = (
        _pairwise_distribution_test_rows(models, args.normality_alpha, args.pairwise_test_mode)
        if args.write_pairwise_distribution_tests
        else []
    )
    failure_rows = (
        _pairwise_catastrophic_failure_test_rows(models, failure_threshold, args.normality_alpha)
        if args.write_pairwise_failure_tests
        else []
    )

    if args.write_pairwise_distribution_tests:
        _write_pairwise_compact_latex(
            os.path.join(args.output_dir, "stats_pairwise_distribution_tests.tex"),
            pairwise_rows,
            "Pairwise method comparisons. Values are median (IQR) localization error.",
            "tab:stats_pairwise_distribution_tests",
        )

    if args.write_pairwise_failure_tests:
        _write_pairwise_failure_latex(
            os.path.join(args.output_dir, "stats_pairwise_catastrophic_failure_tests.tex"),
            failure_rows,
            (
                "Pairwise catastrophic failure-rate comparisons. Shapiro-Wilk is reported for "
                "paired failure-indicator deltas; Wilcoxon signed-rank is forced for significance."
            ),
            "tab:stats_pairwise_catastrophic_failure_tests",
        )
        _write_json(
            os.path.join(args.output_dir, "stats_pairwise_catastrophic_failure_tests.json"),
            {
                "failure_threshold": failure_threshold,
                "normality_alpha": args.normality_alpha,
                "decision_rule": (
                    "Shapiro-Wilk is reported as a diagnostic; Wilcoxon signed-rank is forced "
                    "for all non-tied catastrophic failure indicator comparisons."
                ),
                "rows": failure_rows,
            },
        )


if __name__ == "__main__":
    main()
