""" Dataset-level statistics for geometric confinement.

For every landmark-structure pair of a confinement config and every landmark radius, computes over the
training split, with phi the signed distance to the structure boundary (negative inside, positive outside,
in voxels):
  - band:          quantiles of phi over all ground-truth landmark sphere voxels, pooled over the training set
                   (admissible band of SurfDistConLoss; its upper bound is used by VolDistConLoss),
  - overlap:       the mean relative overlap of Eq. 9 (reported for reference, not used by the distance losses).

All are dataset-level quantities, so they are not affected by local annotation noise in single samples.
They are written to a "statistics" block of the confinement config, which is read by the distance-based
confinement (training config: "GeoConType": "distance"):

    "statistics": {"3": {"0-1": {"band": [-2.4, 3.1], "overlap": 0.64, ...}, ...}, "5": {...}}

The signed distance is computed with the same function and method as in the loss, so the band matches what
the loss sees during training.

Usage:
    python confinement_statistics.py \
        --dataset-directory absolute/path/to/Dataset101_CTPel_r20 \
        --confinement-config confinement_config \
        --radii 3 5 7 10 \
        --write
"""

""" IMPORTS """

import os
import json
import argparse

import numpy as np
import torch
from monai.transforms import Compose, LoadImaged, EnsureTyped, Orientationd

from detection.transforms.label import LoadPointsd
from detection.losses.losses import signed_distance_map, radius_key, _surface_volume_confinement
from detection.data.data import add_root


""" HELPERS """

def _pairs(config: dict) -> list[tuple[str, int, int]]:
    """ All (type, landmark_id, segment_id) pairs defined in a confinement config. """
    surface, volume, _ = _surface_volume_confinement(config)
    pairs = []
    for kind, confined in (("volume", volume), ("surface", surface)):
        for landmark_id, segment_ids in enumerate(confined):
            if segment_ids[0] is None:
                continue
            for segment_id in segment_ids:
                pairs.append((kind, landmark_id, int(segment_id)))
    return pairs


def _sphere_indices(center: np.ndarray, radius: float, shape: tuple[int, ...]) -> tuple[np.ndarray, ...]:
    """ Indices of the voxels of a sphere mask, identical to GenerateSphereMaskd, restricted to the field of view. """
    lower = np.maximum(np.floor(center - radius).astype(int), 0)
    upper = np.minimum(np.ceil(center + radius).astype(int) + 1, shape)
    if np.any(upper <= lower):
        return tuple(np.zeros(0, dtype=int) for _ in shape)
    grid = np.meshgrid(*[np.arange(l, u) for l, u in zip(lower, upper)], indexing="ij")
    grid = np.stack(grid, axis=0).astype(np.float32)
    dist = np.sqrt(np.sum((grid - center.reshape(-1, *([1] * len(shape)))) ** 2, axis=0))
    inside = dist <= radius
    return tuple(g[inside].astype(int) for g in grid)


""" STATISTICS """

def confinement_statistics(dataset_directory: str, config: dict, radii: list[float],
                           quantiles: tuple[float, float] = (0.025, 0.975),
                           datalist: str = "datalist_trainval.json",
                           max_distance: int = 32, sdf_method: str = "pool") -> dict:

    with open(os.path.join(dataset_directory, datalist), "r") as json_file:
        datalist_json = json.load(json_file)
    train_data = add_root(datalist_json["training"], datalist_json["root"])

    pairs = _pairs(config)
    segment_ids = sorted({s for _, _, s in pairs})
    n_landmarks = config["meta"]["n_landmarks"]

    if max(radii) + 1 >= max_distance:
        print(f"WARNING: max_distance {max_distance} is close to the largest radius {max(radii)}; "
              f"signed distances are clamped at ±{max_distance}.")

    transforms = Compose([
        LoadImaged(keys=["label_seg"], ensure_channel_first=True),
        EnsureTyped(keys=["label_seg"]),
        Orientationd(keys=["label_seg"], axcodes="LPS", lazy=False),
        LoadPointsd(keys="label_lm", channels=list(range(n_landmarks))),
    ])

    overlaps = {(r, p): [] for r in radii for p in pairs}
    distances = {(r, p): [] for r in radii for p in pairs}

    for n, item in enumerate(train_data):
        print(f" Processing sample {n+1}/{len(train_data)}...       ", end="\r")
        data = transforms({"label_seg": item["label_seg"], "label_lm": item["label_lm"]})
        label = torch.as_tensor(data["label_seg"])[0]
        points = data["label_lm"].float().numpy()
        shape = tuple(label.shape)

        masks, sdfs = {}, {}
        for segment_id in segment_ids:
            mask = (label == segment_id).float()
            masks[segment_id] = mask.numpy() > 0.5
            sdfs[segment_id] = signed_distance_map(mask[None, None], max_distance, sdf_method, dim=mask.ndim)[0, 0].numpy()

        for kind, landmark_id, segment_id in pairs:
            center = points[landmark_id]
            if int(center[0]) == -1:
                continue
            for r in radii:
                idx = _sphere_indices(center, r, shape)
                if idx[0].size == 0:
                    continue
                phi = sdfs[segment_id][idx]
                overlaps[(r, (kind, landmark_id, segment_id))].append(float(masks[segment_id][idx].mean()))
                distances[(r, (kind, landmark_id, segment_id))].append(phi)
    print()

    statistics = {}
    for r in radii:
        block = {}
        for p in pairs:
            kind, landmark_id, segment_id = p
            values = overlaps[(r, p)]
            if not values:
                print(f"WARNING: no valid samples for pair {landmark_id}-{segment_id} at radius {r}.")
                continue
            phi = np.concatenate(distances[(r, p)])
            block[f"{landmark_id}-{segment_id}"] = {
                "type": kind,
                "band": [round(float(np.quantile(phi, quantiles[0])), 3),
                         round(float(np.quantile(phi, quantiles[1])), 3)],
                "overlap": round(float(np.mean(values)), 4),
                "n": len(values),
            }
        statistics[radius_key(r)] = block

    return statistics


def show_statistics(statistics: dict) -> None:
    print(f"\n{'radius':>6} {'pair':>8} {'type':>8} {'band [vox]':>16} {'overlap':>8} {'n':>4}")
    for r, block in statistics.items():
        for pair, s in block.items():
            band = f"[{s['band'][0]:.1f}, {s['band'][1]:.1f}]"
            print(f"{r:>6} {pair:>8} {s['type']:>8} {band:>16} {s['overlap']:>8.2f} {s['n']:>4}")


""" RUN """

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Compute dataset-level signed-distance targets for distance-based geometric confinement.")
    parser.add_argument("--dataset-directory", required=True, help="Absolute path to the dataset (contains the datalist and the confinement config).")
    parser.add_argument("--confinement-config", default="confinement_config", help="Name of the confinement config, without '.json'.")
    parser.add_argument("--datalist", default="datalist_trainval.json", help="Datalist whose 'training' split is used.")
    parser.add_argument("--radii", type=float, nargs="+", required=True, help="Landmark radii in voxels.")
    parser.add_argument("--quantiles", type=float, nargs=2, default=[0.025, 0.975], help="Lower and upper quantile of the signed-distance band.")
    parser.add_argument("--max-distance", type=int, default=32, help="Clamp of the signed distance (voxels); must match the loss.")
    parser.add_argument("--sdf-method", default="pool", choices=["pool", "edt"], help="Signed distance method; must match the loss.")
    parser.add_argument("--write", action="store_true", help="Write the statistics into the confinement config.")
    args = parser.parse_args()

    config_filepath = os.path.join(args.dataset_directory, args.confinement_config + ".json")
    with open(config_filepath, "r") as json_file:
        config = json.load(json_file)

    statistics = confinement_statistics(args.dataset_directory, config, args.radii, tuple(args.quantiles),
                                        args.datalist, args.max_distance, args.sdf_method)
    show_statistics(statistics)

    if args.write:
        config.setdefault("statistics", {}).update(statistics)
        config["statistics_meta"] = {"split": "training", "datalist": args.datalist,
                                     "quantiles": list(args.quantiles),
                                     "max_distance": args.max_distance, "sdf_method": args.sdf_method}
        with open(config_filepath, "w") as json_file:
            json.dump(config, json_file, indent=4)
        print(f"\nStatistics written to {config_filepath}")
