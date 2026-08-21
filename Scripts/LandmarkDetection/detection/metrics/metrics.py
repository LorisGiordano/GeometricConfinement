from collections.abc import Sequence
from monai.utils import MetricReduction
from monai.metrics.regression import RegressionMetric
import torch

class IoUMetric(RegressionMetric):

    def __init__(self, corners: bool = False, reduction: MetricReduction | str = MetricReduction.MEAN, get_not_nans: bool = False) -> None:
        super().__init__(reduction=reduction, get_not_nans=get_not_nans)
        self.corners = corners

    def _compute_metric(self, outputs_bbox: torch.Tensor, labels_bbox: torch.Tensor) -> torch.Tensor:
        
        metrics = []
        for out_bb, lab_bb in zip(outputs_bbox, labels_bbox):

            # center to corners
            corner1_output, corner2_output = out_bb[0]
            corner1_label, corner2_label = lab_bb[0]
            if not self.corners:
                center_output, size_output = out_bb[0]
                center_label, size_label = lab_bb[0]
                corner1_output = center_output - size_output/2
                corner2_output = corner1_output + size_output
                corner1_label = center_label - size_label/2
                corner2_label = corner1_label + size_label
            corner1_output = torch.clamp(corner1_output, min=0)

            sum_volume = torch.prod(corner2_output-corner1_output) + torch.prod(corner2_label-corner1_label)
            
            # Calculate intersection volume
            intersection_width = min(corner2_output[0], corner2_label[0]) - max(corner1_output[0], corner1_label[0])
            intersection_height = min(corner2_output[1], corner2_label[1]) - max(corner1_output[1], corner1_label[1])
            intersection_depth = min(corner2_output[2], corner2_label[2]) - max(corner1_output[2], corner1_label[2])
            
            intersection_volume = intersection_width * intersection_height * intersection_depth
            if intersection_width <= 0 or intersection_height <= 0 or intersection_depth <= 0:
                intersection_volume = 0

            # Calculate union volume
            union_volume = sum_volume - intersection_volume

            # Calculate IoU
            iou = intersection_volume / union_volume
            
            metrics.append(torch.stack([iou]))
        
        return torch.stack(metrics)
    
class EuclideanDistanceMetric(RegressionMetric):

    def __init__(self, per_landmark: bool = False, reduction: MetricReduction | str = MetricReduction.MEAN, get_not_nans: bool = False) -> None:
        super().__init__(reduction=reduction, get_not_nans=get_not_nans)
        self.per_landmark = per_landmark

    def _compute_metric(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        
        if outputs.shape != labels.shape:
            raise Exception(f"Input tensors have different shape. Got {outputs.shape} for prediction and {labels.shape} for label.")
        
        if outputs.ndim == 3:
            euclidean_distance = torch.linalg.norm(outputs - labels, dim=2)
            if not self.per_landmark:
                euclidean_distance = euclidean_distance.mean(dim=1, keepdim=True)
        elif outputs.ndim == 2:
            euclidean_distance = torch.linalg.norm(outputs - labels, dim=1)
            if not self.per_landmark:
                euclidean_distance = euclidean_distance.mean(dim=0, keepdim=True)
        else:
            raise ValueError(f"Unsupported tensor shape: {outputs.shape}")

        return euclidean_distance