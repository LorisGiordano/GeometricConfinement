""" IMPORTS """

from typing import Optional, Union
from torch.nn.modules.loss import _Loss
from monai.losses import DiceCELoss, DiceFocalLoss, DiceLoss
from monai.utils import LossReduction
import torch
import torch.nn as nn
import torch.nn.functional as F


""" LandmarkDetectionLoss """

class SigmoidMSELoss(nn.Module):
    def __init__(self, reduction: Optional[str] = 'mean'):
        super(SigmoidMSELoss, self).__init__()
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        loss = F.mse_loss(inputs, targets, reduction='none')

        # reduce loss
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        
        return loss
    

def _surface_volume_confinement(config, channels: list[int] = [], shift_with_n_landmarks: bool = False) -> tuple[list, list, int]:
    n_landmarks = config["meta"]["n_landmarks"]
    surface_confined = [[None],]*n_landmarks
    volume_confined = [[None],]*n_landmarks

    if "surface" in config.keys():
        for cc in config["surface"]:
            landmark_id = int(cc["landmark"])
            segment_id = cc["segment"]

            if not isinstance(segment_id, list):
                segment_id = [int(segment_id)]

            if shift_with_n_landmarks:
                segment_id = [int(s + n_landmarks) for s in segment_id]

            surface_confined[landmark_id] = segment_id
    
    if "volume" in config.keys():
        for cc in config["volume"]:
            landmark_id = int(cc["landmark"])
            segment_id = cc["segment"]

            if not isinstance(segment_id, list):
                segment_id = [segment_id]

            if shift_with_n_landmarks:
                segment_id = [int(s + n_landmarks) for s in segment_id]

            volume_confined[landmark_id] = segment_id

    n_landmarks = len(channels)
    if channels:
        surface_confined = [surface_confined[i] for i in channels]
        volume_confined = [volume_confined[i] for i in channels]

    return surface_confined, volume_confined, n_landmarks

class LandmarkDetectionLoss(_Loss):

    def __init__(
            self, 
            detection_loss: str = "DiceCELoss", 
            GeoCon: dict = {}, 
            channels: list[int] = [], 
            dim: int = 3,
            reduction: Union[str, LossReduction] = LossReduction.MEAN) -> None:
        
        super().__init__()
        
        if detection_loss == "DiceCELoss":
            self.detection_loss = DiceCELoss(sigmoid=True, softmax=False, squared_pred=False, reduction=reduction)
        elif detection_loss == "DiceFocalLoss":
            self.detection_loss = DiceFocalLoss(sigmoid=True, reduction=reduction)
        elif detection_loss == "DiceLoss":
            self.detection_loss = DiceLoss(sigmoid=True, reduction=reduction)
        elif detection_loss == "MSELoss":
            self.detection_loss = torch.nn.MSELoss(reduction=reduction) # SigmoidMSELoss()
        elif detection_loss == "L1Loss":
            self.detection_loss = torch.nn.L1Loss(reduction=reduction)
        else:
            raise Exception(f"Non implemented detection loss function '{detection_loss}'.")
        
        self.n_losses = 1
        
        self.geocon_active = False
        if GeoCon:
            self.geocon_active = True
            self.surfcon_loss = SurfConLoss(dim=dim, reduction=reduction)
            self.volcon_loss = VolConLoss(dim=dim, reduction=reduction)
            self.surface_confined, self.volume_confined, self.n_lm = _surface_volume_confinement(GeoCon, channels)
            self.n_losses += 1

    def forward(self, det_outputs: torch.Tensor, det_labels: torch.Tensor, seg_labels: Optional[torch.Tensor] = None):

        loss = self.detection_loss(det_outputs, det_labels)
        
        if self.geocon_active:
            if seg_labels is None:
                raise Exception("Segmentation labels are necessary to compute geometric confinement.")
            conf_loss = 0
            n_conf = 0
            for i, segment_ids in enumerate(self.volume_confined):
                if segment_ids[0] is None:
                    continue
                det_output = det_outputs[:,i,...].unsqueeze(1)
                for segment_id in segment_ids:
                    n_conf += 1
                    conf_loss += self.volcon_loss(det_output, seg_labels[:, segment_id-1, ...].unsqueeze(1))
            for i, segment_ids in enumerate(self.surface_confined):
                if segment_ids[0] is None:
                    continue
                det_output = det_outputs[:,i,...].unsqueeze(1)
                for segment_id in segment_ids:
                    n_conf += 1
                    conf_loss += self.surfcon_loss(det_output, seg_labels[:, segment_id-1, ...].unsqueeze(1))
            conf_loss = conf_loss / n_conf if n_conf > 0 else 0
            loss += conf_loss
                
        return loss / self.n_losses
            
        
class SurfConLoss(_Loss):
    def __init__(self, sigmoid: bool = True, 
                 alpha: float = 0.0, beta: int = 2, reduction: Union[str, LossReduction] = LossReduction.MEAN,
                 segmentation_based: bool = True, dim: int = 3) -> None:
        super().__init__()
        self.sigmoid = sigmoid
        self.reduction = reduction
        self.alpha = alpha
        self.beta = beta
        self.segmentation_based = segmentation_based
        self.dim = dim

    def _derive_segmentation_from_images(self, labels: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        
        seg_labels = torch.zeros_like(labels)
        for b, (image,label) in enumerate(zip(images, labels)):
            image = image.squeeze()
            for i, label in enumerate(labels):
                local_image = label * image
                threshold = torch.median(local_image)
                seg_labels[b,i] = (image > threshold).float()

        return seg_labels

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor, images: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        if not self.segmentation_based:
            if not images:
                raise Exception("Images must be provided for intensity-based Surface Confinement loss.")
            
            batch_images = images
            if images.ndim == self.dim + 1:
                batch_images = batch_images.unsqueeze(0)
            labels = self._derive_segmentation_from_images(labels, batch_images)
            if images.ndim == self.dim + 1:
                labels = labels.squeeze(0)


        if self.sigmoid:
            outputs = outputs + self.alpha
            outputs = torch.sigmoid(outputs)

        if outputs.ndim == self.dim + 1:
            reduce_axis = torch.arange(1, len(labels.shape)).tolist()
        else:
            reduce_axis = torch.arange(2, len(labels.shape)).tolist()
        inner = torch.sum(outputs * labels, dim=reduce_axis)
        inner /= torch.sum(outputs, dim=reduce_axis)


        # TODO: for irregular surfaces, calculate the mean overlap between structure and landmark 
        # during preprocessing and then use that value instead of 1/2
        loss = 1 - torch.sqrt(4 * inner * (1 - inner)) ** self.beta

        if self.reduction == LossReduction.MEAN:
            loss = torch.mean(loss)
        elif self.reduction == LossReduction.SUM:
            loss = torch.sum(loss)

        return loss


class VolConLoss(_Loss):
    def __init__(self, sigmoid: bool = True, alpha: float = 0.0, beta: int = 2,
                 reduction: Union[str, LossReduction] = LossReduction.MEAN,
                 segmentation_based: bool = True,
                 dim: int = 3) -> None:
        super().__init__()
        self.sigmoid = sigmoid
        self.reduction = reduction
        self.alpha = alpha
        self.beta = beta
        self.segmentation_based = segmentation_based
        self.dim = dim

    def _derive_segmentation_from_images(self, labels: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        
        seg_labels = torch.zeros_like(labels)
        for b, (image,label) in enumerate(zip(images, labels)):
            image = image.squeeze()
            for i, label in enumerate(labels):
                local_image = label * image
                threshold1 = torch.min(local_image)
                threshold2 = torch.max(local_image)
                local_image = local_image[local_image > threshold1]
                local_image = local_image[local_image < threshold2]
                seg_labels[b,i] = ((local_image >= threshold1) & (local_image <= threshold2)).float()

        return seg_labels

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor, images: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        if not self.segmentation_based:
            if not images:
                raise Exception("Images must be provided for intensity-based Surface Confinement loss.")
            
            batch_images = images
            if images.ndim == self.dim + 1:
                batch_images = batch_images.unsqueeze(0)
            labels = self._derive_segmentation_from_images(labels, batch_images)
            if images.ndim == self.dim + 1:
                labels = labels.squeeze(0)


        if self.sigmoid:
            outputs = outputs + self.alpha
            outputs = torch.sigmoid(outputs)

        if outputs.ndim == self.dim + 1:
            reduce_axis = torch.arange(1, len(labels.shape)).tolist()
        else:
            reduce_axis = torch.arange(2, len(labels.shape)).tolist()
        inner = torch.sum(outputs * labels, dim=reduce_axis)
        inner /= torch.sum(outputs, dim=reduce_axis)

        loss = 1 - inner ** self.beta

        if self.reduction == LossReduction.MEAN:
            loss = torch.mean(loss)
        elif self.reduction == LossReduction.SUM:
            loss = torch.sum(loss)

        return loss
    

""" TASK PRIORITIZATION """

class TaskPrioritization(_Loss):
    
    def __init__(self, update_steps: int = 5, alpha: float = 0.5, gravitational_constant: Optional[float] = None, verbose: bool = False) -> None:

        super().__init__()
        
        self.alpha = alpha
        self.update_steps = update_steps
        self.gravitational_constant = gravitational_constant
        self.verbose = verbose
        
        self.difficulty_segmentation = 1/2
        self.difficulty_detection = 1/2
        
        self._current_step = 1
        self._previous_segmentation_loss = 0
        self._previous_detection_loss = 0
        
        self.priority_history = [(1/2, 1/2)]
        
    def initialize(self, segmentation_loss, detection_loss):
        self._previous_segmentation_loss = segmentation_loss
        self._previous_detection_loss = detection_loss
        
    def forward(self, segmentation_loss, detection_loss):
        
        if (self._current_step % self.update_steps) == 0:
            
            with torch.no_grad():
                if self.gravitational_constant:
                    _potential_energy_density_segmentation = self.gravitational_constant * segmentation_loss
                    _kinetic_energy_density_segmentation = (((segmentation_loss - self._previous_segmentation_loss)/self.update_steps)**2)/2
                    _energy_density_segmentation = _potential_energy_density_segmentation + _kinetic_energy_density_segmentation
                    
                    _potential_energy_density_detection = self.gravitational_constant * detection_loss
                    _kinetic_energy_density_detection = (((detection_loss - self._previous_detection_loss)/self.update_steps)**2)/2
                    _energy_density_detection = _potential_energy_density_detection + _kinetic_energy_density_detection
                    
                    _difficulty_segmentation = _energy_density_segmentation
                    _difficulty_detection = _energy_density_detection
                    
                else:
                    _difficulty_segmentation = torch.exp((segmentation_loss - self._previous_segmentation_loss)/self._previous_segmentation_loss)
                    _difficulty_detection = torch.exp((detection_loss - self._previous_detection_loss)/self._previous_detection_loss)

                self.difficulty_segmentation = (1-self.alpha) * self.difficulty_segmentation + self.alpha * _difficulty_segmentation
                self.difficulty_detection = (1-self.alpha) * self.difficulty_detection + self.alpha * _difficulty_detection

                norm = self.difficulty_segmentation + self.difficulty_detection
                self.difficulty_segmentation /= norm
                self.difficulty_detection /= norm
                
                self.priority_history.append((self.difficulty_segmentation.item(), self.difficulty_detection.item()))
                
                if self.verbose:
                    print(f"Segmentation: {self.difficulty_segmentation}, Detection: {self.difficulty_detection}")
                
        
        task_prioritization_loss = self.difficulty_segmentation * segmentation_loss + self.difficulty_detection * detection_loss
        
        self._current_step += 1
        self._previous_segmentation_loss = segmentation_loss
        self._previous_detection_loss = detection_loss
        
        return task_prioritization_loss
    
    def history(self):
        
        return self.priority_history