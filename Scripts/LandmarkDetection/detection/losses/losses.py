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
            reduction: Union[str, LossReduction] = LossReduction.MEAN,
            confinement: str = "region",
            radius: Optional[float] = None,
            confinement_params: dict = {}) -> None:
        """
        confinement: "region" for the overlap-based surface/volume confinement (SurfConLoss/VolConLoss),
                     "distance" for the distance-based surface/volume confinement (SurfDistConLoss/VolDistConLoss).
        radius: landmark sphere radius in voxels, used to select dataset-level statistics and default targets.
        confinement_params: keyword arguments for the distance-based losses: tau, max_distance, sdf_method, alpha.
        """

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
        self.confinement = confinement
        if GeoCon:
            self.geocon_active = True
            self.surface_confined, self.volume_confined, self.n_lm = _surface_volume_confinement(GeoCon, channels)
            self.n_losses += 1

            if confinement == "region":
                self.surfcon_loss = SurfConLoss(dim=dim, reduction=reduction)
                self.volcon_loss = VolConLoss(dim=dim, reduction=reduction)

            elif confinement == "distance":
                params = dict(confinement_params)
                if "tau" not in params:
                    params["tau"] = float(radius) if radius is not None else 3.0
                self.surfdistcon_loss = SurfDistConLoss(dim=dim, reduction=reduction, **params)
                self.voldistcon_loss = VolDistConLoss(dim=dim, reduction=reduction, **params)
                self.dim = dim

                # dataset-level distance targets per (type, output channel, segment)
                landmark_ids = channels if channels else list(range(len(self.surface_confined)))
                self.confinement_targets = {}
                missing = []
                for kind, confined in (("volume", self.volume_confined), ("surface", self.surface_confined)):
                    for i, segment_ids in enumerate(confined):
                        if segment_ids[0] is None:
                            continue
                        for segment_id in segment_ids:
                            targets, from_stats = _distance_targets(GeoCon, kind, landmark_ids[i], int(segment_id), radius)
                            self.confinement_targets[(kind, i, int(segment_id))] = targets
                            if not from_stats:
                                missing.append(f"{landmark_ids[i]}-{segment_id}")
                if missing:
                    print(f"WARNING: No dataset statistics for radius {radius} and pairs {missing}; "
                          f"using idealized targets (surface: band [-r, r]; volume: fully inside). "
                          f"Run confinement_statistics.py to compute them.")

                stats_meta = GeoCon.get("statistics_meta", {})
                if stats_meta and stats_meta.get("sdf_method", self.voldistcon_loss.sdf_method) != self.voldistcon_loss.sdf_method:
                    print(f"WARNING: Statistics were computed with sdf_method '{stats_meta['sdf_method']}' "
                          f"but the loss uses '{self.voldistcon_loss.sdf_method}'.")

            else:
                raise Exception(f"Non implemented confinement '{confinement}'.")

    def _distance_confinement(self, det_outputs: torch.Tensor, seg_labels: torch.Tensor) -> tuple[torch.Tensor, int]:
        conf_loss = 0
        n_conf = 0
        sdf_cache = {}
        for kind, confined in (("volume", self.volume_confined), ("surface", self.surface_confined)):
            for i, segment_ids in enumerate(confined):
                if segment_ids[0] is None:
                    continue
                det_output = det_outputs[:, i, ...].unsqueeze(1)
                for segment_id in segment_ids:
                    segment_id = int(segment_id)
                    seg_label = seg_labels[:, segment_id-1, ...].unsqueeze(1)
                    if segment_id not in sdf_cache:
                        sdf_cache[segment_id] = signed_distance_map(seg_label, self.voldistcon_loss.max_distance,
                                                                    self.voldistcon_loss.sdf_method, self.dim)
                    targets = self.confinement_targets[(kind, i, segment_id)]
                    if kind == "volume":
                        conf_loss += self.voldistcon_loss(det_output, seg_label, targets["hi"], sdf=sdf_cache[segment_id])
                    else:
                        conf_loss += self.surfdistcon_loss(det_output, seg_label, targets["band"], sdf=sdf_cache[segment_id])
                    n_conf += 1
        return conf_loss, n_conf

    def forward(self, det_outputs: torch.Tensor, det_labels: torch.Tensor, seg_labels: Optional[torch.Tensor] = None):

        loss = self.detection_loss(det_outputs, det_labels)

        if self.geocon_active and self.confinement == "distance":
            if seg_labels is None:
                raise Exception("Segmentation labels are necessary to compute geometric confinement.")
            conf_loss, n_conf = self._distance_confinement(det_outputs, seg_labels)
            conf_loss = conf_loss / n_conf if n_conf > 0 else 0
            loss += conf_loss

        elif self.geocon_active:
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


""" DISTANCE-AWARE CONFINEMENT """

def _dilate_cross(x: torch.Tensor, dim: int = 3) -> torch.Tensor:
    """ Binary dilation with the face-connected neighbourhood (6-connectivity in 3D, 4 in 2D). """
    pool = F.max_pool3d if dim == 3 else F.max_pool2d
    out = x
    for axis in range(dim):
        kernel = [1] * dim
        padding = [0] * dim
        kernel[axis] = 3
        padding[axis] = 1
        out = torch.maximum(out, pool(x, kernel_size=kernel, stride=1, padding=padding))
    return out


def _dilate_cube(x: torch.Tensor, dim: int = 3) -> torch.Tensor:
    """ Binary dilation with the full neighbourhood (26-connectivity in 3D, 8 in 2D). """
    pool = F.max_pool3d if dim == 3 else F.max_pool2d
    return pool(x, kernel_size=3, stride=1, padding=1)


def _distance_steps(masks: torch.Tensor, max_distance: int, dim: int = 3) -> torch.Tensor:
    """ Number of dilation steps needed to reach each voxel from the mask (0 inside the mask, capped at max_distance).
    Alternating face- and full-neighbourhood dilations gives an octagonal approximation of the Euclidean distance. """
    covered = masks
    steps = torch.zeros_like(masks)
    for k in range(int(max_distance)):
        steps += 1 - covered
        covered = _dilate_cross(covered, dim) if k % 2 == 0 else _dilate_cube(covered, dim)
    return steps


@torch.no_grad()
def signed_distance_map(masks: torch.Tensor, max_distance: int = 32, method: str = "pool", dim: int = 3) -> torch.Tensor:
    """ Signed distance (in voxels) to the boundary of binary masks of shape (B, C, *spatial).
    Negative inside, positive outside, zero on the interface between inside and outside voxels,
    clamped to [-max_distance, max_distance].

    method="pool": GPU-friendly octagonal approximation by iterative dilation (default, used during training).
    method="edt": exact Euclidean distance with scipy on CPU (slower, useful as a reference). """

    masks = (masks > 0.5).float()

    if method == "edt":
        import numpy as np
        from scipy.ndimage import distance_transform_edt

        m = masks.detach().cpu().numpy().astype(bool)
        sdf = np.zeros(m.shape, dtype=np.float32)
        for idx in np.ndindex(*m.shape[:2]):
            fg = m[idx]
            if not fg.any():
                sdf[idx] = max_distance
                continue
            if fg.all():
                sdf[idx] = -max_distance
                continue
            d_out = distance_transform_edt(~fg)
            d_in = distance_transform_edt(fg)
            sdf[idx] = np.where(fg, -(d_in - 0.5), d_out - 0.5)
        sdf = torch.from_numpy(sdf).to(masks.device)

    elif method == "pool":
        d_out = _distance_steps(masks, max_distance, dim)
        d_in = _distance_steps(1 - masks, max_distance, dim)
        sdf = torch.where(masks > 0.5, -(d_in - 0.5), d_out - 0.5)

    else:
        raise Exception(f"Non implemented signed distance method '{method}'.")

    return sdf.clamp(-max_distance, max_distance)


class _DistanceConfinement(_Loss):
    """ Shared machinery of the distance-based confinement losses: landmark probabilities, signed distance
    to the structure boundary, and probability-weighted expectations. """

    def __init__(self, sigmoid: bool = True, alpha: float = 0.0, tau: float = 3.0,
                 max_distance: int = 32, sdf_method: str = "pool",
                 reduction: Union[str, LossReduction] = LossReduction.MEAN,
                 dim: int = 3, eps: float = 1e-6) -> None:
        super().__init__()
        self.sigmoid = sigmoid
        self.alpha = alpha
        self.tau = tau
        self.max_distance = max_distance
        self.sdf_method = sdf_method
        self.reduction = reduction
        self.dim = dim
        self.eps = eps

    def _prepare(self, outputs: torch.Tensor, labels: torch.Tensor, sdf: Optional[torch.Tensor]):
        outputs = outputs.float()
        labels = labels.float()
        if self.sigmoid:
            outputs = torch.sigmoid(outputs + self.alpha)

        if outputs.ndim == self.dim + 1:
            reduce_axis = torch.arange(1, len(labels.shape)).tolist()
        else:
            reduce_axis = torch.arange(2, len(labels.shape)).tolist()

        if sdf is None:
            batch_labels = labels if labels.ndim == self.dim + 2 else labels.unsqueeze(0)
            sdf = signed_distance_map(batch_labels, self.max_distance, self.sdf_method, self.dim).reshape(labels.shape)

        mass = torch.sum(outputs, dim=reduce_axis) + self.eps
        expectation = lambda values: torch.sum(outputs * values, dim=reduce_axis) / mass
        return sdf, expectation

    def _saturate(self, excess: torch.Tensor) -> torch.Tensor:
        """ Bounded penalty in [0, 1): 0 inside the admissible region, growing with the distance outside it. """
        return 1 - torch.exp(-excess / self.tau)

    def _reduce(self, loss: torch.Tensor) -> torch.Tensor:
        if self.reduction == LossReduction.MEAN:
            loss = torch.mean(loss)
        elif self.reduction == LossReduction.SUM:
            loss = torch.sum(loss)
        return loss


class SurfDistConLoss(_DistanceConfinement):
    """ Distance-based surface confinement.

    E_p[1 - exp(-excess / tau)], with p the normalized landmark probability map and excess the distance by which
    a voxel lies outside the band [lo, hi] of signed distances to the structure boundary (negative inside,
    positive outside, in voxels). Mass within the band is free, and mass outside it is penalized in proportion
    to its distance, on either side of the surface. It replaces the overlap ratio of SurfConLoss.
    """

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor,
                band: tuple[float, float] = (-3.0, 3.0),
                sdf: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        outputs: landmark logits (B, 1, *spatial)
        labels:  binary structure segmentation (B, 1, *spatial)
        band:    admissible signed-distance interval (lo, hi) around the interface, in voxels
        sdf:     optional precomputed signed distance map of labels (same shape), reused across pairs
        """
        sdf, expectation = self._prepare(outputs, labels, sdf)
        lo, hi = band
        excess = F.relu(lo - sdf) + F.relu(sdf - hi)
        loss = expectation(self._saturate(excess))
        return self._reduce(loss)


class VolDistConLoss(_DistanceConfinement):
    """ Distance-based volume confinement.

    E_p[1 - exp(-relu(phi - hi) / tau)]: probability mass outside the structure (phi > hi, default hi = 0) is
    penalized in proportion to how far it lies outside, and mass inside is free. It replaces the overlap
    ratio of VolConLoss, which penalizes all outside mass equally regardless of its distance.
    """

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor, hi: float = 0.0,
                sdf: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        outputs: landmark logits (B, 1, *spatial)
        labels:  binary structure segmentation (B, 1, *spatial)
        hi:      largest admissible signed distance in voxels (0 = fully contained)
        sdf:     optional precomputed signed distance map of labels (same shape), reused across pairs
        """
        sdf, expectation = self._prepare(outputs, labels, sdf)
        loss = expectation(self._saturate(F.relu(sdf - hi)))
        return self._reduce(loss)


def radius_key(radius: float) -> str:
    """ Key used for the landmark radius in the 'statistics' block of a confinement config (3 -> '3', 2.5 -> '2.5'). """
    return f"{float(radius):g}"


def _distance_targets(config: dict, kind: str, landmark_id: int, segment_id: int,
                      radius: Optional[float]) -> tuple[dict, bool]:
    """ Distance targets for one landmark-structure pair, from the dataset-level statistics stored in the
    confinement config for this radius (see confinement_statistics.py), or from the idealized geometry:
      surface: band [-r, r] (a sphere of radius r centred on the interface)
      volume:  hi = 0 (a sphere fully inside the structure)
    For volume pairs the dataset upper bound is never tightened below 0, so the loss asks at most for
    containment, while still allowing landmarks whose spheres partly leave the structure in the data. """

    stats = None
    if radius is not None:
        stats = config.get("statistics", {}).get(radius_key(radius), {}).get(f"{landmark_id}-{segment_id}")

    if kind == "surface":
        if stats is not None:
            return {"band": (float(stats["band"][0]), float(stats["band"][1]))}, True
        r = float(radius) if radius is not None else 1.0
        return {"band": (-r, r)}, False

    if stats is not None:
        return {"hi": max(float(stats["band"][1]), 0.0)}, True
    return {"hi": 0.0}, False



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