""" IMPORTS """

from monai.transforms import (RandAffine, RandAffined)
from monai.utils import (GridSampleMode, GridSamplePadMode, convert_to_tensor,)

from collections.abc import (Hashable, Mapping, Sequence)
from monai.config import (KeysCollection, SequenceStr)
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.meta_tensor import MetaTensor

from monai.transforms.inverse import TraceableTransform
from monai.utils.enums import TraceKeys
from monai.data.meta_obj import get_track_meta

import torch

import numpy as np


""" CLASS """

class RandAffinePointd(RandAffined):

    backend = RandAffine.backend

    def __init__(
        self,
        image_keys: KeysCollection,
        point_keys: KeysCollection,
        spatial_size: Sequence[int] | int | None = None,
        prob: float = 0.1,
        rotate_range: Sequence[tuple[float, float] | float] | float | None = None,
        shear_range: Sequence[tuple[float, float] | float] | float | None = None,
        translate_range: Sequence[tuple[float, float] | float] | float | None = None,
        scale_range: Sequence[tuple[float, float] | float] | float | None = None,
        mode: SequenceStr = GridSampleMode.BILINEAR,
        padding_mode: SequenceStr = GridSamplePadMode.REFLECTION,
        cache_grid: bool = False,
        device: torch.device | None = None,
        allow_missing_keys: bool = False,
        lazy: bool = False,
    )-> None:
        if isinstance(image_keys, str):
            image_keys = [image_keys]
        if isinstance(point_keys, str):
            point_keys = [point_keys]
        self.image_shape_key = image_keys[0]
        self.point_keys = point_keys
        self._rand_affine = RandAffined(
            keys=image_keys,
            spatial_size=spatial_size,
            prob=prob,
            rotate_range=rotate_range,
            scale_range=scale_range,
            translate_range=translate_range,
            shear_range=shear_range,
            mode=mode,
            padding_mode=padding_mode,
            cache_grid=cache_grid,
            device=device,
            allow_missing_keys=allow_missing_keys,
            lazy=lazy,
        )

    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None) -> 'RandAffinePointd':
        self._rand_affine.set_random_state(seed, state)
        return self

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor], lazy: bool | None = None) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)

        # apply random affine to images
        d = self._rand_affine(data=d, lazy=lazy)

        # get random affine matrix
        A = self._rand_affine.rand_affine.rand_affine_grid.get_transformation_matrix()
        A = torch.inverse(A).float()

        # apply random affine matrix to bboxes
        for key in self.point_keys:
            bbox = d[key]
            bbox = convert_to_tensor(bbox, track_meta=get_track_meta())
            shape = torch.tensor(d[self.image_shape_key].squeeze().shape)
            extra_info = {
                "affine": A,
                "do_resampling": self._rand_affine._do_transform,
                "shape": shape,
            }
            meta_info = TraceableTransform.track_transform_meta(
                bbox,
                sp_size=shape,
                affine=A,
                extra_info=extra_info,
                orig_size=3,
                transform_info=self.get_transform_info(),
                lazy=lazy,
            )

            hom = torch.cat([bbox[0] - 0.5, torch.ones(1)]).reshape(4,1)
            new_hom = (A @ hom).permute(*torch.arange(hom.ndim - 1, -1, -1))
            new_hom = new_hom[0,:3] + 0.5
            tr_bbox = bbox.clone()
            tr_bbox[0] = new_hom
            tr_bbox = tr_bbox.copy_meta_from(meta_info) if isinstance(tr_bbox, MetaTensor) else tr_bbox
            d[key] = tr_bbox
        return d

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        d = self._rand_affine.inverse(d)
        for key in self.point_keys:
            transform = self.pop_transform(d[key])
            do_resampling = transform[TraceKeys.EXTRA_INFO]["do_resampling"]
            if do_resampling:
                A = transform[TraceKeys.EXTRA_INFO]["affine"]
                A = torch.inverse(A).float()
                shape = transform[TraceKeys.EXTRA_INFO]["shape"]
                bbox = d[key]
                hom = torch.cat([bbox[0] - shape/2, torch.ones(1)]).reshape(4, 1)
                new_hom = (A @ hom).permute(*torch.arange(hom.ndim - 1, -1, -1))
                new_hom = new_hom[0, :3] + shape/2
                tr_bbox = bbox.clone()
                tr_bbox[0] = new_hom
                d[key] = tr_bbox

        return d


""" MAIN """

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from scipy import ndimage
    import scipy

    def _test_RandAffineBBoxesd(title, rotate_range=None, scale_range=None, translate_range=None, shear_range=None, invert=False, world=128, sphere=False):

        # synthetic volume
        D, H, W = world, world, world
        center1 = (world/2) * torch.rand(3) + world/4
        center2 = (world/2) * torch.rand(3) + world/4
        size = (world/4) * torch.ones(3, dtype=int)
        vol1 = torch.zeros((1, D, H, W), dtype=float)
        vol2 = torch.zeros((1, D, H, W), dtype=float)

        # add sphere
        if sphere:
            x, y, z = torch.meshgrid(torch.arange(D), torch.arange(H), torch.arange(W), indexing='ij')
            grid = torch.stack((x, y, z), dim=0).float()
            dist1 = torch.sqrt(torch.sum((grid - center1.view(3, 1, 1, 1)) ** 2, dim=0))
            vol1[0] = (dist1 <= size[0]/2).float()
            dist2 = torch.sqrt(torch.sum((grid - center2.view(3, 1, 1, 1)) ** 2, dim=0))
            vol2[0] = (dist2 <= size[0]/2).float()

        # add rectangle
        else:
            vol1[0, int(center1[0]-size[0]/2):int(center1[0]+size[0]/2), int(center1[1]-size[1]/2):int(center1[1]+size[1]/2), int(center1[2]-size[2]/2):int(center1[2]+size[2]/2)] = torch.ones((1, int(size[0]), int(size[1]), int(size[2])))
            vol2[0, int(center2[0]-size[0]/2):int(center2[0]+size[0]/2), int(center2[1]-size[1]/2):int(center2[1]+size[1]/2), int(center2[2]-size[2]/2):int(center2[2]+size[2]/2)] = torch.ones((1, int(size[0]), int(size[1]), int(size[2])))

        # bounding box around object
        bbox1 = torch.stack([center1, size])
        bbox2 = torch.stack([center2, size])

        # create sample and transform
        sample = [{"image": vol1, "seg":vol2, "bbox1": bbox1, "bbox2": bbox2}, {"image": vol2, "bbox": bbox2}]

        # create transform
        transforms = RandAffineBBoxesd(["image", "seg"], ["bbox1", "bbox2"], prob=1.0, rotate_range=rotate_range, scale_range=scale_range, translate_range=translate_range, shear_range=shear_range)

        # apply transform
        out = transforms(sample[0])
        if invert:
            out = transforms.inverse(out)


        # get original bbox center
        orig_ctr = bbox1[0].detach().cpu().numpy()
        orig_ctr = [round(c) for c in orig_ctr]

        # get new bbox center
        new_ctr  = out["bbox1"].detach().cpu().numpy()[0]
        new_ctr = [round(n) for n in new_ctr]

        # get center of mass of object
        com = ndimage.center_of_mass(out["image"].detach().cpu().numpy()[0])
        com = [round(c+0.5) for c in com]

        # get original image x, y, z views
        orig_img_x = vol1[0, int(center1[0])]
        orig_img_y = vol1[0, :, int(center1[1])]
        orig_img_z = vol1[0, :, :, int(center1[2])]

        # get transformed image x, y, z views
        new_img_x  = out["image"].detach().cpu().numpy()[0, com[0]]
        new_img_y  = out["image"].detach().cpu().numpy()[0, :, com[1]]
        new_img_z  = out["image"].detach().cpu().numpy()[0, :, :, com[2]]

        # get original bounding box point and size
        b = bbox1[0]-size/2
        orig_rc_x = (b[2], b[1]), (size[2], size[1])
        orig_rc_y = (b[2], b[0]), (size[2], size[0])
        orig_rc_z = (b[1], b[0]), (size[1], size[0])

        # get transformed bounding box point and size
        o = out["bbox1"]
        o[0] = torch.tensor([int(c - s / 2) for c, s in zip(o[0], o[1])])
        o = o[0]
        new_rc_x  = (o[2], o[1]), (size[2], size[1])
        new_rc_y  = (o[2], o[0]), (size[2], size[0])
        new_rc_z  = (o[1], o[0]), (size[1], size[0])

        # plot
        fig, axes = plt.subplots(3, 2, figsize=(10, 15))
        axes[0,0].imshow(orig_img_x, cmap="gray", origin="lower")
        axes[0,0].set_title(f"Original Beam Slice X")
        if orig_rc_x:
            (x, y), (w, h) = orig_rc_x
            axes[0,0].add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor="red", linewidth=2))
        axes[0,1].imshow(new_img_x, cmap="gray", origin="lower")
        axes[0,1].set_title(f"Transformed Beam Slice X")
        if new_rc_x:
            (x, y), (w, h) = new_rc_x
            axes[0,1].add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor="red", linewidth=2))

        axes[1,0].imshow(orig_img_y, cmap="gray", origin="lower")
        axes[1,0].set_title(f"Original Beam Slice Y")
        if orig_rc_y:
            (x, y), (w, h) = orig_rc_y
            axes[1,0].add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor="red", linewidth=2))
        axes[1,1].imshow(new_img_y, cmap="gray", origin="lower")
        axes[1,1].set_title(f"Transformed Beam Slice Y")
        if new_rc_y:
            (x, y), (w, h) = new_rc_y
            axes[1,1].add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor="red", linewidth=2))

        axes[2,0].imshow(orig_img_z, cmap="gray", origin="lower")
        axes[2,0].set_title(f"Original Beam Slice Z")
        if orig_rc_z:
            (x, y), (w, h) = orig_rc_z
            axes[2,0].add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor="red", linewidth=2))
        axes[2,1].imshow(new_img_z, cmap="gray", origin="lower")
        axes[2,1].set_title(f"Transformed Beam Slice Z")
        if new_rc_z:
            (x, y), (w, h) = new_rc_z
            axes[2,1].add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor="red", linewidth=2))
        fig.suptitle(title, fontsize=20)
        plt.tight_layout()
        plt.show()

        print(f"{title}:\n  - Original center:\t{orig_ctr},\n  - Transformed center:\t{new_ctr},\n  - Transformed COM:\t{com}\n")

    def test_RandAffineBBoxesd():
        

        _test_RandAffineBBoxesd("Rotation", rotate_range=(1.0, 1.0, 1.0))

        _test_RandAffineBBoxesd("Translation", translate_range=(10, 10, 10))

        _test_RandAffineBBoxesd("Rotation + translation", rotate_range=(1.0, 1.0, 1.0), translate_range=(10, 10, 10))

        _test_RandAffineBBoxesd("Rotation + translation + scale + shear", rotate_range=(1.0, 1.0, 1.0), scale_range=(0.2, 0.2, 0.2), translate_range=(10, 10, 10), shear_range=(0.3, 0.3, 0.3))

        _test_RandAffineBBoxesd("Rotation + translation + scale + shear with inversion", rotate_range=(1.0, 1.0, 1.0), scale_range=(0.2, 0.2, 0.2), translate_range=(10, 10, 10), shear_range=(0.3, 0.3, 0.3), invert=True)

    test_RandAffineBBoxesd()