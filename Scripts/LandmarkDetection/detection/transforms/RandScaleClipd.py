""" IMPORTS """

from monai.transforms import RandomizableTransform, MapTransform, RandScaleIntensity
from monai.config import KeysCollection
from monai.data.meta_obj import get_track_meta
from monai.utils import convert_to_tensor

import numpy as np

from typing import Hashable
from abc import ABCMeta
from collections.abc import Hashable


""" CUSTOM SCALING AUGMENTATION """

class RandScaleClipd(RandomizableTransform, MapTransform):
    """ https://docs.monai.io/en/latest/_modules/monai/transforms/intensity/dictionary.html#RandScaleIntensityd """
    
    backend = RandScaleIntensity.backend

    def __init__(self, keys: KeysCollection, min_scale: float, max_scale: float, prob: float = 0.15, channel_wise: bool = False) -> None:
        
        MapTransform.__init__(self, keys, False)
        RandomizableTransform.__init__(self, prob)
        factors = [min_scale-1, max_scale-1]
        self.scaler = RandScaleIntensity(factors=factors, prob=1.0, channel_wise=channel_wise)

    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None):
        
        super().set_random_state(seed, state)
        self.scaler.set_random_state(seed, state)
        return self

    def __call__(self, data):
        
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        # expect all the specified keys have same spatial shape and share same random holes
        first_key: Hashable = self.first_key(d)
        if first_key == ():
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        # all the keys share the same random scale factor
        self.scaler.randomize(d[first_key])
        for key in self.key_iterator(d):
            image_data = d[key]
            min_intensity = np.min(image_data)
            max_intensity = np.max(image_data)
            image_data = self.scaler(image_data, randomize=False)
            d[key] = np.clip(image_data, min_intensity, max_intensity)
            
        return d