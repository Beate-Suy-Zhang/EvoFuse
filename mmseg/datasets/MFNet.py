# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class MFNetDataset(BaseSegDataset):
    """
    MFNet dataset.
    """
    
    METAINFO = dict(
        classes = ('Background', 'Car', 'Person', 'Bike', 'Curve', 'Car_stop',
                    'Guardrail', 'Color_cone', 'Bump', ),
        palette = [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], 
                    [6, 6, 6], [7, 7, 7], [8, 8, 8],]
        )
    

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label, metainfo=self.METAINFO, **kwargs)
        
        