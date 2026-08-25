# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class WHUDataset(BaseSegDataset):
    """
    WHU dataset.
    """
    METAINFO = dict(
        classes = ('Farmland', 'City', 'Village', 
                    'Water', 'Forest', 'Road', 'Others'), 
        palette = [[1, 1, 1], [2, 2, 2], [3, 3, 3], 
                    [4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7]]
    )
    # METAINFO = {
    #     'classes': ['Others', 'Farmland', 'City', 'Village', 
    #                 'Water', 'Forest', 'Road',], 
    #     'palette': [[153, 102, 153], [0, 102, 204], [0, 0, 255], [0, 255, 255], 
    #                 [255, 0, 0], [0, 167, 85], [255, 255, 0], ],
    # }
    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label, **kwargs)
        
        