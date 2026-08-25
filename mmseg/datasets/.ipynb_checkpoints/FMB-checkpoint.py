# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class FMBDataset(BaseSegDataset):
    """
    FMB dataset.
    """
    METAINFO = dict(
         classes=( 'Person', 'Motorcycle','Road',
                   'Car',     'Pole',     'Vegetation','Sky',
                   'Truck',   'Building', 'Lamp',      'Sign',
                   'Bus',     'Sidewalk', 'Background','Bicycle'
                 ),
         palette=[[5, 5, 5],  [12,12,12], [1, 1, 1],
                  [9, 9, 9],  [14,14,14], [6, 6, 6],  [7, 7, 7],
                  [10,10,10], [3, 3, 3],  [4, 4, 4],  [8, 8, 8],
                  [11,11,11], [2, 2, 2],  [0, 0, 0],  [13,13,13]
                 ])
    
    # METAINFO = dict(
    #      classes=( 'Person', 'Motorcycle','Road',
    #                'Truck',   'Building', 'Lamp',      'Background',
    #                'Car',     'Pole',     'Vegetation','Sky',
    #                'Bus',     'Sidewalk', 'Sign',      'Bicycle'),
    #      palette=[[5, 5, 5],  [12,12,12], [1, 1, 1],
    #               [10,10,10], [3, 3, 3],  [4, 4, 4],  [0, 0, 0],
    #               [9, 9, 9],  [14,14,14], [6, 6, 6],  [7, 7, 7],
    #               [11,11,11], [2, 2, 2],  [8, 8, 8],  [13,13,13]])
    
    # METAINFO = dict(
    #      classes=( 'Person', 'Motorcycle', 'Road',
    #               'Car', 'Pole', 'Vegetation', 'Sky',
    #               'Truck', 'Building', 'Lamp', 'Sign',
    #               'Bus', 'Sidewalk','nan','Background','nan2', 'Bicycle'),
    #      palette=[ [5, 5, 5],  [12,12,12], [1, 1, 1],
    #               [9, 9, 9],  [14,14,14], [6, 6, 6],  [7, 7, 7],
    #               [10,10,10], [3, 3, 3],  [4, 4, 4],  [8, 8, 8],
    #               [11,11,11], [2, 2, 2],[20,20,20],[0, 0, 0],[13,13,13],[13,13,13],])
    #METAINFO = dict()

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label, metainfo=self.METAINFO, **kwargs)
        
        