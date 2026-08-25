# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class FMBDataset(BaseSegDataset):
    """
    FMB dataset.
    """
    
    METAINFO = dict(
        classes = ('Background', 'Road', 'Sidewalk', 'Building', 'TrafficLight', 'TrafficSign',
                    'Vegetation', 'Sky', 'Person', 'Vehicle', 'Pole'),
        palette = [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], 
                    [6, 6, 6], [7, 7, 7], [8, 8, 8],
                    [9, 9, 9], [10, 10, 10]]
        )
    
    # METAINFO = dict(
    #     classes = ('Road', 'Sidewalk', 'Building', 'TrafficLight', 'TrafficSign',
    #                 'Vegetation', 'Sky', 'Person', 'Vehicle', 'Pole'),
    #     palette = [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], 
    #                 [6, 6, 6], [7, 7, 7], [8, 8, 8],
    #                 [9, 9, 9], [10, 10, 10]]
    #     )
    
    '''
    # METAINFO = dict(
    #     classes = ('Road', 'Sidewalk', 'Building', 'TrafficLight', 'TrafficSign',
    #                 'Vegetation', 'Sky', 'Person', 'Vehicle', 'Motorcycle', 'Pole'),
    #     palette = [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], 
    #                 [6, 6, 6], [7, 7, 7], [8, 8, 8],
    #                 [9, 9, 9], [10, 10, 10], [11, 11, 11]]
    #     )
    
    # METAINFO = dict(
    #     classes = ('Road', 'Sidewalk', 'Building', 'TrafficLight', 'TrafficSign',
    #                 'Vegetation', 'Sky', 'Person', 'Car', 'Truck', 'Motorcycle', 'Pole'),
    #     palette = [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], 
    #                 [6, 6, 6], [7, 7, 7], [8, 8, 8],
    #                 [9, 9, 9], [10, 10, 10], [11, 11, 11], [12, 12, 12]]
    #     )
    
    # METAINFO = dict(
    #     classes = ('Road', 'Sidewalk', 'Building', 'TrafficLight', 'TrafficSign',
    #                 'Vegetation', 'Sky', 'Person', 'Car', 'Motorcycle', 'Pole'),
    #     palette = [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], 
    #                 [6, 6, 6], [7, 7, 7], [8, 8, 8],
    #                 [9, 9, 9], [10, 10, 10], [11, 11, 11]]
    #     )
    
    # METAINFO = dict(
    #     classes = ('Background', 'Road', 'Sidewalk', 'Building', 'TrafficLight', 'TrafficSign',
    #                 'Vegetation', 'Sky', 'Person', 'Car', 'Truck', 'Bus', 'Motorcycle', 'Bike', 'Pole'),
    #     palette = [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], 
    #                 [6, 6, 6], [7, 7, 7], [8, 8, 8],
    #                 [9, 9, 9], [10, 10, 10], [11, 11, 11], [12, 12, 12], [13, 13, 13], [14, 14, 14]]
    #     )
    '''
    

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label, metainfo=self.METAINFO, **kwargs)
        
        