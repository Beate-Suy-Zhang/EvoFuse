# EvoFuse

Jinyuan Liu, Bowei Zhang, Ludan Sun, Xingyuan Li, Long Ma, Risheng Liu, Xin Fan, **"Symbiotic Evolutionary Learning for Task-Adaptive Infrared and Visible Image Fusion"**,
IEEE Transactions on Pattern Analysis and Machine Intelligence **(TPAMI)**, 2026.

[[Paper](https://doi.org/10.1109/TPAMI.2026.3722665)]

![Overview of EvoFuse](Figure/First_Figure.png)


## Updates

[2026-08-10] Our paper is available online in IEEE Transactions on Pattern Analysis and Machine Intelligence! [[Paper](https://doi.org/10.1109/TPAMI.2026.3722665)]  


## Introduction

We propose **EvoFuse**, a symbiotic evolutionary learning framework for task-adaptive infrared and visible image fusion. EvoFuse formulates fusion and downstream perception as a multi-objective optimization problem and dynamically evolves task-aware loss-weight configurations under a Pareto-inspired non-degradation criterion. A diverse reweight architecture provides multi-branch representation capacity during training and can be re-parameterized into an ultra-compact single-branch model for efficient inference. In addition, a saliency discriminative loss emphasizes semantically important regions and improves the utility of fused images for heterogeneous downstream tasks.


## Environment

```bash
# create virtual environment
conda create -n DCEvo python=3.9
conda activate DCEvo
# install requirements
pip install -r requirements.txt
```


## Datasets Preparation

We provide a demo dataset in **"DCEvo/datasets"**.

We test image fusion according to full datasets in the [[IVIF ZOO Project](https://github.com/RollingPlain/IVIF_ZOO/)].  


## Test Image Fusion

Our checkpoints can be found in **"DCEvo/ckpt"**. Then, you can test our pure fusion method through

```bash
python test_Fusion.py
```


## Color Gray Images

You can color the output gray images for task-guided image fusion training and testing through

```bash
python tocolor.py
```


## Fusion Results

1. Quantitative comparison of EvoFuse and state-of-the-art infrared and visible image fusion methods on the TNO, RoadScene, FMB, and M3FD datasets.

![Quantitative comparison of image fusion](Figure/fusionTable.png)

2. Qualitative comparison of EvoFuse and existing image fusion methods on the TNO, RoadScene, and M3FD datasets.

![Qualitative comparison of image fusion](Figure/fusionResultComp.png)


## Test Task-Guided Image Fusion

Testing this pipeline needs to generate **RGB Pure Fusion** images in **"DCEvo/datasets/M3FD/images"**.
You can test our task-guided fusion method through

```bash
python test_task_guided_fusion.py
```


## Results of Task-Adaptive Downstream IVIF Applications

### Image Semantic Segmentation

1. Quantitative comparison on the FMB and MFNet datasets.

![Quantitative comparison of semantic segmentation](Figure/SSTable.png)

2. Qualitative comparison on the FMB and MFNet datasets.

![Qualitative comparison of semantic segmentation](Figure/SSComp.png)


### Salient Object Detection

1. Quantitative comparison on the VT821, VT1000, and VT5000 datasets.

![Quantitative comparison of salient object detection](Figure/SODTable.png)

2. Qualitative comparison on the VT821, VT1000, and VT5000 datasets.

![Qualitative comparison of salient object detection](Figure/SODComp.png)


### Multiclass Object Detection

1. Quantitative comparison on the M3FD and MSOD datasets.

![Quantitative comparison of object detection](Figure/ODtable.png)

2. Qualitative comparison on the M3FD and MSOD datasets.

![Qualitative comparison of object detection](Figure/ODComp.png)


### Remote Sensing Semantic Segmentation

1. Quantitative comparison on the WHU and Potsdam datasets.

![Quantitative comparison of remote sensing semantic segmentation](Figure/RStable.png)

2. Qualitative comparison on the Potsdam and WHU datasets.

![Qualitative comparison of remote sensing semantic segmentation](Figure/RSComp.png)


## Complexity Analysis

EvoFuse achieves a favorable balance between inference efficiency and perceptual adaptability with a compact re-parameterized fusion network.

![Complexity analysis](Figure/Complexity.png)


## Train Pure Image Fusion

You can train our Pure fusion method through

```bash
python Fusion_train.py
```

Training dataset is available in [[BaiduNetdisk](https://pan.baidu.com/s/1xfnkQUQ-5fLT9e7XS7XV1A?pwd=msrs)].  


## Train Task-Guided Image Fusion

Training this pipeline needs to generate **RGB Pure Fusion** images in **"DCEvo/datasets/M3FD/train/images"**.
You can train our task-guided fusion method through

```bash
python DCEvo_train.py
```


## Citation

```bibtex
@ARTICLE{Liu_2026_TPAMI,
    author  = {Liu, Jinyuan and Zhang, Bowei and Sun, Ludan and Li, Xingyuan and Ma, Long and Liu, Risheng and Fan, Xin},
    title   = {Symbiotic Evolutionary Learning for Task-Adaptive Infrared and Visible Image Fusion},
    journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
    year    = {2026},
    pages   = {1--18},
    doi     = {10.1109/TPAMI.2026.3722665}
}
```
