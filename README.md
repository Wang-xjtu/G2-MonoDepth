# T-PAMI2024: G2-MonoDepth

**[G2-MonoDepth: A General Framework of Generalized Depth Inference from Monocular RGB+X Data](https://arxiv.org/abs/2310.15422)**

**Haotian Wang, Meng Yang, Nanning Zheng**

**IEEE Transactions on Pattern Analysis and Machine Intelligence (T-PAMI), May 2024**

## News

- Training code is released! 17/05/2024

## Abstract

![function_example_副本](https://github.com/Wang-xjtu/G2-MonoDepth/assets/56944916/97775fe9-e991-4e43-a90a-2a0a80e54aae)

Monocular depth inference is a fundamental problem for scene perception of robots. Specific robots may be equipped with a camera plus an optional depth sensor of any type and located in various scenes of different scales, whereas recent advances derived multiple individual sub-tasks. It leads to additional burdens to fine-tune models for specific robots and thereby high-cost customization in large-scale industrialization. This paper investigates a unified task of monocular depth inference, which infers high-quality depth maps from all kinds of input raw data from various robots in unseen scenes. A basic benchmark G2-MonoDepth is developed for this task, which comprises four components: (a) a unified data representation RGB+X to accommodate RGB plus raw depth with diverse scene scale/semantics, depth sparsity ([0%, 100%]) and errors (holes/noises/blurs), (b) a novel unified loss to adapt to diverse depth sparsity/errors of input raw data and diverse scales of output scenes, (c) an improved network to well propagate diverse scene scales from input to output, and (d) a data augmentation pipeline to simulate all types of real artifacts in raw depth maps for training. G2-MonoDepth is applied in three sub-tasks including depth estimation, depth completion with different sparsity, and depth enhancement in unseen scenes, and it always outperforms SOTA baselines on both real-world data and synthetic data.

## Requirments

Python=3.8

Pytorch=2.0

## Train

#### Prepare your data

1. save your rgbd datasets in `./RGBD_Datasets`

```
└── RGBD_Datasets
 ├── Dataset1
 │   ├── rgb
 │   │   ├── file1.png
 │   │   ├── file2.png
 │   │   └── ...
 │   └── depth
 │       ├── file1.png
 │       ├── file2.png
 │       └── ...
 └── Dataset2
     ├── rgb
     │   ├── file1.png
     │   ├── file2.png
     │   └── ...
     └── depth
         ├── file1.png
         ├── file2.png
         └── ...    
```

**Notably:** `depth` should be stored in 16-bit data. Specifically, depth maps are normalized by `depth/max_depth*65535`, where `max_depth` is `20`(m) for indoor dataset and `100`(m) for outdoor dataset. We release the [UnrealCV](https://drive.google.com/file/d/1svV_j8IwjH1fcF4iDtAAh4MRw0Ig00X-/view?usp=drive_link) dataset as one example.

2. save your hole datasets in `./Hole_Datasets`

```
└── Hole_Datasets
 ├── Dataset1
 │   ├── file1.png
 │   ├── file2.png
 │   └── ...
 └── Dataset2
     ├── file1.png
     ├── file2.png
     └── ...
```

**Notably:** hole maps should be stored in Uint8 format. Specifically, `pixels without holes = 255` and `pixels within holes = 0`. We release the [hole collected from HRWSI](https://drive.google.com/file/d/1iKJEWgd36ebEVbG-01_gDipYuCCs7ZQZ/view?usp=drive_link) dataset as one example.

#### Start your training

1. Run `train.py`

```
python train.py
```

2. The trained model is saved in `./checkpoints/models`

## Test

1. Download and save [test model](https://drive.google.com/file/d/1Cp0tRkQE0AAtvtMQcYVnb-cOj9J4CWdZ/view?usp=drive_link) to `./checkpoints/models`

2. Download and unzip [test dataset](https://drive.google.com/file/d/1rIkCjvSGQd4b-haedEkLkd7pbJM5hiel/view?usp=drive_link) to `./Test_Datasets`

3. Run `test.py`

```python
python test.py
```

**Notably:** `gt` in test data are also stored in 16-bit data. Specifically, depth maps are normalized by `gt/max_depth*65535`, where `max_depth` is `20`(m) for indoor dataset and `100`(m) for outdoor dataset.

## Citation

```
@ARTICLE{10373158,
  author={Wang, Haotian and Yang, Meng and Zheng, Nanning},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={G2-MonoDepth: A General Framework of Generalized Depth Inference From Monocular RGB+X Data}, 
  year={2024},
  volume={46},
  number={5},
  pages={3753-3771},
  keywords={Task analysis;Data models;Estimation;Training;Semantics;Pipelines;Service robots;Robot;unified model;generalization;depth estimation;depth completion;depth enhancement},
  doi={10.1109/TPAMI.2023.3346466}}
```
