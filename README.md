# T-PAMI2023: G2-MonoDepth

The offical resposity of the paper :

[G2-MonoDepth: A General Framework of Generalized Depth Inference from Monocular RGB+X Data](https://arxiv.org/abs/2310.15422), Haotian Wang, Meng Yang, Nanning Zheng

## Abstract
![function_example_副本](https://github.com/Wang-xjtu/G2-MonoDepth/assets/56944916/97775fe9-e991-4e43-a90a-2a0a80e54aae)

Monocular depth inference is a fundamental problem for scene perception of robots. Specific robots may be equipped with a camera plus an optional depth sensor of any type and located in various scenes of different scales, whereas recent advances derived multiple individual sub-tasks. It leads to additional burdens to fine-tune models for specific robots and thereby high-cost customization in large-scale industrialization. This paper investigates a unified task of monocular depth inference, which infers high-quality depth maps from all kinds of input raw data from various robots in unseen scenes. A basic benchmark G2-MonoDepth is developed for this task, which comprises four components: (a) a unified data representation RGB+X to accommodate RGB plus raw depth with diverse scene scale/semantics, depth sparsity ([0%, 100%]) and errors (holes/noises/blurs), (b) a novel unified loss to adapt to diverse depth sparsity/errors of input raw data and diverse scales of output scenes, (c) an improved network to well propagate diverse scene scales from input to output, and (d) a data augmentation pipeline to simulate all types of real artifacts in raw depth maps for training. G2-MonoDepth is applied in three sub-tasks including depth estimation, depth completion with different sparsity, and depth enhancement in unseen scenes, and it always outperforms SOTA baselines on both real-world data and synthetic data.

## Requirments

Python=3.8

Pytorch=2.0 

## Training

Training code is coming soon.

## Testing 

1. Download and save [model](https://drive.google.com/file/d/1Cp0tRkQE0AAtvtMQcYVnb-cOj9J4CWdZ/view?usp=drive_link) to ./checkpoints/

2. Download and unzip [test dataset](https://drive.google.com/file/d/1rIkCjvSGQd4b-haedEkLkd7pbJM5hiel/view?usp=drive_link)

3. Run test.py

```python
python test.py
```

## Citation

```
@misc{wang2023g2monodepth,
      title={G2-MonoDepth: A General Framework of Generalized Depth Inference from Monocular RGB+X Data}, 
      author={Haotian Wang and Meng Yang and Nanning Zheng},
      year={2023},
      eprint={2310.15422},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


