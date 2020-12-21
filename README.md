# Introduction
Official pytorch code for the NeurIPS 2020 spotlight paper "[Fewer is More: A Deep Graph Metric Learning Perspective Using Fewer Proxies](https://proceedings.neurips.cc/paper/2020/hash/ce016f59ecc2366a43e1c96a4774d167-Abstract.html)".
Slides of our NeurIPS 2020 spotlight talk are avialable [here](https://github.com/YuehuaZhu/ProxyGML/blob/main/SpotlightPPT.pptx) and each page of ppt contains comments. Welcome to share our work in the group meeting!

# Requirements and Installation
We recommended the following dependencies.

- PyTorch==1.2
- Pillow==5.2.0
- tqdm==4.26.0
- matplotlib==2.2.2
- pandas==0.23.4
- scipy==1.2.1
- scikit-learn==0.20.3
- scikit-image==0.14.2
- h5py==2.9.0


### The pipline of ProxyGML

<img src="https://github.com/YuehuaZhu/ProxyGML/blob/main/net/pipline.png" width="745" alt="pipline"/> 

## Datasets

1. Download three public benchmarks for deep metric learning 
   - [CUB-200-2011](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz)
   - Cars-196 ([Img](http://imagenet.stanford.edu/internal/car196/car_ims.tgz), [Annotation](http://imagenet.stanford.edu/internal/car196/cars_annos.mat))
   - Stanford Online Products ([Link](https://cvgl.stanford.edu/projects/lifted_struct/))
   
2. All datasets are preprocessed as follows ( take CUB200-2011 for eaxample)  and stored in .\data

```
cub200                                         
└───train
|    └───0
|           │   xxx.jpg
|           │   ...
|
|    ...
|
|    └───99
|           │   xxx.jpg
|           │   ...

└───test
|    └───100
|           │   xxx.jpg
|           │   ...
|
|    ...
|
|    └───199
|           │   xxx.jpg
|           │   ...
|    ...
```

## Training Embedding Network

Note that a fine-grained combination of parameter N and parameter r resulted in better overall performance than that described in the paper.

### CUB-200-2011
```bash
python train.py -b 32 --gpu 2 --dataset cub200 --freeze_BN --epochs 50 --dim 512 --r 0.05 -C 100 --N 12 --weight_lambda 0.3 --centerlr 0.03 --rate 0.1 --new_epoch_to_decay 20 40
```

### Cars-196
```bash
python train.py -b 32 --gpu 2 --dataset cars196 --freeze_BN --epochs 50 --dim 512 --r 0.05 -C 98 --N 12 --weight_lambda 0.3 --centerlr 0.03 --rate 0.1 --new_epoch_to_decay 20 40
```

### Stanford Online Products
```bash
python train.py -b 32 --gpu 2 --dataset online_products --epochs 50 --dim 512 --r 0.05 -C 11318 --N 1 --weight_lambda 0.0 --centerlr 0.3 --rate 0.1 --new_epoch_to_decay 20 40
```
