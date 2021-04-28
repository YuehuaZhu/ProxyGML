# ProxyGML: A Simple yet Powerful Metric Learning Strategy
Official PyTorch code for the NeurIPS 2020 spotlight paper "[Fewer is More: A Deep Graph Metric Learning Perspective Using Fewer Proxies](https://proceedings.neurips.cc/paper/2020/hash/ce016f59ecc2366a43e1c96a4774d167-Abstract.html)". In this paper, we propose a novel graph-based deep metric learning loss, namely [ProxyGML](https://github.com/YuehuaZhu/ProxyGML/blob/main/loss/ProxyGML.py), which is simple to implement. The pipeline of ProxyGML is as shown below.

<img src="https://github.com/YuehuaZhu/ProxyGML/blob/main/net/pipline.png" width="745" alt="pipline"/> 

## Slides&Poster&Video

[Slides](https://github.com/YuehuaZhu/ProxyGML/blob/main/SpotlightPPT.pptx) and [poster](https://github.com/YuehuaZhu/ProxyGML/blob/main/poster.pdf) of our NeurIPS 2020 10-min spotlight talk are available. Each page of the slides contains comments, which we assume can be of help to better understand our work. Our [3-min poster video presentation](https://nips.cc/virtual/2020/protected/poster_ce016f59ecc2366a43e1c96a4774d167.html) is also available. Feel free to use our slides if you want to share our work in your group meeting or introduce it to your friends!

## Installation
We recommend the following dependencies.

- pytorch==1.2
- pillow==5.2.0
- tqdm==4.26.0
- matplotlib==2.2.2
- pandas==0.23.4
- scipy==1.2.1
- scikit-learn==0.20.3
- scikit-image==0.14.2
- h5py==2.9.0

## Datasets

1. Download three public benchmarks for deep metric learning 
   - [CUB-200-2011](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz)
   - Cars-196 ([Img](http://imagenet.stanford.edu/internal/car196/car_ims.tgz), [Annotation](http://imagenet.stanford.edu/internal/car196/cars_annos.mat))
   - Stanford Online Products ([Link](https://cvgl.stanford.edu/projects/lifted_struct/))
   
2. We train our model in the paradigm of classification. So all datasets are preprocessed into train/test parts as follows ( take CUB200-2011 for example)  and stored in .\data

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

## Getting Started

Note that a fine-grained combination of parameter $N$ and parameter $r$ resultes in better overall performance than that described in the paper.

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

## Citation

If you find our paper or this project helps your research, please kindly consider citing our work via:
```
@inproceedings{Zhu2020ProxyGML,
  title={Fewer is More: A Deep Graph Metric Learning Perspective Using Fewer Proxies},
  author={Yuehua Zhu and Muli Yang and Cheng Deng and Wei Liu},
  booktitle={NeurIPS},
  year={2020}
}
```
