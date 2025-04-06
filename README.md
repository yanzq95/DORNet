<p align="center">
<h3 align="center"> DORNet: A Degradation Oriented and Regularized Network for <br> Blind Depth Super-Resolution
<br>
:star2: CVPR 2025 (Oral) :star2:
</h3>
  
<p align="center"><a href="https://scholar.google.com/citations?user=VogTuQkAAAAJ&hl=zh-CN">Zhengxue Wang</a><sup>1*</sup>, 
<a href="https://yanzq95.github.io/">Zhiqiang Yan✉</a><sup>1*</sup>, 
<a href="https://jspan.github.io/">Jinshan Pan</a><sup>1</sup>,
<a href="https://guangweigao.github.io/">Guangwei Gao</a><sup>2</sup>,
<a href="https://cszn.github.io/">Kai Zhang</a><sup>3</sup>,
  <a href="https://scholar.google.com/citations?user=6CIDtZQAAAAJ&hl=zh-CN">Jian Yang✉</a><sup>1</sup>  <!--&Dagger;-->
</p>

<p align="center">
  <sup>*</sup>Equal contribution&nbsp;&nbsp;&nbsp;
  <sup>✉</sup>Corresponding author&nbsp;&nbsp;&nbsp;<br>
  <sup>1</sup>Nanjing University of Science and Technology&nbsp;&nbsp;&nbsp;
  <br>
  <sup>2</sup>Nanjing University of Posts and Telecommunications&nbsp;&nbsp;&nbsp;
  <sup>3</sup>Nanjing University&nbsp;&nbsp;&nbsp;
</p>

<p align="center">
<img src="Figs/Pipeline.png", width="800"/>
</p>


Overview of DORNet. Given $\boldsymbol D_{up}$ as input, the degradation learning first encodes it to produce degradation representations $\boldsymbol {\tilde{D}}$  and $\boldsymbol D $. Then, $\boldsymbol {\tilde{D}}$,  $\boldsymbol D $, $\boldsymbol D_{lr} $, and $\boldsymbol I_{r}$ are fed into multiple degradation-oriented feature transformation (DOFT) modules, generating the HR depth $\boldsymbol D_{hr}$. Finally, $\boldsymbol D$ and $\boldsymbol D_{hr}$ are sent to the degradation regularization to obtain $\boldsymbol D_{d}$, which is used as input for the degradation loss $\mathcal L_{deg}$ and the contrastive loss $\mathcal L_{cont}$. The degradation regularization only applies during training and adds no extra overhead in testing.

## Dependencies

```bash
Python==3.11.5
PyTorch==2.1.0
numpy==1.23.5 
torchvision==0.16.0
scipy==1.11.3
Pillow==10.0.1
tqdm==4.65.0
scikit-image==0.21.0
mmcv-full==1.7.2
```

## Datasets

[RGB-D-D](https://github.com/lingzhi96/RGB-D-D-Dataset)

[TOFDSR](https://yanzq95.github.io/projectpage/TOFDC/index.html)

[NYU-v2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)

## Models

Pretrained models can be found in  <a href="https://github.com/yanzq95/DORNet/tree/main/checkpoints">checkpoints</a>.


## Training

For the RGB-D-D and NYU-v2 datasets, we use a single GPU to train our DORNet. For the larger TOFDC dataset, we employ multiple GPUs to accelerate training.

### DORNet
```
Train on real-world RGB-D-D
> python train_nyu_rgbdd.py
Train on real-world TOFDSR
> python -m torch.distributed.launch --nproc_per_node 2 train_tofdsr.py --result_root 'experiment/TOFDSR'
Train on synthetic NYU-v2
> python train_nyu_rgbdd.py
```

### DORNet-T
```
Train on real-world RGB-D-D
> python train_nyu_rgbdd.py --tiny_model
Train on real-world TOFDSR
> python -m torch.distributed.launch --nproc_per_node 2 train_tofdsr.py --result_root 'experiment/TOFDSR_T' --tiny_model
Train on synthetic NYU-v2
> python train_nyu_rgbdd.py --tiny_model
```

## Testing

### DORNet
```
Test on real-world RGB-D-D
> python test_nyu_rgbdd.py
Test on real-world TOFDSR
> python test_tofdsr.py
Test on synthetic NYU-v2
> python test_nyu_rgbdd.py
```

### DORNet-T
```
Test on real-world RGB-D-D
> python test_nyu_rgbdd.py --tiny_model
Test on real-world TOFDSR
> python test_tofdsr.py --tiny_model
Test on synthetic NYU-v2
> python test_nyu_rgbdd.py --tiny_model
```

## Experiments

### Quantitative comparison

<p align="center">
<img src="Figs/Params_Time.png", width="500"/>
<br>
Complexity on RGB-D-D (w/o Noisy) tested by a 4090 GPU. A larger circle diameter indicates a higher inference time.
</p>



### Visual comparison

<p align="center">
<img src="Figs/RGBDD.png", width="1000"/>
<br>
Visual results on the real-world RGB-D-D dataset (w/o Noise).
</p>

