# SeaNet-PyTorch
### This repository is an official PyTorch implementation of the paper "Soft-edge Assisted Network for Single Image Super-Resolution". (TIP 2020)

Paper can be download from <a href="https://ieeexplore.ieee.org/abstract/document/9007623">SeaNet</a> 

Homepage: <a href="https://junchenglee.com/projects/TIP2020_SEANET/">SeaNet</a> 

The task of single image super-resolution (SISR) is a highly ill-posed inverse problem since reconstructing the highfrequency details from a low-resolution image is challenging. Most previous CNN-based super-resolution (SR) methods tend to directly learn the mapping from the low-resolution image to the high-resolution image through some complex convolutional neural networks. However, the method of blindly increasing the depth of the network is not the best choice because the performance improvement of such methods is marginal but the computational cost is huge. A more efficient method is to integrate the image prior knowledge into the model to assist the image reconstruction. Indeed, the soft-edge has been widely applied in many computer vision tasks as the role of an important
image feature. In this paper, we propose a Soft-edge assisted Network (SeaNet) to reconstruct the high-quality SR image with the help of image soft-edge. The proposed SeaNet consists of three sub-nets: a rough image reconstruction network (RIRN), a soft-edge reconstruction network (Edge-Net), and an image refinement network (IRN). The complete reconstruction process consists of two stages. In Stage-I, the rough SR feature maps and the SR soft-edge are reconstructed by the RIRN and Edge-Net, respectively. In Stage-II, the outputs of the previous stages are fused and then fed to the IRN for high-quality SR image reconstruction. Extensive experiments show that our SeaNet converges rapidly and achieves excellent performance under the assistance of image soft-edge. 

<p align="center">

<img src="images/EdgeNet.png" width="800px"/> 

</p>

###SEAN is another name for SeaNet, which is convenient for us to conduct experiments.

All reconstructed SR images can be download from <a href="https://www.jianguoyun.com/p/DaKOxvEQ19ySBxi_1o4B">SR_Images</a> 

All test datasets (Preprocessed HR images) can be downloaded from <a href="https://www.jianguoyun.com/p/DcrVSz0Q19ySBxiTs4oB">here</a>.

All original test datasets (HR images) can be downloaded from <a href="https://www.jianguoyun.com/p/DaSU0L4Q19ySBxi_qJAB">here</a>.

## Prerequisites:

1. Python 3.6

2. PyTorch >= 0.4.0

3. numpy

4. skimage

5. imageio

6. matplotlib

7. tqdm

For more informaiton, please refer to <a href="https://github.com/thstkdgus35/EDSR-PyTorch">EDSR</a> and <a href="https://github.com/yulunzhang/RCAN">RCAN</a>.

## Document

Train/             : all train files are stored here

Test/              : all test files are stored here

README.md          : read me first

demo.sh            : all running instructions

## Dataset

We use DIV2K dataset to train our model. Please download it from <a href="https://data.vision.ee.ethz.ch/cvl/DIV2K/">here</a>  or  <a href="https://cv.snu.ac.kr/research/EDSR/DIV2K.tar">SNU_CVLab</a>.

Extract the file and put it into the Train/dataset.

Only DIV2K is used as the training dataset, and Flickr2K is not used as the training dataset !!!

We use generate_edge.m to extract the soft-edge of DIV2K and put them into the Train/dataset/DIV2K/DIV2K_train_EDGE

##Training

Using --ext sep_reset argument on your first running. 

You can skip the decoding part and use saved binaries with --ext sep argument in second time.

```python
cd Train/

# SEAN x2  LR: 48 * 48  HR: 96 * 96

python main.py --template SEAN --save SEAN_X2 --scale 2 --reset --save_results --patch_size 96 --ext sep_reset

# SEAN x3  LR: 48 * 48  HR: 144 * 144

python main.py --template SEAN --save SEAN_X3 --scale 3 --reset --save_results --patch_size 144 --ext sep_reset

# SEAN x4  LR: 48 * 48  HR: 192 * 192

python main.py --template SEAN --save SEAN_X4 --scale 4 --reset --save_results --patch_size 192 --ext sep_reset
```

##Testing

All original test datasets (HR images) can be downloaded from <a href="https://www.jianguoyun.com/p/DaSU0L4Q19ySBxi_qJAB">here</a>.

Different from previous works to select the best weight as the final model weight, we use the weight of the last epoch as our final model weight directly.

Using pre-trained model for test, all test datasets must be pretreatment by  Prepare_TestData_HR_LR.m and all pre-trained model should be put into Test/model/ first.

```python
#SEAN x2
python main.py --data_test MyImage --scale 2 --model SEAN --pre_train ../model/SEAN_x2.pt --test_only --save_results --chop --save "SEAN" --testpath ../LR/LRBI --testset Set5

#SEAN+ x2
python main.py --data_test MyImage --scale 2 --model SEAN --pre_train ../model/SEAN_x2.pt --test_only --save_results --chop --self_ensemble --save "SEAN_plus" --testpath ../LR/LRBI --testset Set5

#SEAN x3
python main.py --data_test MyImage --scale 3 --model SEAN --pre_train ../model/SEAN_x3.pt --test_only --save_results --chop --save "SEAN" --testpath ../LR/LRBI --testset Set5

#SEAN+ x3
python main.py --data_test MyImage --scale 3 --model SEAN --pre_train ../model/SEAN_x3.pt --test_only --save_results --chop --self_ensemble --save "SEAN_plus" --testpath ../LR/LRBI --testset Set5

#SEAN x4
python main.py --data_test MyImage --scale 4 --model SEAN --pre_train ../model/SEAN_x4.pt --test_only --save_results --chop --save "SEAN" --testpath ../LR/LRBI --testset Set5

#SEAN+ x4
python main.py --data_test MyImage --scale 4 --model SEAN --pre_train ../model/SEAN_x4.pt --test_only --save_results --chop --self_ensemble --save "SEAN_plus" --testpath ../LR/LRBI --testset Set5
```

We also introduce self-ensemble strategy to improve our SEAN and denote the self-ensembled version as SEAN+.

More running instructions can be found in demo.sh.

## Performance

We use  Test/PSNR_SSIM_Results_BI_model.txt for PSRN/SSIM test.

<p align="center">

<img src="images/TABLE1.png" width="800px"/> 

</p>

<p align="center">

<img src="images/TABLE2.png" width="800px"/> 

</p>

<img src="images/TABLE3.png" width="800px"/> 

</p>

<img src="images/results1.png" width="800px"/> 

</p>

<img src="images/results2.png" width="800px"/> 

</p>

Training curves:

<img src="images/loss_L1_x2.png" width="235px"/><img src="images/loss_L1_x3.png" width="235px" /> <img src="images/loss_L1_x4.png" width="235px"/> 

</p>

This work was completed in 2018, a long time ago, so there may be omissions in the code finishing process. If you have any questions, please contact me!

```
@InProceedings{fang2020soft,
    title = {Soft-Edge Assisted Network for Single Image Super-Resolutionn},
    author = {Fang, Faming and Li, Juncheng and Zeng, Tieyong},
    booktitle = {IEEE Transactions on Image Processing},
    volume = {29},
    pages = {4656--4668},
    year = {2020},
    publisher = {IEEE}
}
```

```
@InProceedings{fang2020multilevel,
    title = {Multi-level Edge Features Guided Network for Image Denoising},
    author = {Fang, Faming and Li, Juncheng, Yuan Yiting, Zeng, Tieyong, and Zhang Guxiu},
    booktitle = {IEEE Transactions on Neural Networks and Learning Systems},
    publisher = {IEEE}
}
```


This implementation is for non-commercial research use only. 
If you find this code useful in your research, please cite the above papers.
