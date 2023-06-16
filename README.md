# UHD-multi-exposure-image-fusion-algorithm
Ultra HD resolution multi-exposure image fusion algorithm, which employs an implicit function to generate a 3D LUT grid of arbitrary resolution to obtain a clear ultra HD image.

## abstract
With the rising imaging resolution of handheld devices, existing multi-exposure image fusion algorithms are difffcult to generate a high dynamic range image with ultra-high resolution in real-time. To tackle this issue, we introduce 3D LUT technology, which can enhance images with ultra high-definition (UHD) resolution in real time on resource constrained devices. However, since the fusion of information from multiple images with different exposure rates is uncertain, and this uncertainty signiffcantly trials the generalization power of the 3D LUT grid. For this, we propos a UHD-MEF network to model uncertainty on the grid to guarantee that the learning space of the model is robust.Furthermore, we provide an editable pattern for the multi-exposure image fusion algorithm by using the implicit representation function to match the requirements in different scenarios. Extensive experiments demonstrate that our proposed method is highly competitive in efffciency and accuracy.


![Image text](https://github.com/zzr-idam/UHD-multi-exposure-image-fusion-algorithm/blob/main/mobie.png)
![Image text](https://github.com/zzr-idam/UHD-multi-exposure-image-fusion-algorithm/blob/main/f1.png)


## method
The deep network learns a set of features of the input image, after which a 3D LUT grid is generated based on the size of the input coordinate matrix, and finally, the 3D LUT grid is interpolated on the information of the raw input.

## properties
1. our method can run in real time on a single GPU [RTX3090 with 24G RAM]; 2. it can also run in real time on the cell phone [huawei].


