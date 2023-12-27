## Homework 1

# 概述

本次作业的目标是编写一个图像滤波函数，并使用它来创建混合图像，使用Oliva、Torralba和Schyns的SIGGRAPH 2006论文的简化版本。混合图像是一种静态图像，其解释会随着观察距离的变化而改变。**基本思想是在高频部分可用的情况下，高频往往会主导感知，但在距离较远时，只能看到信号的低频（平滑）部分。**通过混合一幅图像的高频部分与另一幅图像的低频部分，您可以得到一幅混合图像，在不同的观察距离下会导致不同的解释。


# 实现细节

这个项目旨在让您熟悉Python、NumPy和图像滤波。一旦您创建了图像滤波函数，构建混合图像就相对简单了。

这个项目要求您实现5个函数，<u>每个函数都建立在前一个函数的基础上</u>：

1. `cross_correlation_2d`
2. `convolve_2d`
3. `gaussian_blur_kernel_2d`
4. `low_pass`
5. `high_pass`

## 图像滤波

图像滤波（或卷积）是一种基本的图像处理工具。NumPy有许多内置且高效的函数可用于执行图像滤波，但是为了这个任务，您将从头开始编写自己的滤波函数。具体来说，您将实现`cross_correlation_2d`，然后实现`convolve_2d`，后者将使用`cross_correlation_2d`。

**Correlation**

$G[i,j]=\sum_{u=-k}^k\sum_{v=-k}^kh[u,v]F[i+u,j+v]$.

**Convolution**

$G[i,j]=\sum_{u=-k}^k\sum_{v=-k}^kh[u,v]F[i-u,j-v]$.

## 高斯模糊

有几种不同的方法可以模糊图像，例如对相邻像素取未加权平均值。高斯模糊是一种特殊的加权相邻像素平均。要实现高斯模糊，您将实现一个函数`gaussian_blur_kernel_2d`，该函数生成给定高度和宽度的核，然后可以将其传递给`convolve_2d`，以及一张图像，以生成图像的模糊版本。

## 高通和低通滤波器

回想一下，低通滤波器是从图像（或者说任何信号）中去除细节的滤波器，而高通滤波器只保留细节，并去除图像中的粗细节。因此，使用上述描述的高斯模糊，实现`high_pass`和`low_pass`函数。

## 混合图像

混合图像是一个图像的低通滤波版本和另一个图像的高通滤波版本之和。有一个自由参数，可以为每对图像进行调整，以控制从第一个图像中去除多少高频以及在第二个图像中保留多少低频。这被称为“截止频率”。在论文中建议使用两个截止频率（为每个图像调整一个），您可以尝试这样做。在起始代码中，通过更改用于构建混合图像中的高斯滤波器的标准差（sigma）来控制截止频率。我们为您提供了创建混合图像的代码，使用了上述描述的函数。

## 禁止使用的函数

仅在此任务中，您被禁止使用任何Numpy、Scipy、OpenCV或其他预实现的滤波函数。您可以使用基本的矩阵操作，如`np.shape`、`np.zeros`和`np.transpose`。这个限制将在以后的任务中解除，但目前，您应该使用for循环或Numpy向量化来将核应用于图像中的每个像素。您的大部分代码将位于`cross_correlation_2d`和`gaussian_blur_kernel_2d`中，其他函数将直接或通过您实现的其他函数使用这些函数。

## 调试和示例

我们在`gui.py`中为您提供了一个GUI，以帮助您调试图像滤波算法。要查看示例图像的预标记版本，请运行：


```
python3 gui.py -t resources/sample-correspondence.json -c resources/sample-config.json
```

我们为您提供了一对需要使用GUI进行对齐的图像。用UI指定眼睛对眼睛、鼻子对鼻子等，代码会使用仿射变换来进行对齐。我们鼓励您创建其他示例（例如，表情变化、不同对象之间的变形、随时间的变化等）。有关一些灵感，请参阅混合图像项目页面。该项目页面还包含了Siggraph演示的材料。
