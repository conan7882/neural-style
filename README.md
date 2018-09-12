# Neural Style

- TensorFlow implementation of [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) (2015).
- This paper combines the content and style of two different images by using features extracted from a pre-trained CNN (for image classification task).
- The style transfer process proposed in this paper is a optimization process, which minimizes the difference of content and style features between the output image and input context and style images.

## Requirements
- Python 3.3+
- [Tensorflow 1.3](https://www.tensorflow.org/)
- [TensorCV](https://github.com/conan7882/DeepVision-tensorflow) 

## Implementation Details

- VGG19 is used the same as the paper. Content Layer is conv4_2 and style layers are conv1_1, conv2_1, conv3_1, conv4_1 and conv5_1.
- Inspired by [this implementationn](https://github.com/anishathalye/neural-style), Both content and style features are normalized based on the size of content and style images, respectively. I found this is useful especially when the two images have large difference in size.
- The image is initialized by the content image. This helps to obtain to a good result faster. 
- The weights for content and style costs used in this implementation are 5e-4 and 0.2, respectively. If the image initialized from a random noise, higher content cost weight maybe needed.
- [Total variation regularization](https://en.wikipedia.org/wiki/Total_variation_denoising) is used to reduce noise in the result image. The weight 0.01 is used for total variation regularization.
- [L-BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS) is used for optimization. The maximum iteration is set to be 500, though the result does not change after 200 iterations. 

<!--
## TODO

- [x] Style transfer initialized by content image
- [ ] Tweaking hyperparameters for random initialization
- [ ] Color preserve
- [ ] Mask transfer
- [ ] Multiple styles
-->


## Result
<p align = 'center'>
<img src ="nerual_style/fig/cat.png" height="300px" />
</p>
<p align = 'center'>
<a href = 'nerual_style/fig/vangohg.jpg'><img src ="nerual_style/fig/van_s.png" height="300px" /></a>
<img src ="nerual_style/fig/cat_van.png" height="300px" />
<a href = 'nerual_style/fig/chong.jpg'><img src ="nerual_style/fig/chong_s.png" height="300px" /></a>
<img src ="nerual_style/fig/cat_chong.png" height="300px" />
<a href = 'nerual_style/fig/la_muse.jpg'><img src ="nerual_style/fig/la_s.png" height="300px" /></a>
<img src ="nerual_style/fig/cat_la.png" height="300px" />
<a href = 'nerual_style/fig/mo.jpg'><img src ="nerual_style/fig/mo_s.png" height="300px" /></a>
<img src ="nerual_style/fig/cat_mo.png" height="300px" />
<a href = 'nerual_style/fig/the_scream.jpg'><img src ="nerual_style/fig/scream_s.png" height="300px" /></a>
<img src ="nerual_style/fig/cat_scream.png" height="300px" />
<a href = 'nerual_style/fig/oil.jpg'><img src ="nerual_style/fig/oil_s.png" height="300px" /></a>
<img src ="nerual_style/fig/cat_oil.png" height="300px" />
</p>

<!--## Tweaking parameters-->

## Preparation

1. Setup directories in file `main.py`. 
    - `STYLE_PATH` - directory of style image
    - `CONTENT_PATH` - directory of content image
    - `VGG_PATH` - directory of pre-trained VGG19 parameters
    - `SAVE_DIR` - directory of saving result images
   
2. Download the pre-trained VGG parameters
    - Download pre-trained VGG19 model [here](https://github.com/machrisaa/tensorflow-vgg#tensorflow-vgg16-and-vgg19) and put it in `VGG_PATH`.


## Run Script:

Put style and content images in `STYLE_PATH` and  `CONTENT_PATH`, just run:

```
python main.py --style STYLE_IM_FILE --content CONTENT_IM_FILE --wstyle STYLE_WEIGHT \
--wcontent CONTENT_WEIGHT --wvariation TOTAL_VARIATION_WEIGHT
```	

- Add ``` --rescale``` to re-scale style image to the size comparable to content image if style image is larger than content image.
- Use ``` --cscale SIDE``` to re-scale the content image with largest side = SIDE.
-  Result will be saved in `SAVE_DIR` every 20 iteraton.




<!--# Art Style Transfer

<!--
[![Build Status](https://travis-ci.org/conan7882/art_style_transfer_TensorFlow.svg?branch=master)](https://travis-ci.org/conan7882/art_style_transfer_TensorFlow)
[![Coverage Status](https://coveralls.io/repos/github/conan7882/art_style_transfer_TensorFlow/badge.svg?branch=master)](https://coveralls.io/github/conan7882/art_style_transfer_TensorFlow?branch=master)
-->

- This repository contains implementations of art style transfer algorithms in recent papers.
- The source code in the repository can be used to demostrate the algorithms as well as test on your own data.

## Requirements
- Python 3.3+
- [Tensorflow 1.3](https://www.tensorflow.org/)
- [TensorCV](https://github.com/conan7882/DeepVision-tensorflow) 


## Algorithms 

- [Neural Style](https://github.com/conan7882/art_style_transfer_TensorFlow/tree/master/nerual_style#neural-style) (2015)


## Neural Style

- This algorithm combines two images (content and style image) by using content and style features extracted from a pre-trained CNN (ex. VGG).
- The stylized image is generated by minimizing the difference of content and style features between a random noise image and input images. 
- Details of the implementation and more results can be find [here](https://github.com/conan7882/art_style_transfer_TensorFlow/tree/master/nerual_style). Some results:

<p align = 'center'>
<img src ="nerual_style/fig/cat.png" height="300px" />
<a href = 'nerual_style/fig/chong.jpg'><img src ="nerual_style/fig/chong_s.png" height="300px" /></a>
<img src ="nerual_style/fig/cat_chong.png" height="300px" />
</p>
-->
