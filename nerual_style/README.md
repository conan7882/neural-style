# Neural Style

- TensorFlow implementation of [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) (2015).
- This paper combine the content and style of two different images by using image features extracted from pre-trained CNN (for image classification task).
- The style transfer process proposed in this paper is a optimization process, which minimizes the difference of content and style features between a random noise image and input context and style images.

## Requirements
- Python 3.3+
- [Tensorflow 1.3](https://www.tensorflow.org/)
- [TensorCV](https://github.com/conan7882/DeepVision-tensorflow) 

## Implementation Details

- VGG19 is used the same as the paper. Content Layer is conv4_2 and style layers are conv1_1, conv2_1, conv3_1, conv4_1 and conv5_1.

- Both content and style features is normalized based on the size of content and style images, which is inpired by [this implementation](https://github.com/anishathalye/neural-style). Because I found this is useful especially when the two images have large difference in size.

- The image is initialized by the content image. This helps to converge to good result faster. 

- The weights of content and style costs used in this implementation are 5e-4 and 0.2, respectively. Tweaking is needed when other types of normalization and initialization are used. Usually higher content cost weight if initialization from random noise.

- [Total variation regularization](https://en.wikipedia.org/wiki/Total_variation_denoising) is used to reduce noise in the result image.

## Result
<p align = 'center'>
<img src ="fig/cat.png" height="300px" />
</p>
<p align = 'center'>
<img src ="fig/van_s.png" height="300px" />
<img src ="fig/test_520.png" height="300px" />
<img src ="fig/chong_s.png" height="300px" />
<img src ="fig/test_520 4.png" height="300px" />
<img src ="fig/la_s.png" height="300px" />
<img src ="fig/test_520 2.png" height="300px" />
<img src ="fig/mo_s.png" height="300px" />
<img src ="fig/test_500 2.png" height="300px" />
<img src ="fig/scream_s.png" height="300px" />
<img src ="fig/test_500.png" height="300px" />
<img src ="fig/oil_s.png" height="300px" />
<img src ="fig/test_500 3.png" height="300px" />
</p>





