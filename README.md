# Face Mask Detection With MobileNet
- A Deep Learning project in Computer Vision to detect facemask on images using Keras, OpenCV and Streamlit.
- Finished time: 23/09/2022.
## About the dataset
- The dataset contains 3 folders labeled as to which class they belong to. the 3 classes are "withmask", "withoumask", and "mask_weared_incorrect". Each folder holds 2994 images of people that belong to such a labeled class.
- Dataset link: https://www.kaggle.com/datasets/vijaykumar1799/face-mask-detection
## About MobileNet
- MobileNet is a simple but efficient and not very computationally intensive convolutional neural networks for mobile vision applications. MobileNet is widely used in many real-world applications which includes object detection, fine-grained classifications, face attributes, and localization.
  ### Depthwise Separable Convolution
  - The MobileNet model is based on depthwise separable
  convolutions which is a form of factorized convolutions
  which factorize a standard convolution into a depthwise
  convolution and a 1×1 convolution called a pointwise convolution.

  - A standard convolution
  both filters and combines inputs into a new set of outputs
  in one step. The depthwise separable convolution splits this
  into two layers, a separate layer for filtering and a separate
  layer for combining.
  
  - For Normal convolution operation, see this image below:
  
  ![alt text](https://github.com/LTPhat/FaceMaskDetection-MobileNet-/blob/main/conv.png)
  
  **Standard convolutions have the computational cost of:**
    $$D_K.D_K.M.N.D_F.D_F$$
  Where $M,N$ are input and output channels, respectively. $D_K \times D_K$ is kernel size,  $D_F \times D_F$ is feature map size.
  
  - For Depthwise Convolution, the filtering and combination steps can be split into two
  steps via the use of factorized convolutions called depthwise, separable convolutions for substantial reduction in computational cost.
  - Depthwise separable convolution are made up of two layers: Depthwise convolutions and Pointwise convolutions. 
  - Depthwise convolutions is used to apply a single filter per each input channel (input depth). Pointwise convolution, a simple 1×1 convolution, is then used to create a linear combination of the output of the depthwise layer
  
  ![alt text](https://github.com/LTPhat/FaceMaskDetection-MobileNet-/blob/main/depth_wise_conv.png)
  
  **Depthwise separable convolutions cost:**
  $$D_K \times D_K \times M \times D_F \times D_F + M \times N \times D_F \times D_F $$
  
  $\Rightarrow$ **Reduce in computational cost:** $$\frac{D_K \times D_K \times M \times D_F \times D_F + M \times N \times D_F \times D_F}{D_K.D_K.M.N.D_F.D_F}= \frac{1}{N}+\frac{1}{D^2_K}$$
  ### Architecture
  
  ![alt text](https://github.com/LTPhat/FaceMaskDetection-MobileNet-/blob/main/mobile_net.png)
  
## Mobilenet in Keras
- Syntax and parameters:

```sh
tf.keras.applications.MobileNet(
  input_shape=None,
  alpha=1.0,
  depth_multiplier=1,
  dropout=0.001,
  include_top=True,
  weights="imagenet",
  input_tensor=None,
  pooling=None,
  classes=1000,
  classifier_activation="softmax",
  **kwargs)
```
- Call function in Keras:
```sh
from tensorflow.keras.applications import MobileNet
base = MobileNet(include_top = False, weights = "imagenet", input_shape = (128,128,3))
```
- Freeze previous layers weights:
```sh
for layer in base.layers:
  layer.trainable = False
```

## Kaggle and model
```sh
!pip install -q kaggle

```
- Upload API dataset in json file from local.
```sh
Saving kaggle.json to kaggle.json
{'kaggle.json': b'{"username":"ltp0203","key":"fffdf39b9c00572aa906e7402280150a"}'}

```
- Create kaggle folder
```sh
! mkdir ~/.kaggle
```
- Copy the kaggle.json to folder created.
```sh
! cp kaggle.json ~/.kaggle/
```
- Permission for the json to act
```sh
! chmod 600 ~/.kaggle/kaggle.json
```
- Download dataset from kaggle FER2013
```sh
!kaggle datasets download -d vijaykumar1799/face-mask-detection
```
- Model execution is saved in: **FacemaskDetection.ipynb** https://github.com/LTPhat/FaceMaskDetection-MobileNet-/blob/main/FacemaskDetection.ipynb
- Loss and Accuracy evaluation:

![alt text](https://github.com/LTPhat/FaceMaskDetection-MobileNet-/blob/main/loss_acc.png)


## Real-time detection
- Algorithm: Haarcascades
- Source: https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
- Results are saved in **real_time_detect*.*
## About Streamlit

- Streamlit is an open-source app framework for Machine Learning and Data Science teams.
- Install Streamlit:
```sh
! pip install streamlit
```

- Web surface to deploy trained model:

![alt text](https://github.com/LTPhat/FaceMaskDetection-MobileNet-/blob/main/web_img/main_surface.png)

- Upload image to predict: 

![alt text](https://github.com/LTPhat/FaceMaskDetection-MobileNet-/blob/main/web_img/upload_img.png)

## Results 

Some results after uploading images:

![alt text](https://github.com/LTPhat/FaceMaskDetection-MobileNet-/blob/main/web_img/result.png)

![alt text](https://github.com/LTPhat/FaceMaskDetection-MobileNet-/blob/main/web_img/result2.png)

![alt text](https://github.com/LTPhat/FaceMaskDetection-MobileNet-/blob/main/web_img/result8.png)

![alt text](https://github.com/LTPhat/FaceMaskDetection-MobileNet-/blob/main/web_img/result3.png)

![alt text](https://github.com/LTPhat/FaceMaskDetection-MobileNet-/blob/main/web_img/result9.png)

![alt text](https://github.com/LTPhat/FaceMaskDetection-MobileNet-/blob/main/web_img/result6.png)

![alt text](https://github.com/LTPhat/FaceMaskDetection-MobileNet-/blob/main/web_img/result10.png)

## References

[1] MobileNet, MobileNetV2, and MobileNetV3: https://keras.io/api/applications/mobilenet/

[2] MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications: https://arxiv.org/pdf/1704.04861.pdf

[3] MobileNets - Mô hình gọn nhẹ cho mobile applications: https://viblo.asia/p/cnn-architecture-series-1-mobilenets-mo-hinh-gon-nhe-cho-mobile-applications-1VgZvJV1ZAw

  
  
  
  
