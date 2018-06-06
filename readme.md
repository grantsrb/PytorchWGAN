# Convolutional Wasserstein GAN
### May 11th, 2018

## Description
Generative Adversarial Networks are a deep learning model architecture that fabricate realistic data in the image of a real data set.

The basic setup of a GAN consists of two networks. One of the two networks, known as the discriminator, tries to distinguish between real and generated images. The other network, known as the generator, generates images with the goal of fooling the first network. 

Wasserstein GANs use the Wassertein Distance as the optimization metric between real and generated distributions. This makes the GAN more stable during training, improves the diversity of the generated images, and reduces the sensitivity to hyperparameters. 

A reason for these benefits is that the Wasserstein Distance is continuous and defined even when the two distributions are equal to 0 (unlike the KL or JS Divergences and many others). This means that we can get a meaningful gradient even when the two distributions are completely different.


## WGAN Results
### CIFAR10
![cifar10 image1](./readme_imgs/cifar10_img1.png)
![cifar10 image2](./readme_imgs/cifar10_img2.png)
![cifar10 image3](./readme_imgs/cifar10_img3.png)
![cifar10 image4](./readme_imgs/cifar10_img4.png)
![cifar10 image5](./readme_imgs/cifar10_img5.png)
![cifar10 image6](./readme_imgs/cifar10_img6.png)
![cifar10 image7](./readme_imgs/cifar10_img7.png)

### German Traffic Signs
![traffic data image1](./readme_imgs/traffic_img1.png)
![traffic data image2](./readme_imgs/traffic_img2.png)
![traffic data image3](./readme_imgs/traffic_img3.png)
![traffic data image4](./readme_imgs/traffic_img4.png)
![traffic data image5](./readme_imgs/traffic_img5.png)
![traffic data image6](./readme_imgs/traffic_img6.png)
![traffic data image7](./readme_imgs/traffic_img7.png)

### MNIST
![mnist image1](./readme_imgs/mnist_img1.png)
![mnist image2](./readme_imgs/mnist_img2.png)
![mnist image3](./readme_imgs/mnist_img3.png)
![mnist image4](./readme_imgs/mnist_img4.png)
![mnist image5](./readme_imgs/mnist_img5.png)
![mnist image6](./readme_imgs/mnist_img6.png)
![mnist image7](./readme_imgs/mnist_img7.png)

#### MNIST Loss Figures
![MNIST Losses](./readme_imgs/mnist_losses_figure.png)
![MNIST Loss Difference](./readme_imgs/mnist_diff_figure.png)

Loss difference is calculated as the difference between the loss from the real data and the loss from the generated data.


### Sources:
[Wasserstein GAN](https://arxiv.org/abs/1701.07875)
[Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)
[Improved Training of Wasserstein GANs Github Repo](https://github.com/igul222/improved_wgan_training)
[Read-Through: Wasserstein GAN](https://www.alexirpan.com/2017/02/22/wasserstein-gan.html)
