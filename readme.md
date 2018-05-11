# Convolutional Wasserstein GAN
### May 11th, 2018

## Description
Generative Adversarial Networks are a deep learning model architecture that fabricate realistic data in the image of a real data set.

The basic setup of a GAN consists of two networks. One of the two networks, known as the discriminator, tries to distinguish between real and generated images. The other network, known as the generator, generates images with the goal of fooling the first network. 

Early GANs used an optimization method that sought to minimize the KL divergence between the generated data distribution and the real data distribution. The issue with this approach is that the KL divergence is not defined at points where one of the two distributions is zero. This causes problems in the optimization because in cases where the distance metric is undefined, the optimization cannot know how to update the functions. Many types of probability distance measures fall into this trap.

Wasserstein GANs were created to solve this problem. Wasserstein distance (aka Earth Mover distance) is defined at all points. Thus the Wassertein metric avoids the issue that the KL divergence (and many more divergences) fall into.

Since the Wasserstein metric is defined regardless of how well the discriminator is doing, there is less of a need to balance the training between the discriminator and the generator. The discriminator can be trained much faster than the generator network and the generator network will still be updated in the appropriate direction.

There are a few key differences to converting a KL divergence optimization to a Wasserstein distance optimization. First, the outputs from the discriminator should be raw (no softmax or log functions). The optimization for the discriminator should seek to maximize the difference between the discriminator outputs for the real distribution and the discriminator outputs for the fake distribution. The generator optmization seeks to maximize the outputs of the discriminator for the fake distribution.  
