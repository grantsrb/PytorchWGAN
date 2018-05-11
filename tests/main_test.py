import torchvision
import torch
import fc_gan
import conv_gan
import res_gan
from test_trainer import Trainer
from PIL import Image
import numpy as np
import pickle

def preprocess(imgs, mean, std):
    return (imgs-mean)/(std+1e-7)

def postprocess(remakes, mean, std):
    return remakes*(std+1e-7) + mean

def scale_down(imgs):
    return imgs/255.

def scale_up(imgs):
    return imgs*255.

def cuda_if(tobj):
    if torch.cuda.is_available():
        tobj = tobj.cuda()
    return tobj

def save_imgs(fakes, save_root):
    for i,img in enumerate(fakes):
        png = Image.fromarray(img)
        png.save(save_root+"_img"+str(i)+".png")

if __name__ == "__main__":

    save_root = "default"
    n_critic = 5
    batch_size = 64
    lr = 1e-3
    clip_coef = .01
    z_size = 100
    convergence_bound = 1e-7
    model_type='conv'
    scale = False
    trainable_z = True
    disc_bnorm = True
    gen_bnorm = True
    process = False
    resume = False
    torchvision_dataset = True
    cifar_loc = "/home/grantsrb/machine_learning/datasets/cifar10"
    mnist_loc = "/home/grantsrb/machine_learning/datasets/mnist"
    data_loc = mnist_loc
    #torchvision_dataset = False
    #data_loc = "/home/grantsrb/machine_learning/datasets/traffic_sign_data"

    if torchvision_dataset:
        dataset = torchvision.datasets.CIFAR10(data_loc, train=True, download=True)
        imgs = dataset.train_data
        print("img shape:", imgs.shape)
    else:
        with open(data_loc+'/train.p', mode='rb') as f:
            dataset = pickle.load(f)
        imgs = dataset['features']
    mean = float(imgs.mean())
    std = float(imgs.std())
    imgs = cuda_if(torch.FloatTensor(imgs.transpose((0,3,1,2))))
    if scale:
        imgs = scale_down(imgs)
    if process:
        imgs = preprocess(imgs, mean, std)
    
    if model_type.lower() == "fc" or model_type.lower() == "dense":
        GAN = fc_gan.GAN
    elif model_type.lower() == "conv":
        GAN = conv_gan.GAN
    elif model_type.lower() == "res":
        GAN = res_gan.GAN

    gan = GAN(imgs.shape,z_size=z_size,trainable_z=trainable_z,disc_bnorm=disc_bnorm,gen_bnorm=gen_bnorm)
    gan = cuda_if(gan)
    trainer = Trainer(gan, lr=lr)
    if resume:
        trainer.load_model(save_root)

    perm = cuda_if(torch.randperm(len(imgs)).long())
    if model_type.lower() == "fc" or model_type.lower() == "dense":
        imgs = imgs.view(len(imgs), -1)
    #reals = imgs[:n_samples]
    epoch = 0
    counter = 0
    diff = 100
    best_diff = 2

    perm = cuda_if(torch.randperm(len(imgs)).long())
    while True:
        counter += 1
        epoch += 1
        if (counter+1)*batch_size >= len(perm):
            perm = cuda_if(torch.randperm(len(imgs)).long())
            counter = 0
        idxs = perm[counter*batch_size:(counter+1)*batch_size]
        reals = imgs[idxs]
        fakes = gan.generate(batch_size)
        trainer.train(reals, fakes.data, clip_coef=clip_coef)
        
