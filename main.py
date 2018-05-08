import torchvision
import torch
import fc_gan
import conv_gan
from trainer import Trainer
from PIL import Image
import numpy as np
import pickle

def preprocess(imgs, mean, std):
    return (imgs-mean)/(std+1e-7)

def postprocess(remakes, mean, std):
    return remakes*(std+1e-7) + mean

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
    lr = 5e-5
    clip_coef = .01
    z_size = 300
    convergence_bound = .0001
    model_type='conv'
    trainable_z = True
    disc_bnorm = True
    gen_bnorm = True
    process = False
    torchvision_dataset = False
    #data_loc = "/home/grantsrb/machine_learning/datasets/cifar10"
    data_loc = "/home/grantsrb/machine_learning/datasets/traffic_sign_data"

    if torchvision_dataset:
        dataset = torchvision.datasets.CIFAR10(data_loc, train=True, download=True)
        imgs = dataset.train_data
    else:
        with open(data_loc+'/train.p', mode='rb') as f:
            dataset = pickle.load(f)
        imgs = dataset['features']
    mean = float(imgs.mean())
    std = float(imgs.std())
    imgs = cuda_if(torch.FloatTensor(imgs.transpose((0,3,1,2))))
    if process:
        imgs = preprocess(imgs, mean, std)
    
    if model_type.lower() == "fc" or model_type.lower() == "dense":
        GAN = fc_gan.GAN
    else:
        GAN = conv_gan.GAN
    gan = GAN(imgs.shape,z_size=z_size,trainable_z=trainable_z,disc_bnorm=disc_bnorm,gen_bnorm=gen_bnorm)
    gan = cuda_if(gan)
    trainer = Trainer(gan, lr=lr)
    perm = cuda_if(torch.randperm(len(imgs)).long())
    n_samples = n_critic*batch_size
    if model_type.lower() == "fc" or model_type.lower() == "dense":
        imgs = imgs.view(len(imgs), -1)
    #reals = imgs[:n_samples]
    epoch = 0
    counter = 0
    diff = 100
    best_diff = 2
    while diff > convergence_bound or epoch < 100:

        if (counter+1)*n_samples > len(perm):
            counter = 0
            perm = cuda_if(torch.randperm(len(imgs)).long())
        idxs = perm[counter*n_samples:(counter+1)*n_samples]
        counter += 1
        epoch += 1
        reals = imgs[idxs]

        trainer.train(reals, batch_size=batch_size, clip_coef=clip_coef, n_critic=n_critic)
        diff = trainer.print_statistics(epoch)
        if epoch % 10 == 0:
            n_imgs = 4
            fakes = trainer.get_imgs(n_imgs)
            if process:
                fakes = postprocess(fakes, mean, std)
            if diff < best_diff:
                save_imgs(fakes.astype(np.uint8), save_root+"_best")
                best_diff = diff
            save_imgs(fakes.astype(np.uint8), save_root)
            trainer.save_model(save_root)
    
