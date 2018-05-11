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

def scale_down(imgs):
    return imgs/255.

def scale_up(imgs):
    return imgs*255.

def cuda_if(tobj):
    if torch.cuda.is_available():
        tobj = tobj.cuda()
    return tobj

def save_imgs(fakes, save_root):
    fakes = fakes.squeeze()
    for i,img in enumerate(fakes):
        png = Image.fromarray(img)
        png.save(save_root+"_img"+str(i)+".png")

if __name__ == "__main__":

    save_root = "mnist"
    img_folder = save_root+"_imgs"
    max_epochs = int(1e7)
    n_critic = 6
    batch_size = 64
    disc_lr = 5e-4
    gen_lr = 1e-4
    clip_coef = .01
    z_size = 100
    convergence_bound = 1e-13
    model_type='conv'
    scale = True
    trainable_z = True
    disc_bnorm = True
    gen_bnorm = True
    process = False
    resume = False
    data_type = "mnist"

    if "traffic" == data_type:
        data_loc = "/home/grantsrb/machine_learning/datasets/traffic_sign_data"
        with open(data_loc+'/train.p', mode='rb') as f:
            dataset = pickle.load(f)
        imgs = dataset['features']
    elif "cifar10" == data_type:
        data_loc = "/home/grantsrb/machine_learning/datasets/cifar10"
        dataset = torchvision.datasets.CIFAR10(data_loc, train=True, download=True)
        imgs = np.asarray(dataset.train_data)
    elif "mnist" == data_type:
        data_loc = "/home/grantsrb/machine_learning/datasets/mnist"
        dataset = torchvision.datasets.MNIST(data_loc, train=True, download=True)
        imgs = np.asarray(dataset.train_data)
        imgs = imgs.reshape((imgs.shape[0], imgs.shape[1], imgs.shape[2], 1))
        imgs = np.pad(imgs, ((0,0),(2,2),(2,2),(0,0)), 'constant', constant_values=0)

    print("img shape:", imgs.shape)
    print("img max:", np.max(imgs))
    print("img min:", np.min(imgs))
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
        GAN = conv_gan.DCGAN

    gan = GAN(imgs.shape,z_size=z_size,trainable_z=trainable_z,disc_bnorm=disc_bnorm,gen_bnorm=gen_bnorm)
    gan = cuda_if(gan)
    trainer = Trainer(gan, gen_lr=gen_lr, disc_lr=disc_lr)
    if resume:
        trainer.load_model(save_root)

    perm = cuda_if(torch.randperm(len(imgs)).long())
    if model_type.lower() == "fc" or model_type.lower() == "dense":
        imgs = imgs.view(len(imgs), -1)
    #reals = imgs[:n_samples]
    counter = 0
    best_diff = 2
    d_losses = []
    g_losses = []

    for epoch in range(max_epochs):

        if (epoch % 100 == 0 and epoch < 1000) or (epoch % 500 == 0):
            temp_n_critic = 100
            n_samples = temp_n_critic*batch_size
        else:
            temp_n_critic = n_critic
            n_samples = temp_n_critic*batch_size

        if (counter+1)*n_samples > len(perm):
            counter = 0
            perm = cuda_if(torch.randperm(len(imgs)).long())
        idxs = perm[counter*n_samples:(counter+1)*n_samples]
        counter += 1
        reals = imgs[idxs]

        trainer.train(reals, batch_size=batch_size, clip_coef=clip_coef, n_critic=temp_n_critic)
        disc_loss, gen_loss = trainer.get_statistics(epoch)
        d_losses.append(disc_loss)
        g_losses.append(gen_loss)
        if epoch % 20 == 0:
            trainer.print_statistics(epoch)
            n_imgs = 8
            fakes = trainer.get_imgs(n_imgs)
            if scale:
                fakes = scale_up(fakes)
            if process:
                fakes = postprocess(fakes, mean, std)
            save_imgs(fakes.astype(np.uint8), save_root)
            if epoch % 1000 == 0:
                save_imgs(fakes.astype(np.uint8), img_folder+"/"+str(epoch)+"_"+save_root)
                np.save(save_root+"_d_losses.npy", np.asarray(d_losses))
                np.save(save_root+"_g_losses.npy", np.asarray(g_losses))
            trainer.save_model(save_root)
    
