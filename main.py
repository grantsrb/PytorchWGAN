import torchvision
import torch
from gan import GAN
from trainer import Trainer

def preprocess(imgs, mean, std):
    return (imgs-mean)/(std+1e-7)

def postprocess(remakes, mean, std):
    return remakes*(std+1e-7) + mean

def cuda_if(tobj):
    if torch.cuda.is_available():
        tobj = tobj.cuda()
    return tobj

if __name__ == "__main__":

    save_root = "default"
    n_epochs = 100000
    n_critic = 5
    batch_size = 64
    lr = 5e-5
    clip_coef = .01
    emb_size = 200
    process = False

    cifar = torchvision.datasets.CIFAR10("/Users/satchelgrant/Datasets/cifar10", train=True, download=True)
    imgs = cifar.train_data
    imgs = cuda_if(torch.FloatTensor(imgs.transpose((0,3,1,2))))
    if process:
        mean = imgs.mean()
        std = imgs.std()
        imgs = preprocess(imgs, mean, std)
    
    gan = GAN(imgs.shape, emb_size=emb_size)
    gan = cuda_if(gan)
    trainer = Trainer(gan, lr=lr)
    perm = cuda_if(torch.randperm(len(imgs)).long())
    imgs = imgs.view(len(imgs), -1)
    n_samples = n_critic*batch_size
    counter = 0
    for epoch in range(n_epochs):

        if (counter+1)*n_samples > len(perm):
            counter = 0
            perm = cuda_if(torch.randperm(len(imgs)).long())
        idxs = perm[counter*n_samples:(counter+1)*n_samples]
        counter += 1

        reals = imgs[idxs]
        trainer.train(reals, batch_size=batch_size, clip_coef=clip_coef, n_critic=n_critic)
        trainer.print_statistics()
        if epoch % 10 == 0:
            n_imgs = 10
            trainer.save_imgs(save_root, n_imgs)
            trainer.save_model(save_root)
    
