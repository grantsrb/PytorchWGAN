import torchvision
from gan import GAN
from trainer import Trainer

def preprocess(imgs, mean, std):
    return (imgs-mean)/(std+1e-7)

def postprocess(remakes, mean, std):
    return remakes*(std+1e-7) + mean

if __name__ == "__main__":
    cifar = torchvision.datasets.CIFAR10("/Users/satchelgrant/Datasets/cifar10", train=True, download=True)
    imgs = cifar.train_data
    imgs = cuda_if(torch.FloatTensor(imgs.transpose((0,3,1,2))))
    if process:
        mean = imgs.mean()
        std = imgs.std()
        imgs = preprocess(imgs, mean, std)
    
    gan = GAN(imgs.shape, emb_size=emb_size)
    trainer = Trainer(gan, lr=lr, batch_size=batch_size, clip_coef=clip_coef, n_critic_loops=n_critic_loops)
    
    for epoch in range(n_epochs):
        trainer.train()
        trainer.print_statistics()
        if epoch % 10 == 0:
            n_imgs = 10
            trainer.save_imgs(save_root, n_imgs)
            trainer.save_model(save_root)
    
