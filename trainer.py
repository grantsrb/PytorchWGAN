import torch.optim as optim
from torch.autograd import Variable
import torch
import numpy as np
from PIL import Image

class Trainer:
    def cuda_if(self, tobj):
        if torch.cuda.is_available():
            tobj = tobj.cuda()
        return tobj

    def __init__(self, gan, lr=1e-5):
        self.gan = gan
        self.disc_optim = optim.RMSprop(gan.discriminator.parameters(), lr=lr)
        self.gen_optim = optim.RMSprop(gan.generator.parameters(), lr=lr)
        self.avg_real = 0
        self.avg_fake = 0

    def train(self, reals, batch_size=64, clip_coef=.01, n_critic=5):
        fakes = self.gan.generate(len(reals))
        fakes = fakes.data
        perm = self.cuda_if(torch.randperm(len(reals)).long())

        # Discrim Updating
        for i in range(n_critic):
            self.disc_optim.zero_grad()

            idxs = perm[i*batch_size:(i+1)*batch_size]
            rs = Variable(reals[idxs])
            fs = Variable(fakes[idxs])
            r_loss = self.gan.discriminate(rs).mean()
            f_loss = self.gan.discriminate(fs).mean()
            disc_loss = f_loss - r_loss # Seek to maximize r_loss
            disc_loss.backward()
            self.disc_optim.step()

            self.gan.clip_ds_ps(clip_coef) # Wasserstein clipping maneuver

            # Tracking the losses
            self.avg_real = .99*self.avg_real + .01*r_loss.data[0]
            self.avg_fake = .99*self.avg_fake + .01*f_loss.data[0]

        # Generator Updating
        self.gen_optim.zero_grad()
        self.fakes = self.gan.generate(batch_size) 
        f_loss = -self.gan.discriminate(self.fakes).mean()
        f_loss.backward()
        self.gen_optim.step()

    def print_statistics(self):
        print("Avg Real:",self.avg_real, "– Avg Fake:",self.avg_fake,
                           "– Diff:",abs(self.avg_real-self.avg_fake))

    def save_model(self, save_root):
        torch.save(self.gan.state_dict(), save_root+"_gan.p")
        torch.save(self.disc_optim.state_dict(), save_root+"_d_optim.p")
        torch.save(self.gen_optim.state_dict(), save_root+"_g_optim.p")

    def save_imgs(self, save_root, n_imgs=10):
        imgs = self.fakes.data.cpu().numpy()
        imgs = imgs[:n_imgs]
        imgs = imgs.reshape((-1, *self.gan.img_shape[-3:])).transpose((0,2,3,1)).astype(np.uint8)
        for i,img in enumerate(imgs):
            png = Image.fromarray(img)
            png.save(save_root+"_img"+str(i)+".png")
        


            

        
