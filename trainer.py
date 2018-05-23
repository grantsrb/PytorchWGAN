import torch.optim as optim
from torch.autograd import Variable
import torch.autograd as autograd
import torch
import numpy as np

class Trainer:
    def cuda_if(self, tobj):
        if torch.cuda.is_available():
            tobj = tobj.cuda()
        return tobj

    def __init__(self, gan, gen_lr=5e-5, disc_lr=5e-5, grad_norm=False):
        self.gan = gan
        self.disc_optim = optim.RMSprop(gan.discriminator.parameters(), lr=disc_lr)
        self.gen_optim = optim.RMSprop(gan.generator.parameters(), lr=gen_lr)
        self.avg_real = 0
        self.avg_fake = 0
        self.grad_norm = grad_norm

    def get_losses(self, reals, fakes, virtual_batch):
        if virtual_batch is not None:
            start_idx = len(virtual_batch)
            x = torch.cat([virtual_batch, reals, fakes], dim=0)
        else:
            start_idx = 0
            x = torch.cat([reals, fakes], dim=0)
        loss = self.gan.discriminate(x)
        mid_idx = start_idx + len(reals)
        r_loss = loss[start_idx:mid_idx]
        f_loss = loss[mid_idx:]
        return r_loss, f_loss

    def get_grad_loss(self, reals, fakes, virtual_batch, lambda_):
        epsilon = self.cuda_if(torch.randn(len(reals)))
        combo = reals.data.permute(1,2,3,0)*epsilon + fakes.data.permute(1,2,3,0)*(1-epsilon)
        combo = Variable(combo.permute(3,0,1,2), requires_grad=True)
        if virtual_batch is not None:
            x = torch.cat([Variable(virtual_batch), combo], dim=0)
            start_idx = len(virtual_batch)
        else:
            x = combo
            start_idx = 0
        combo_loss = self.gan.discriminate(x)[start_idx:].sum()
        grads = autograd.grad(outputs=combo_loss, inputs=combo, create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradnorms = (grads.view(len(reals), -1).norm(2, dim=-1)-1).pow(2)
        return lambda_*gradnorms

    def train(self, reals, batch_size=64, clip_coef=.01, n_critic=5, virtual_batch=None):
        perm = self.cuda_if(torch.randperm(len(reals)).long())

        # Discrim Updating
        for i in range(n_critic):
            fakes = self.gan.generate(batch_size)
            self.disc_optim.zero_grad()

            idxs = perm[i*batch_size:(i+1)*batch_size]
            r_loss, f_loss = self.get_losses(Variable(reals[idxs]), Variable(fakes.data), virtual_batch)

            disc_loss = f_loss.mean() - r_loss.mean() # Seek to maximize r_loss
            disc_loss.backward()
            self.disc_optim.step()

            self.gan.clip_ds_ps(clip_coef) # Wasserstein clipping maneuver

            # Tracking the losses
            self.avg_real = .99*self.avg_real + .01*float(r_loss.data[0])
            self.avg_fake = .99*self.avg_fake + .01*float(f_loss.data[0])

        # Generator Updating
        self.gen_optim.zero_grad()
        self.fakes = self.gan.generate(batch_size) 
        f_loss = -self.gan.discriminate(self.fakes).mean()
        f_loss.backward()
        self.gen_optim.step()

    def grad_based_train(self, reals, batch_size=64, n_critic=5, lambda_=10, virtual_batch=None):
        perm = self.cuda_if(torch.randperm(len(reals)).long())

        # Discrim Updating
        for i in range(n_critic):
            fakes = self.gan.generate(batch_size)

            idxs = perm[i*batch_size:(i+1)*batch_size]
            r_loss, f_loss = self.get_losses(Variable(reals[idxs]), Variable(fakes.data), virtual_batch)

            grad_loss = self.get_grad_loss(reals[idxs], fakes.data, virtual_batch, lambda_)

            step = f_loss.squeeze() - r_loss.squeeze() + grad_loss.squeeze()
            disc_loss = step.mean()
            self.disc_optim.zero_grad()
            disc_loss.backward()
            self.disc_optim.step()

            # Tracking the losses
            self.avg_real = .99*self.avg_real + .01*float(r_loss.data.mean())
            self.avg_fake = .99*self.avg_fake + .01*float(f_loss.data.mean())

        # Generator Updating
        self.gen_optim.zero_grad()
        self.fakes = self.gan.generate(batch_size) 
        f_loss = -self.gan.discriminate(self.fakes).mean()
        f_loss.backward()
        self.gen_optim.step()

    def get_statistics(self, epoch):
        return self.avg_real, self.avg_fake
        
    def print_statistics(self, epoch):
        diff = abs(self.avg_real-self.avg_fake)
        print("Epoch:", epoch, "– Avg Real:",self.avg_real, "– Avg Fake:",self.avg_fake,
                           "– Diff:",diff)
        return diff

    def save_model(self, save_root):
        torch.save(self.gan.state_dict(), save_root+"_gan.p")
        torch.save(self.disc_optim.state_dict(), save_root+"_d_optim.p")
        torch.save(self.gen_optim.state_dict(), save_root+"_g_optim.p")

    def load_model(self, save_root):
        self.gan.load_state_dict(torch.load(save_root+"_gan.p"))
        self.disc_optim.load_state_dict(torch.load(save_root+"_d_optim.p"))
        self.gen_optim.load_state_dict(torch.load(save_root+"_g_optim.p"))

    def get_imgs(self, n_imgs=10):
        imgs = self.fakes.data.cpu().numpy()
        imgs = imgs[:n_imgs]
        imgs = imgs.reshape((-1, *self.gan.img_shape[-3:])).transpose((0,2,3,1))
        return imgs
        
