import os
import time
import copy
import torch
import datetime
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image

from msssim import *
from model import Generator
from model import Discriminator
from inception_score import *
from gumbel import *


class Solver(object):
    
    def __init__(self, data_loader, config):
        
        # Data loader
        self.data_loader = data_loader
        
        # Model hyper-parameters
        self.dataset = config.dataset
        self.z_dim = config.z_dim
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.image_size = config.image_size
        self.lambda_gp = config.lambda_gp
        self.num_gen = config.num_gen
        self.ms_num_image = config.ms_num_image

        # Training settings
        self.total_step = config.total_step
        self.batch_size = config.batch_size
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.d_iters = config.d_iters
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.gum_t = config.gum_temp
        self.gum_orig = config.gum_orig
        self.gum_t_decay = config.gum_temp_decay
        self.step_t_decay = config.step_t_decay
        self.start_anneal = config.start_anneal
        self.min_t = config.min_temp
        self.use_tensorboard = config.use_tensorboard
        self.pretrained_model = config.pretrained_model
        
        # Test settings
        self.test_size = config.test_size
        self.test_model = config.test_model
        self.test_ver = config.test_ver
        self.version = config.version
        self.result_path = os.path.join(config.result_path, self.version)
        self.nrow = config.nrow
        self.ncol = config.ncol
        
        # Step size
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.score_epoch = config.score_epoch
        self.score_start = config.score_start

        #load-balancing
        self.load_balance = config.load_balance
        self.balance_weight = config.balance_weight
        self.matching_weight = config.matching_weight
        
        # Path
        self.log_path = os.path.join(config.log_path, self.version)
        self.sample_path = os.path.join(config.sample_path, self.version)
        self.model_save_path = os.path.join(config.model_save_path, self.version)
        self.model_save_path_test = config.model_save_path

        
        self.build_model()

        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()


    def build_model(self):
        
        # Define the generator and discriminator

        #Generator will output batch x num_gen x 3 x imsize x imsize
        self.G = Generator(self.image_size, self.z_dim, self.g_conv_dim, self.num_gen).cuda()
        self.D = Discriminator(self.image_size, self.d_conv_dim).cuda()
        self.Gum = Gumbel_Net(self.num_gen, self.z_dim).cuda()


        # Loss and optimizer
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.gum_optimizer = torch.optim.Adam(self.Gum.parameters(), self.d_lr, [self.beta1, self.beta2], weight_decay=0.1)

        # print networks
        print(self.G)
        print(self.D)
        print(self.Gum)

    def many_to_one(self, z, many_images, gumbel_out, num_gen):
        if num_gen == 1:
            gumbel_out = Variable(torch.ones(z.size(0), 1, 3, self.image_size, self.image_size)).cuda()

        gumbel_images = torch.sum(torch.mul(many_images, gumbel_out), dim=1)  # batch x 3 x imsize x imsize
        return gumbel_images

    def load_pretrained_model(self):
        self.G.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_G.pth'.format(self.pretrained_model))))
        self.D.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_D.pth'.format(self.pretrained_model))))
        self.Gum.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_Gum.pth'.format(self.pretrained_model))))
        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def tensor2var(self, x, grad=False):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, requires_grad=grad)
    
    def var2tensor(self, x):
        return x.data.cpu()

    def var2numpy(self, x):
        return x.data.cpu().numpy()

    def denorm(self, x):
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def reset_grad(self):
        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()
        self.gum_optimizer.zero_grad()
    
    def train(self):

        # Data iterator
        data_iter = iter(self.data_loader)
        step_per_epoch = len(self.data_loader)

        score_step = self.score_epoch * step_per_epoch
        score_start = self.score_start * step_per_epoch
        start_anneal = self.start_anneal * step_per_epoch
        step_t_decay = self.step_t_decay * step_per_epoch

        self.balance_w_start = copy.deepcopy(self.balance_weight)

        # Fixed inputs for debugging
        real_images, _ = next(data_iter)
        save_image(self.denorm(real_images), os.path.join(self.sample_path, 'real.png')) 
        fixed_z = self.tensor2var(torch.randn(self.batch_size, self.z_dim))
        
        # Start with trained model
        if self.pretrained_model:
            start = self.pretrained_model + 1
        else:
            start = 0
        
        # Start time
        start_time = time.time()
        for step in range(start, self.total_step):
            
            # ================== Train D ================== #
            self.D.train()
            self.G.train()
            self.Gum.train()

            for i in range(self.d_iters):
                # Fetch real images
                try:
                    real_images, _ = next(data_iter)
                except:
                    data_iter = iter(self.data_loader)
                    real_images, _ = next(data_iter)

                # Compute loss with real images
                real_images = self.tensor2var(real_images)
                d_out_real = self.D(real_images)
                d_loss_real = - torch.mean(d_out_real)


                # apply Gumbel Softmax
                z = self.tensor2var(torch.randn(real_images.size(0), self.z_dim))
                many_images, feature = self.G(z, self.gum_t) #batch x num_gen x 3 x imsize x imsize
                gumbel_out, _, _ = self.Gum(z, feature, self.image_size, self.gum_t, True)  # batch x num_gen x 3 x imsize x imsize
                gumbel_images = self.many_to_one(z, many_images, gumbel_out, self.num_gen)

                # Compute loss with fake images
                d_out_fake = self.D(gumbel_images)
                d_loss_fake = torch.mean(d_out_fake)
                
                # Backward + Optimize
                d_loss = d_loss_real + d_loss_fake
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()
                
                # Compute gradient penalty
                alpha = torch.rand(real_images.size(0), 1, 1, 1).cuda().expand_as(real_images)
                interpolated = Variable(alpha * real_images.data + (1 - alpha) * gumbel_images.data, requires_grad=True)
                out = self.D(interpolated)
                
                grad = torch.autograd.grad(outputs=out,
                                           inputs=interpolated,
                                           grad_outputs=torch.ones(out.size()).cuda(),
                                           retain_graph=True,
                                           create_graph=True,
                                           only_inputs=True)[0]
                
                grad = grad.view(grad.size(0), -1)
                grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
                d_loss_gp = torch.mean((grad_l2norm - 1)**2)
                
                # Backward + Optimize
                d_loss = self.lambda_gp * d_loss_gp
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()
                
            # ================== Train G ================== #

            # Create random noise
            z = self.tensor2var(torch.randn(self.batch_size, self.z_dim))

            # Apply gumbel
            many_images, feature = self.G(z, self.gum_t)  # batch x num_gen x 3 x imsize x imsize
            gumbel_out, softmax_out, logit_dist = self.Gum(z, feature, self.image_size, self.gum_t, True)  # batch x num_gen x 3 x imsize x imsize
            gumbel_images = self.many_to_one(z, many_images, gumbel_out, self.num_gen)

            out = self.D(gumbel_images)
            g_out_fake = torch.mean(out)
            
            # Backward + Optimize
            g_loss_fake = - torch.mean(g_out_fake)
            self.reset_grad()
            g_loss_fake.backward()
            self.g_optimizer.step()
            self.gum_optimizer.step()


            # ===================Train Gumbel ===================#
            if self.load_balance == True:
                balance_weight = self.balance_weight

                z = self.tensor2var(torch.randn(self.batch_size, self.z_dim))
                many_images, feature = self.G(z, self.gum_t)  # batch x num_gen x 3 x imsize x imsize
                gumbel_out, softmax_out, logit_dist = self.Gum(z, feature, self.image_size, self.gum_t,
                                                               True)  # batch x num_gen x 3 x imsize x imsize

                target = Variable(torch.ones(self.num_gen)).cuda().float() / self.num_gen
                dist=softmax_out.sum(dim=0)/softmax_out.sum()
                balance_loss = F.mse_loss(dist, target) * balance_weight

                self.reset_grad()
                balance_loss.backward()
                self.gum_optimizer.step()

            # Print out log info
            if (step+1) % self.log_step == 0:
                if self.load_balance == True:
                    elapsed = time.time() - start_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))
                    print("Elapsed [{}], G_step [{}/{}], D_step[{}/{}], d_out_real: {:.4f}, "
                          "d_out_fake: {:.4f}, g_out_fake: {:.4f}, load_balance: {:.4f}".
                          format(elapsed, step+1, self.total_step, (step+1) * (self.d_iters),
                                 self.total_step * self.d_iters, d_out_real.data[0],
                                 d_out_fake.data[0], g_out_fake.data[0], balance_loss.data[0]))
                    print("Gumbel choice for 1 instances : ", softmax_out.max(dim=1)[1].data[0])
                    print("Logit choice (underlying distribution :", logit_dist.data[0])
                else:
                    elapsed = time.time() - start_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))
                    print("Elapsed [{}], G_step [{}/{}], D_step[{}/{}], d_out_real: {:.4f}, "
                          "d_out_fake: {:.4f}, g_out_fake: {:.4f}".
                          format(elapsed, step + 1, self.total_step, (step + 1) * (self.d_iters),
                                 self.total_step * self.d_iters, d_out_real.data[0],
                                 d_out_fake.data[0], g_out_fake.data[0]))
                    print("Gumbel choice for 1 instances : ", softmax_out.max(dim=1)[1].data[0])
                    print("Logit choice (underlying distribution :", logit_dist.data[0])


            # Sample images

            if (step+1) % self.sample_step == 0:
                many_images, feature = self.G(z, self.gum_t)  # batch x num_gen x 3 x imsize x imsize
                gumbel_out, _, _ = self.Gum(z, feature, self.image_size, self.gum_t, True)  # batch x num_gen x 3 x imsize x imsize
                gumbel_images = self.many_to_one(z, many_images, gumbel_out, self.num_gen)  # batch x num_gen x 3 x imsize x imsize

                save_image(self.denorm(gumbel_images.data),
                    os.path.join(self.sample_path, '{}_fake.png'.format(step+1)))

            
            # Save check points and inception scores

            if (step+1)>=score_start and (step+1) % score_step==0: # start from 25
            # if score_start == 0:
                torch.save(self.G.state_dict(),
                           os.path.join(self.model_save_path, '{}_G.pth'.format(step + 1)))
                torch.save(self.D.state_dict(),
                           os.path.join(self.model_save_path, '{}_D.pth'.format(step + 1)))
                torch.save(self.Gum.state_dict(),
                           os.path.join(self.model_save_path, '{}_Gum.pth'.format(step + 1)))

                # uncomment for automatic calculation of inception score and msssim
                # if self.dataset == 'cifar':
                #     print("calculating inception score for step %d...)" % (step + 1))
                #     print_inception_score(self.G, self.Gum, self.z_dim, self.num_gen, self.image_size, self.gum_t)
                # if self.dataset == 'CelebA' or self.dataset=='LSUN':
                #     print("calculating MS-SSIM score for step %d...)" % (step + 1))
                #     print_msssim_score(self.G, self.z_dim, self.ms_num_image, self.image_size, self.gum_t)

            # temperature reduction
            if (step+1) >= start_anneal and (step+1) % (step_t_decay) == 0 and self.gum_t > 0.01:
                self.gum_t = self.gum_orig * exp(-self.gum_t_decay*((step+1)-start_anneal))
                self.gum_t = max(self.gum_t,self.min_t)
                print('self.gum_t changed by : ', exp(-self.gum_t_decay*((step+1)-start_anneal)),' now temperature : ', self.gum_t )

            if (step+1) >= start_anneal and (step+1) % (step_t_decay) == 0:
                self.balance_weight = self.balance_w_start * exp(-self.gum_t_decay*((step+1)-start_anneal))
                self.balance_weight = max(self.balance_weight,0.001)
                print('self.balance_weight changed by : ', exp(-self.gum_t_decay*((step+1)-start_anneal)),' now balance : ', self.balance_weight )
