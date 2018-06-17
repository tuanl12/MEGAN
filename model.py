import torch
import numpy as np
import torch.nn as nn

class Generator(nn.Module):
    """Generator."""

    def __init__(self, image_size=64, z_dim=100, conv_dim=64, num_gen=10):
        super(Generator, self).__init__()
        self.num_gen = num_gen
        self.image_size = image_size
        self.z_dim = z_dim
        self.conv_dim = conv_dim
        repeat_num = int(np.log2(image_size)) - 3
        mult = 2 ** repeat_num
        # self.Gum = Gumbel_Net(self.num_gen, self.z_dim)

        self.all_layers = nn.ModuleList()
        self.all_layers2 = nn.ModuleList()
        self.sum_linear = nn.Linear(self.conv_dim*int(mult/2)*8*8, self.z_dim)
        self.relu = nn.ReLU(inplace=True)

        for j in range(self.num_gen):
            layers = []
            layers2 = []

            layers.append(nn.ConvTranspose2d(z_dim, conv_dim * mult, 4))
            layers.append(nn.BatchNorm2d(conv_dim * mult))
            layers.append(nn.ReLU())

            curr_dim = conv_dim * mult
            layers.append(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1))
            layers.append(nn.BatchNorm2d(int(curr_dim / 2)))
            layers.append(nn.ReLU())
            curr_dim = int(curr_dim / 2)

            for i in range(repeat_num-1):
                layers2.append(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1))
                layers2.append(nn.BatchNorm2d(int(curr_dim / 2)))
                layers2.append(nn.ReLU())
                curr_dim = int(curr_dim / 2)


            layers2.append(nn.ConvTranspose2d(curr_dim, 3, 4, 2, 1))
            layers2.append(nn.Tanh())
            self.main1 = nn.Sequential(*layers)
            self.main2 = nn.Sequential(*layers2)
            self.all_layers.append(self.main1)
            self.all_layers2.append(self.main2)


    def forward(self, z_, gum_t):

        z = z_.view(z_.size(0), z_.size(1), 1, 1)
        feature = self.all_layers[0](z).unsqueeze(1) # batch x 1 x 3 x 32 x 32
        for i in range(1, self.num_gen):  # from 1 to 9
            temp = self.all_layers[i](z).unsqueeze(1)
            feature = torch.cat([feature, temp], dim=1)

        out = self.all_layers2[0](feature[:,0]).unsqueeze(1)
        for i in range(1, self.num_gen):  # from 1 to 9
            temp = self.all_layers2[i](feature[:,i]).unsqueeze(1)
            out = torch.cat([out, temp], dim=1)
        feature = self.sum_linear(feature.contiguous().view(feature.size(0), self.num_gen, -1))
        # batch x num_gen x z_dim
        feature = self.relu(feature)
        feature = feature.contiguous().view(-1, self.num_gen * self.z_dim) # batch x n_gen*100

        return out, feature # batch x 3 x 32 x 32


class Discriminator(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, image_size=64, conv_dim=64):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, 4, 2, 1))
        layers.append(nn.LeakyReLU(0.2))

        repeat_num = int(np.log2(image_size)) - 3
        curr_dim = conv_dim
        for i in range(repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1))
            layers.append(nn.LeakyReLU(0.2))
            curr_dim = curr_dim * 2

        self.main = nn.Sequential(*layers)
        self.last = nn.Conv2d(curr_dim, 1, 4)

    def forward(self, x):
        temp = self.main(x) # curr_dim
        out = self.last(temp)
        return out.squeeze()
