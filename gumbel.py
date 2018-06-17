import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Gumbel_Net(nn.Module):
# Accept Z, put it into linear transformation,
# pass it through gumbel_softmax, expand dimension into image size and output it

    def __init__(self, num_generator, z_dim):
        super(Gumbel_Net, self).__init__()
        self.z_dim = z_dim
        self.f_dim = num_generator*self.z_dim

        mlp_layers = []
        mlp_layers.append(nn.Linear(self.z_dim+ self.f_dim, int((self.z_dim+self.f_dim)/2)))
        mlp_layers.append(nn.BatchNorm2d(int((self.z_dim+self.f_dim)/2)))
        mlp_layers.append(nn.ReLU(inplace=True))
        mlp_layers.append(nn.Linear(int((self.z_dim+self.f_dim)/2), self.z_dim))
        mlp_layers.append(nn.BatchNorm2d(self.z_dim))
        mlp_layers.append(nn.ReLU(inplace=True))
        mlp_layers.append(nn.Linear(self.z_dim, num_generator))
        self.mlp = nn.Sequential(*mlp_layers)

    def gumbel_softmax(self, logits, temperature, hard=False):
        """Sample from the Gumbel-Softmax distribution and optionally discretize.
        Args:
          logits: [batch_size, n_class] unnormalized log-probs
          temperature: non-negative scalar
          hard: if True, take argmax, but differentiate w.r.t. soft sample y
        Returns:
          [batch_size, n_class] sample from the Gumbel-Softmax distribution.
          If hard=True, then the returned sample will be one-hot, otherwise it will
          be a probabilitiy distribution that sums to 1 across classes
        """
        y = self.gumbel_softmax_sample(logits, temperature) ## (0.6, 0.2, 0.1,..., 0.11)
        if hard:
            k = logits.size(1) # k is numb of classes
            # y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)  ## (1, 0, 0, ..., 0)
            y_hard = torch.eq(y, torch.max(y, dim=1, keepdim=True)[0]).type_as(y)
            y = (y_hard - y).detach() + y
        return y

    def gumbel_softmax_sample(self, logits, temperature):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        noise = self.sample_gumbel(logits)
        y = (logits + noise) / temperature
        return F.softmax(y)

    def sample_gumbel(self, logits):
        """Sample from Gumbel(0, 1)"""
        noise = torch.rand(logits.size())
        eps = 1e-20
        noise.add_(eps).log_().neg_()
        noise.add_(eps).log_().neg_()
        return Variable(noise.float()).cuda()

    def forward(self, z, feature, imsize, temperature, hard): #z= batch x z_dim // #feature = batch x num_gen x 256*8*8

          # batch x num_gen x z_dim
        out = torch.cat([z,feature], dim=1)
        out = self.mlp(out)
        out_print = F.softmax(out)[:1]
        out = self.gumbel_softmax(out, temperature, hard) # batch x num_generator
        for_print = out.clone()
        out = out.unsqueeze(2).unsqueeze(2).unsqueeze(2) # batch x num_generator x 3 x imsize x imsize
        out = out.repeat(1, 1, 3, int(imsize), int(imsize))
        return out, for_print, out_print # batch x num_generator x 3 x imsize x imsize
