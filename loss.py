import shlex

import torch
import torch.nn as nn


class Loss:
    def __init__(self, criterion='mse'):
        if criterion == 'ce':  # this doesn't work yet.
            self.criterion = nn.CrossEntropyLoss(reduction='sum')
        else:
            self.criterion = nn.MSELoss(reduction='sum')

    def __call__(self, x, x_hat, mu, logvar):
        recons_loss = self.criterion(x_hat, x)
        kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
        return recons_loss, kl_loss

class TwoStageLoss:
    def __init__(self):
        self.z_criterion = nn.MSELoss(reduction='sum')
        self.u_criterion = nn.MSELoss(reduction='sum')

    def z_loss(self, x, x_hat, z_mu, z_logvar):
        z_kl_loss = torch.mean(-0.5 * torch.sum(1 + z_logvar - z_mu ** 2 - z_logvar.exp(), dim=1), dim=0)
        z_recons_loss = self.z_criterion(x_hat, x)
        z_loss = z_recons_loss + z_kl_loss

        return z_recons_loss, z_kl_loss, z_loss


    def u_loss(self, z, z_hat, u_mu, u_logvar):
        u_kl_loss = torch.mean(-0.5 * torch.sum(1 + u_logvar - u_mu ** 2 - u_logvar.exp(), dim = 1), dim = 0)
        u_recons_loss = self.u_criterion(z_hat, z)
        u_loss = u_recons_loss + u_kl_loss

        return u_recons_loss, u_kl_loss, u_loss


