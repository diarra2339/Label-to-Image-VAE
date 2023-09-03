import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import datetime
torch.manual_seed(0)

from loss import TwoStageLoss, Loss
from history import History
from utils import get_device


class Trainer:
    def __init__(self, criterion='mse'):
        self.loss_func = Loss(criterion=criterion)
        self.history = History()

    def train(self, model, train_loader, epochs, lr=1e-4, optimizer=None, gamma=None, ckpt=2):
        device = get_device()
        model = model.to(device)
        optimizer = Adam(model.parameters(), lr=lr) if optimizer is None else optimizer
        gamma = 1 if gamma is None else gamma

        for epoch in range(1, epochs+1):
            print('Epoch {} ...'.format(epoch))
            recons_losses, kl_losses = [], []
            progress_bar = tqdm(train_loader)
            for images, captions, lengths in progress_bar:
                images, captions, lengths = images.to(device), captions.to(device), lengths.to(device)

                recons, mu, logvar = model(images, captions, lengths)
                recons_loss, kl_loss = self.loss_func(x=images, x_hat=recons, mu=mu, logvar=logvar)
                loss = recons_loss + gamma * kl_loss
                recons_losses.append(float(recons_loss))
                kl_losses.append(float(kl_loss))

                # backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                progress_bar.set_description(desc='rec_loss: {:.2f}, kl_loss: {:.2f}'.format(recons_loss, kl_loss))

            # logs
            avg_recons, avg_kl = np.mean(recons_losses), np.mean(kl_losses)
            print('Recons loss: {}, KL loss: {}'.format(avg_recons, avg_kl))
            self.history.save({'recons_loss': avg_recons, 'kl_loss': avg_kl})

            # checkpoint ?
            if epoch % ckpt == 0 or epoch == epochs:
                date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
                path = 'epoch' + str(epoch) + '-' + date_time + '.ckpt'
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, path)



    def train_vae(self, model, train_loader, epochs, lr=1e-4, optimizer=None, gamma=None, ckpt=2):
        device = get_device()
        model = model.to(device)
        optimizer = Adam(model.parameters(), lr=lr) if optimizer is None else optimizer
        gamma = 1 if gamma is None else gamma

        for epoch in range(1, epochs + 1):
            print('Epoch {} ...'.format(epoch))
            recons_losses, kl_losses = [], []
            progress_bar = tqdm(train_loader)
            for images, captions, lengths in progress_bar:
                images, captions, lengths = images.to(device), captions.to(device), lengths.to(device)

                recons, mu, logvar = model(images)
                recons_loss, kl_loss = self.loss_func(x=images, x_hat=recons, mu=mu, logvar=logvar)
                loss = recons_loss + gamma * kl_loss
                recons_losses.append(float(recons_loss))
                kl_losses.append(float(kl_loss))

                # backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                progress_bar.set_description(desc='rec_loss: {:.2f}, kl_loss: {:.2f}'.format(recons_loss, kl_loss))

            # logs
            avg_recons, avg_kl = np.mean(recons_losses), np.mean(kl_losses)
            print('Recons loss: {}, KL loss: {}'.format(avg_recons, avg_kl))
            self.history.save({'recons_loss': avg_recons, 'kl_loss': avg_kl})

            # checkpoint ?
            if epoch % ckpt == 0 or epoch == epochs:
                date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
                path = 'epoch' + str(epoch) + '-' + date_time + '.ckpt'
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, path)


    def train_two_stage(self, model, train_loader, val_loader, epochs, lr=1e-4, ckpt=2):
        device = get_device()
        model = model.to(device)
        z_optimizer = Adam(model.cvae.parameters(), lr=lr)
        u_optimizer = Adam(model.small_vae.parameters(), lr=lr)
        loss_func = TwoStageLoss()

        for epoch in range(1, epochs+1):
            print('Epoch {} ...'.format(epoch))
            z_recons_losses, z_kl_losses = [], []
            u_recons_losses, u_kl_losses = [], []
            progress_bar = tqdm(train_loader)

            for images, captions, lengths in progress_bar:
                images, captions, lengths = images.to(device), captions.to(device), lengths.to(device)

                x_hat, z, z_mu, z_logvar = model.z_forward(images, captions, lengths)
                z = z.detach()  # remove from the computation graph of the CVAE, treat z as a new input
                z_hat, u_mu, u_logvar = model.u_forward(z)

                # for the CVAE
                z_recons_loss, z_kl_loss, z_loss = loss_func.z_loss(x=images, x_hat=x_hat, z_mu=z_mu, z_logvar=z_logvar)
                z_recons_losses.append(float(z_recons_loss))
                z_kl_losses.append(float(z_kl_loss))

                # for the small VAE
                u_recons_loss, u_kl_loss, u_loss = loss_func.u_loss(z=z, z_hat=z_hat, u_mu=u_mu, u_logvar=u_logvar)
                u_recons_losses.append(float(u_recons_loss))
                u_kl_losses.append(float(u_kl_loss))

                # backprop
                z_optimizer.zero_grad()
                u_optimizer.zero_grad()
                z_loss.backward()
                u_loss.backward()
                z_optimizer.step()
                u_optimizer.step()

                progress_bar.set_description(desc='z_rec_loss: {:.2f}, z_kl_loss: {:.2f}, u_rec_loss: {:.2f}, u_kl_loss: {:.2f}'.format(z_recons_loss, z_kl_loss, u_recons_loss, u_kl_loss))

            # logs
            z_avg_recons, z_avg_kl = np.mean(z_recons_losses), np.mean(z_kl_losses)
            u_avg_recons, u_avg_kl = np.mean(u_recons_losses), np.mean(u_kl_losses)
            print('Z_Recons loss: {}, Z_KL loss: {}'.format(z_avg_recons, z_avg_kl))
            print('U_Recons loss: {}, U_KL loss: {}'.format(u_avg_recons, u_avg_kl))
            self.history.save({'z_recons_loss': z_avg_recons, 'z_kl_loss': z_avg_kl})
            self.history.save({'u_recons_loss': u_avg_recons, 'u_kl_loss': u_avg_kl})

            # Validation step
            print('Validation step ...')
            with torch.no_grad():
                val_recons_losses, val_kl_losses = [], []
                for val_images, val_captions, val_lengths in tqdm(val_loader):
                    val_images, val_captions, val_lengths = val_images.to(device), val_captions.to(device), val_lengths.to(device)
                    x_hat, _, val_mu, val_logvar = model.z_forward(val_images, val_captions, val_lengths)
                    val_recons_loss, val_kl_loss, val_loss = loss_func.z_loss(x=val_images, x_hat=x_hat, z_mu=val_mu, z_logvar=val_logvar)
                    val_recons_losses.append(float(val_recons_loss))
                    val_kl_losses.append(float(val_kl_loss))
                print('val_rec_loss: {:.2f}; val_kl_loss: {:.2f}'.format(np.mean(val_recons_losses), np.mean(val_kl_losses)))

            # checkpoint ?
            if epoch % ckpt == 0 or epoch == epochs:
                date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
                path = 'epoch' + str(epoch) + '-' + date_time + '.ckpt'
                # path = 'epoch' + str(epoch) + '-' +  + '.ckpt'
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict()
                }, path)


    def train_caption_encoder(self, model, train_loader, val_loader, epochs, lr=1e-3, ckpt=5):
        device = get_device()
        model = model.to(device)
        loss_func = nn.MSELoss(reduction='sum')
        optimizer = Adam(params=model.parameters(), lr=lr)

        for epoch in range(1, epochs + 1):
            print('Epoch {} ...'.format(epoch))
            losses = []
            for captions, lengths, annots in tqdm(train_loader):
                captions, lengths, annots = captions.to(device), lengths.to(device), annots.to(device)
                preds = model(captions, lengths)
                loss = loss_func(preds, annots)
                losses.append(float(loss))

                # backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            avg_loss = np.mean(losses)

            print('Validation step ...')
            with torch.no_grad():
                val_losses = []
                for val_captions, val_lengths, val_annots in tqdm(val_loader):
                    val_captions, val_lengths, val_annots = val_captions.to(device), val_lengths.to(device), val_annots.to(device)
                    val_preds = model(val_captions, val_lengths)
                    val_loss = loss_func(val_preds, val_annots)
                    val_losses.append(float(val_loss))
                avg_val_loss = np.mean(val_losses)

            # logs
            print('train_loss: {:.2f}; val-loss: {:.2f}'.format(avg_loss, avg_val_loss))
            self.history.save({'loss': avg_loss, 'val_loss': avg_val_loss})

            # checkpoint ?
            if epoch % ckpt == 0 or epoch == epochs:
                date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
                path = 'epoch' + str(epoch) + '-' + date_time + '.ckpt'
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict()
                }, path)



