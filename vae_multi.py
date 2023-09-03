import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm
import datetime

from model import LinVAE, CVAE
from utils import get_device
from history import History


# Two-stage CVAE with multi-task learning
class CVAEMulti3(nn.Module):
    def __init__(self, im_size, channel_dims, im_enc_dim, vocab_size, embed_dim, hidden_size, latent_dim):
        super(CVAEMulti3, self).__init__()
        self.history = History()

        self.latent_dim = latent_dim

        self.cvae = CVAE(im_size=im_size, enc_dim=im_enc_dim, vocab_size=vocab_size, embed_dim=embed_dim,
                         hidden_size=hidden_size, latent_dim=latent_dim, channel_dims=channel_dims)
        self.small_vae = LinVAE(latent_dim=latent_dim, num_layers=3)
        self.multi_class_fc = nn.Linear(in_features=im_enc_dim + 2*hidden_size, out_features=40)

    def z_forward(self, encoded_images, encoded_captions):
        z, z_mu, z_logvar = self.cvae.encode(encoded_images=encoded_images, encoded_captions=encoded_captions)
        x_hat = self.cvae.decode(z, encoded_captions=encoded_captions)
        return x_hat, z, z_mu, z_logvar

    def u_forward(self, z):
        z_hat, u_mu, u_logvar = self.small_vae(z)
        return z_hat, u_mu, u_logvar

    def multi_forward(self, encoded_images, encoded_captions):
        annot = self.multi_class_fc(torch.cat([encoded_images, encoded_captions], dim=1))
        return annot

    def sample_images(self, captions, lengths):
        u = torch.randn(captions.shape[0], self.latent_dim)
        z = self.small_vae.decode(u)
        encoded_captions = self.cvae.cap_encoder(captions, lengths)
        recons = self.cvae.decode(z, encoded_captions)
        return recons

    def accuracy(self, preds, targets):
        # The proportion of labels the model predicted right (as in same sign as the real label)
        temp = preds * targets > 0
        return torch.sum(temp, dim=1) / temp.shape[1]

    def train_step(self, images, captions, lengths, annotations):
        encoded_images = self.cvae.im_encoder(images)  # batch_size x enc_dim
        encoded_captions = self.cvae.cap_encoder(captions, lengths)  # batch_size x 2 * hidden_dim

        x_hat, z, z_mu, z_logvar = self.z_forward(encoded_images, encoded_captions)
        z = z.detach()
        z_hat, u_mu, u_logvar = self.u_forward(z)
        preds = self.multi_forward(encoded_images, encoded_captions)

        # compute the losses
        loss_func = nn.MSELoss(reduction='sum')
        self.z_rec_loss = loss_func(x_hat, images)
        self.z_kl_loss = torch.mean(-0.5 * torch.sum(1 + z_logvar - z_mu ** 2 - z_logvar.exp(), dim=1), dim=0)
        self.u_rec_loss = loss_func(z_hat, z)
        self.u_kl_loss = torch.mean(-0.5 * torch.sum(1 + u_logvar - u_mu ** 2 - u_logvar.exp(), dim=1), dim=0)
        self.multi_loss = loss_func(preds, annotations)
        self.num_correct = torch.sum(preds * annotations > 0)
        self.multi_acc = self.num_correct / preds.shape[0]

    def cvae_train_logs(self):
        logs = {'z_rec_loss': self.z_rec_loss, 'z_kl_loss': self.z_kl_loss,
                'multi_loss': self.multi_loss, 'multi_acc': self.multi_acc}
        self.history.save(logs)

    def small_vae_train_logs(self):
        logs = {'u_rec_loss': self.u_rec_loss, 'u_kl_loss': self.u_kl_loss}
        self.history.save(logs)

    def val_logs(self):
        logs = {'val_z_rec_loss': self.val_z_rec_loss, 'val_z_kl_loss': self.val_z_kl_loss,
                'val_u_rec_loss': self.val_u_rec_loss, 'val_u_kl_loss': self.val_u_kl_loss,
                'val_multi_loss': self.val_multi_loss}
        self.history.save(logs)

    def ckpt(self, epoch):
        date_time = datetime.datetime.now().strftime("%Y-%m-%d_multiVAE")
        path = 'epoch' + str(epoch) + '-' + date_time + '.ckpt'
        # path = 'epoch' + str(epoch) + '-' +  + '.ckpt'
        torch.save({
            'model': self.cpu().state_dict(),
            'history': self.history
        }, path)

    def train_model(self, train_loader, epochs, lr, ckpt=1):
        """
        Train the two VAEs at the same time.
        :param ckpt: how often to do checkpoints
        """
        device = get_device()
        self = self.to(device)
        optimizer = torch.optim.Adam(params=self.parameters(), lr=lr)
        for epoch in range(1, epochs + 1):
            print('Epoch {} ...'.format(epoch))
            z_rec_losses, z_kl_losses, u_rec_losses, u_kl_losses, multi_losses = [], [], [], [], []
            num_correct, total = 0, 0
            prog_bar = tqdm(train_loader)
            for images, captions, lengths, annotations in prog_bar:
                images, captions, lengths, annotations = images.to(device), captions.to(device), lengths.to(device), annotations.to(device)
                self.train_step(images, captions, lengths, annotations)
                z_loss = self.z_rec_loss + self.z_kl_loss + 2 * self.multi_loss
                u_loss = self.u_rec_loss + self.u_kl_loss

                z_rec_losses.append(float(self.z_rec_loss))
                z_kl_losses.append(float(self.z_kl_loss))
                u_rec_losses.append(float(self.u_rec_loss))
                u_kl_losses.append(float(self.u_kl_loss))
                multi_losses.append(float(self.multi_loss))
                num_correct += float(self.num_correct)
                total += float(annotations.shape[0])

                optimizer.zero_grad()
                z_loss.backward()
                u_loss.backward()
                optimizer.step()

                prog_bar.set_description(desc='z_rec: {:.2f}, z_kl: {:.2f}, u_rec: {:.2f}, u_kl: {:.2f}, multi: {:.2f}, multi-acc: {:.2f}'.format(
                    float(self.z_rec_loss), float(self.z_kl_loss), float(self.u_rec_loss), float(self.u_kl_loss), float(self.multi_loss), float(self.multi_acc)))

            if epoch % ckpt == 0 or epoch == epochs:
                self.ckpt(epoch)

            self.z_rec_loss = np.mean(z_rec_losses)
            self.z_kl_loss = np.mean(z_kl_losses)
            self.u_rec_loss = np.mean(u_rec_losses)
            self.u_kl_loss = np.mean(u_kl_losses)
            self.multi_loss = np.mean(multi_losses)
            self.multi_acc = num_correct / total
            print('z_rec: {:.2f}, z_kl: {:.2f}, u_rec: {:.2f}, u_kl: {:.2f}, multi: {:.2f}, multi-acc: {:.2f}'.format(
                self.z_rec_loss, self.z_kl_loss, self.u_rec_loss, self.u_kl_loss,
                self.multi_loss, self.multi_acc))

            self.cvae_train_logs()

    def cvae_train_step(self, images, captions, lengths, annotations):
        encoded_images = self.cvae.im_encoder(images)  # batch_size x enc_dim
        encoded_captions = self.cvae.cap_encoder(captions, lengths)  # batch_size x hidden_dim

        x_hat, z, z_mu, z_logvar = self.z_forward(encoded_images, encoded_captions)
        preds = self.multi_forward(encoded_images, encoded_captions)

        # compute the losses
        loss_func = nn.MSELoss(reduction='sum')
        self.z_rec_loss = loss_func(x_hat, images)
        self.z_kl_loss = torch.mean(-0.5 * torch.sum(1 + z_logvar - z_mu ** 2 - z_logvar.exp(), dim=1), dim=0)
        self.multi_loss = loss_func(preds, annotations)
        self.num_correct = torch.sum(preds * annotations > 0)
        self.multi_acc = self.num_correct / preds.shape[0]

    def train_cvae(self, train_loader, epochs, lr, ckpt):
        device = get_device()
        self = self.to(device)
        optimizer = torch.optim.Adam(params=self.parameters(), lr=lr)
        for epoch in range(1, epochs + 1):
            print('Epoch {} ...'.format(epoch))
            z_rec_losses, z_kl_losses, multi_losses = [], [], []
            num_correct, total = 0, 0
            prog_bar = tqdm(train_loader)
            for images, captions, lengths, annotations in prog_bar:
                images, captions, lengths, annotations = images.to(device), captions.to(device), lengths.to(device), annotations.to(device)
                self.cvae_train_step(images, captions, lengths, annotations)
                z_loss = self.z_rec_loss + self.z_kl_loss + 2 * self.multi_loss

                z_rec_losses.append(float(self.z_rec_loss))
                z_kl_losses.append(float(self.z_kl_loss))
                multi_losses.append(float(self.multi_loss))
                num_correct += float(self.num_correct)
                total += float(annotations.shape[0])

                optimizer.zero_grad()
                z_loss.backward()
                optimizer.step()

                prog_bar.set_description(desc='z_rec: {:.2f}, z_kl: {:.2f}, multi: {:.2f}, multi-acc: {:.2f}'.format(
                    float(self.z_rec_loss), float(self.z_kl_loss), float(self.multi_loss), float(self.multi_acc)))

            if epoch % ckpt == 0 or epoch == epochs:
                self.ckpt(epoch)

            self.z_rec_loss = np.mean(z_rec_losses)
            self.z_kl_loss = np.mean(z_kl_losses)
            self.multi_loss = np.mean(multi_losses)
            self.multi_acc = num_correct / total
            print('z_rec: {:.2f}, z_kl: {:.2f}, multi: {:.2f}, multi-acc: {:.2f}'.format(
                self.z_rec_loss, self.z_kl_loss, self.multi_loss, self.multi_acc))

            self.cvae_train_logs()

    def small_vae_train_step(self, images, captions, lengths, annotations):
        with torch.no_grad():
            encoded_images = self.cvae.im_encoder(images)  # batch_size x enc_dim
            encoded_captions = self.cvae.cap_encoder(captions, lengths)  # batch_size x hidden_dim

            x_hat, z, z_mu, z_logvar = self.z_forward(encoded_images, encoded_captions)
        z_hat, u_mu, u_logvar = self.u_forward(z)
        # compute the losses
        loss_func = nn.MSELoss(reduction='sum')
        self.u_rec_loss = loss_func(z_hat, z)
        self.u_kl_loss = torch.mean(-0.5 * torch.sum(1 + u_logvar - u_mu ** 2 - u_logvar.exp(), dim=1), dim=0)

    def train_small_vae(self, train_loader, epochs, lr):
        device = get_device()
        self = self.to(device)
        optimizer = torch.optim.Adam(params=self.small_vae.parameters(), lr=lr)
        for epoch in range(1, epochs + 1):
            print('Epoch {} ...'.format(epoch))
            z_rec_losses, z_kl_losses, u_rec_losses, u_kl_losses, multi_losses = [], [], [], [], []
            prog_bar = tqdm(train_loader)
            for images, captions, lengths, annotations in prog_bar:
                images, captions, lengths, annotations = images.to(device), captions.to(device), lengths.to(
                    device), annotations.to(device)
                self.small_vae_train_step(images, captions, lengths, annotations)
                u_loss = self.u_rec_loss + self.u_kl_loss

                u_rec_losses.append(float(self.u_rec_loss))
                u_kl_losses.append(float(self.u_kl_loss))

                optimizer.zero_grad()
                u_loss.backward()
                optimizer.step()

                prog_bar.set_description(
                    desc='u_rec: {:.2f}, u_kl: {:.2f}'.format(float(self.u_rec_loss), float(self.u_kl_loss)))

            self.u_rec_loss = np.mean(u_rec_losses)
            self.u_kl_loss = np.mean(u_kl_losses)

            print('u_rec: {:.2f}, u_kl: {:.2f}'.format(self.u_rec_loss, self.u_kl_loss))

            self.small_vae_train_logs()

    def train_two_stages(self, train_loader, epochs, epochs2, lr, ckpt=1):
        """
        Train the CVAE first, then the small VAE.
        :param epochs: number of epochs for the CVAE training
        :param epochs2: number of epochs for the small VAE training
        :param ckpt: frequency of the checkpoints
        :return:
        """
        print('Training CVAE ...')
        self.train_cvae(train_loader=train_loader, epochs=epochs, lr=lr, ckpt=ckpt)
        print('CVAE trained!')
        self.train_small_vae(train_loader=train_loader, epochs=epochs2, lr=lr)

