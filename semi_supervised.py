import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset, DataLoader
from tqdm import tqdm
from collections import defaultdict
import os
import random
import numpy as np
from PIL import Image
from model import ImageEncoder, Decoder
from utils import get_device, get_annotations, resnet_transform
from history import History


# dataset with images and annotations only, no captions.
class SSData(Dataset):
    def __init__(self, image_folder, ann_file, train_split=0.9):
        self.image_folder = image_folder
        self.ann_file = ann_file
        self.map, self.annotations = get_annotations(annot_file=ann_file)  # shaped 30,000 x 40
        self.transform = resnet_transform()
        self._set_loaders(train_split=train_split, batch_size=64, workers=1)

    def __getitem__(self, index):
        image_file = os.path.join(self.image_folder, str(index)+'.jpg')  # get the absolute path of the image
        with open(image_file, 'rb') as file:
            image = Image.open(file).convert('RGB')
        return self.transform(image), self.annotations[index]

    def __len__(self):
        return self.annotations.shape[0]

    def _set_loaders(self, train_split, batch_size, workers):
        num_samples = len(self)
        indices = list(range(num_samples))
        random.shuffle(indices)
        train_indices = indices[:int(num_samples*train_split)]
        test_indices = indices[int(num_samples*(train_split)) : num_samples]

        train_data = Subset(dataset=self, indices=train_indices)
        test_data = Subset(dataset=self, indices=test_indices)

        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=workers)
        self.test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=workers)

    def words_to_annot(self, words):
        valids = [word for word in words if word in self.map.keys()]
        vec = torch.Tensor([-1 for _ in range(40)])
        for word in valids:
            vec[self.map[word]] = torch.Tensor([1])
        return vec


# No use of captions, the annotation vector is used as a latent vector
class SSVAE(nn.Module):
    def __init__(self, im_size, enc_dim, channel_dims=[8, 16 ,32, 64]):
        super(SSVAE, self).__init__()

        self.latent_dim = 40
        self.last_channel = channel_dims[-1]
        self.new_size = im_size // (2 ** len(channel_dims)) - 1

        self.image_encoder = ImageEncoder(im_size=im_size, channel_dims=channel_dims, enc_dim=enc_dim)
        self.fc_1 = nn.Sequential(nn.Linear(in_features=enc_dim, out_features=self.latent_dim), nn.LeakyReLU())

        self.fc_mu = nn.Linear(in_features=self.latent_dim, out_features=self.latent_dim)
        self.fc_logvar = nn.Linear(in_features=self.latent_dim, out_features=self.latent_dim)
        nn.init.zeros_(self.fc_logvar.weight)
        nn.init.zeros_(self.fc_logvar.bias)

        channel_dims.reverse()
        self.fc_dec = nn.Linear(in_features=self.latent_dim, out_features=self.last_channel*self.new_size*self.new_size)
        self.decoder = Decoder(channel_dims=channel_dims)

    def encode(self, x):
        encoded = self.fc_1(self.image_encoder(x))
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        return mu, logvar

    def _reparameterize(self, mu, logvar):
        eps = torch.randn_like(mu)
        return mu + torch.exp(0.5 * logvar) * eps

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self._reparameterize(mu, logvar)
        x_hat = self.decoder(self.fc_dec(z).view(-1, self.last_channel, self.new_size, self.new_size))
        return x_hat, z, mu, logvar

    def train_step(self, x, y):
        x_hat, z, mu, logvar = self(x)
        criterion = nn.MSELoss(reduction='sum')
        self.mse_loss = criterion(x_hat, x)
        self.kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
        self.cap_loss = criterion(z, y)
        self.loss = self.mse_loss + self.kl_loss + self.cap_loss

    def generate(self, n_samples, y, z_prop=0.5):
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim)
            x_hat = self.decoder(self.fc_dec(z_prop*z + (1-z_prop)*y).view(-1, self.last_channel, self.new_size, self.new_size))
            return x_hat

    def reconstruct(self, x):
        x_hat, _, _, _ = self(x)
        return x_hat

    def train_model(self, train_loader, epochs, lr=1e-3):
        device = get_device()
        self = self.to(device)
        h = History()
        optimizer = torch.optim.Adam(params=self.parameters(), lr=lr)
        for epoch in range(1, epochs+1):
            losses = defaultdict(list)
            print('Epoch {} ...'.format(epoch))
            progress_bar = tqdm(train_loader)
            for x, y in progress_bar:
                x, y = x.to(device), y.to(device)
                self.train_step(x, y)

                optimizer.zero_grad()
                self.loss.backward()
                optimizer.step()

                losses['mse_loss'].append(float(self.mse_loss))
                losses['kl_loss'].append(float(self.kl_loss))
                losses['cap_loss'].append(float(self.cap_loss))

                progress_bar.set_description(desc='mse_loss: {:.2f}, kl_loss: {:.2f}, cap_loss: {:.2f}'.format(self.mse_loss, self.kl_loss, self.cap_loss))

            avg_mse, avg_kl, avg_cap = float(np.mean(losses['mse_loss'])), float(np.mean(losses['kl_loss'])), float(np.mean(losses['cap_loss']))
            print('Epoch {}. MSE-loss: {:.2f}, KL-loss: {:.2f}, cap_loss: {:.2f}'.format(epoch, avg_mse, avg_kl, avg_cap))


