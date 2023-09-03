import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm

from history import History
from model_utils import ResBlockEnc, ResBlockDec, ConvLayer, ConvTransLayer


# Convolutional image encoder
from utils import get_device


class ImageEncoder(nn.Module):
    def __init__(self, im_size, channel_dims, enc_dim):
        super(ImageEncoder, self).__init__()

        layers = []
        in_channel = 3
        for channel in channel_dims:
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=channel, kernel_size=(3,3), stride=(2,2)),
                nn.BatchNorm2d(channel), nn.LeakyReLU()
            ))
            in_channel = channel

        self.encoder = nn.Sequential(*layers)   # output shaped (batch_size, channel_dims[-1], im_size //(2**len(channel_dims) -1))

        new_size = im_size // (2**len(channel_dims)) - 1
        self.fc = nn.Linear(in_features=new_size*new_size*channel_dims[-1], out_features=enc_dim)

    def forward(self, x):  # x is shaped (batch_size, 3, im_size, im_size)
        enc = self.encoder(x).view(-1, self.fc.in_features)
        return self.fc(enc)


# convolutional decoder
class Decoder(nn.Module):
    def __init__(self, channel_dims):  # already reversed
        super(Decoder, self).__init__()

        layers = []
        in_channel = channel_dims[0]
        for channel in channel_dims[1:]:
            layers.append(nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channel, out_channels=channel, kernel_size=(3,3), stride=(2,2)),
                nn.BatchNorm2d(channel), nn.LeakyReLU(),
            ))
            in_channel = channel
        layers.append(nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channel, out_channels=3, kernel_size=(3, 3), stride=(2,2), output_padding=(1, 1)),
            nn.Tanh()
        ))

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)


# resnet-based image encoder
class ImageEncoderRes(nn.Module):
    def __init__(self, im_size, enc_dim, channel_dims=None):
        super(ImageEncoderRes, self).__init__()
        self.encoder = nn.Sequential()
        channel_dims = [16, 32, 64, 128, 256] if channel_dims is None else channel_dims
        layers = []
        in_channel = 3
        for channel in channel_dims:
            layers.append(ResBlockEnc(in_channels=in_channel, out_channels=channel, stride=2))
            in_channel = channel
        self.encoder = nn.Sequential(*layers)
        new_size = im_size // (2**len(channel_dims))
        self.fc = nn.Linear(new_size * new_size * channel_dims[-1], enc_dim)

    def forward(self, x):
        enc = self.encoder(x).view(-1, self.fc.in_features)
        return self.fc(enc)


# resnet based decoder
class DecoderRes(nn.Module):
    def __init__(self, channel_dims=None): # already reversed
        super(DecoderRes, self).__init__()
        channel_dims = [256, 128, 64, 32, 16] if channel_dims is None else channel_dims
        layers = []
        in_channel = channel_dims[0]
        for channel in channel_dims:
            layers.append(ResBlockDec(in_channels=in_channel, out_channels=channel, stride=2))
            in_channel = channel
        layers.append(nn.Sequential(
            ConvLayer(in_channels=in_channel, out_channels=in_channel, kernel=3, stride=2, bn=False),
            ConvTransLayer(in_channels=in_channel, out_channels=3, kernel=3, stride=2, out_pad=1, bn=False),
            nn.Tanh()))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)


# LSTM caption encoder
class CaptionEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super(CaptionEncoder, self).__init__()

        self.hidden_size = hidden_size
        self.embed_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.lstm = nn. LSTM(input_size=embed_dim, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(in_features=2 * hidden_size, out_features=hidden_size),
        )

    def forward(self, captions, lengths):
        emb = self.embed_layer(captions)
        packed = pack_padded_sequence(input=emb, lengths=lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hn, cn) = self.lstm(packed)
        out = self.fc(torch.cat([hn.squeeze(0), cn.squeeze(0)], dim=1))
        return out


# this is similar to CaptionEncoder, except the lstm outputs are not passed through a fc layer,
# they are hence doubled in size, compared to the prebious class.
class CaptionEncoderDouble(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super(CaptionEncoderDouble, self).__init__()

        self.hidden_size = hidden_size
        self.embed_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.lstm = nn. LSTM(input_size=embed_dim, hidden_size=hidden_size, batch_first=True)

    def forward(self, captions, lengths):
        emb = self.embed_layer(captions)
        packed = pack_padded_sequence(input=emb, lengths=lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hn, cn) = self.lstm(packed)
        return torch.cat([hn.squeeze(0), cn.squeeze(0)], dim=1)


# can be used exactly as a Caption Encoder, except hidden_size has to be equal 40, the number of annotated features
class CaptionToAnnotation(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size=40):
        super(CaptionToAnnotation, self).__init__()
        assert hidden_size == 40

        self.embed_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.lstm = nn. LSTM(input_size=embed_dim, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(in_features=2*hidden_size, out_features=hidden_size)

    def forward(self, captions, lengths):
        emb = self.embed_layer(captions)
        packed = pack_padded_sequence(input=emb, lengths=lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hn, cn) = self.lstm(packed)
        out = torch.cat([hn.squeeze(0), cn.squeeze(0)], dim=1)
        return self.fc(out)


# encoder that performs the reparameterization trick and generates z
class Encoder(nn.Module):
    def __init__(self, enc_dim, hidden_size, latent_dim):
        super(Encoder, self).__init__()

        self.fc_mu = nn.Linear(in_features=enc_dim+hidden_size, out_features=latent_dim)
        self.fc_logvar = nn.Linear(in_features=enc_dim+hidden_size, out_features=latent_dim)

        # intialize the variance layer to 0 for stable training
        nn.init.zeros_(self.fc_logvar.weight)
        nn.init.zeros_(self.fc_logvar.bias)

    def forward(self, encoded):
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        z = self._reparameterize(mu=mu, logvar=logvar)
        return z, mu, logvar

    def _reparameterize(self, mu, logvar):
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)




# Vanilla VAE
class VAE(nn.Module):
    def __init__(self, im_size, latent_dim, channel_dims):
        super(VAE, self).__init__()
        self.new_size = im_size // (2 ** len(channel_dims)) - 1
        self.last_channel = channel_dims[-1]

        self.encoder = ImageEncoder(im_size=im_size, channel_dims=channel_dims, enc_dim=latent_dim)
        self.fc_mu = nn.Linear(in_features=latent_dim, out_features=latent_dim)
        self.fc_logvar = nn.Linear(in_features=latent_dim, out_features=latent_dim)
        nn.init.zeros_(self.fc_logvar.weight)
        nn.init.zeros_(self.fc_logvar.bias)

        channel_dims.reverse()
        self.fc = nn.Linear(in_features=latent_dim, out_features=self.last_channel*self.new_size*self.new_size)
        self.decoder = Decoder(channel_dims=channel_dims)

    def _reparameterize(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        z = mu + torch.exp(0.5 * logvar) * epsilon
        return z

    def encode(self, x):
        x_encoded = self.encoder(x)
        mu = self.fc_mu(x_encoded)
        logvar = self.fc_logvar(x_encoded)
        z = self._reparameterize(mu=mu, logvar=logvar)
        return z, mu, logvar

    def decode(self, z):
        return self.decoder(self.fc(z).view(-1, self.last_channel, self.new_size, self.new_size))

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        recons = self.decode(z)
        return recons, mu, logvar


# (Typically small) VAE with linear layers only
class LinVAE(nn.Module):
    def __init__(self, latent_dim, num_layers=3):
        super(LinVAE, self).__init__()
        self.latent_dim = latent_dim

        # build the encoder
        enc_layers = []
        for _ in range(num_layers):
            enc_layers.append(nn.Sequential(
                nn.Linear(in_features=latent_dim, out_features=latent_dim),
                nn.BatchNorm1d(latent_dim),
                nn.LeakyReLU()
            ))
        enc_layers.append(nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU()
        ))
        self.encoder = nn.Sequential(*enc_layers)

        self.fc_mu = nn.Linear(in_features=latent_dim, out_features=latent_dim)
        self.fc_logvar = nn.Linear(in_features=latent_dim, out_features=latent_dim)
        nn.init.zeros_(self.fc_logvar.weight)
        nn.init.zeros_(self.fc_logvar.bias)

        # build the decoder
        dec_layers = []
        in_feature = latent_dim
        for _ in range(num_layers):
            dec_layers.append(nn.Sequential(
                nn.Linear(in_features=latent_dim, out_features=latent_dim),
                nn.BatchNorm1d(latent_dim),
                nn.ReLU()
            ))
        dec_layers.append(nn.Sequential(
            nn.Linear(in_features=in_feature, out_features=latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU()
        ))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, z_batch):
        enc = self.encoder(z_batch)
        mu, logvar = self.fc_mu(enc), self.fc_logvar(enc)
        return mu, logvar

    def _reparameterize(self, mu, logvar):
        chi = torch.randn_like(mu)
        u = mu + chi * torch.exp(0.5 * logvar)
        return u

    def decode(self, u_batch):
        return self.decoder(u_batch)

    def forward(self, z_batch):
        mu, logvar = self.encode(z_batch)
        u = self._reparameterize(mu=mu, logvar=logvar)
        z_hat = self.decode(u)
        return z_hat, mu, logvar


# Another version of the previous class to customize the layer dimensions
class LinVAE2(LinVAE):
    def __init__(self, latent_dim, feature_dims=None):
        super(LinVAE2, self).__init__(latent_dim, num_layers=0)

        feature_dims = [256, 256, 256] if feature_dims is None else feature_dims
        # build the encoder
        enc_layers = []
        in_feat = latent_dim
        for out_feat in feature_dims:
            enc_layers.append(nn.Sequential(
                nn.Linear(in_features=in_feat, out_features=out_feat),
                nn.BatchNorm1d(out_feat),
                nn.LeakyReLU()
            ))
            in_feat = out_feat
        enc_layers.append(nn.Sequential(
            nn.Linear(in_features=in_feat, out_features=latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU()
        ))
        self.encoder = nn.Sequential(*enc_layers)

        self.fc_mu = nn.Linear(in_features=latent_dim, out_features=latent_dim)
        self.fc_logvar = nn.Linear(in_features=latent_dim, out_features=latent_dim)
        nn.init.zeros_(self.fc_logvar.weight)
        nn.init.zeros_(self.fc_logvar.bias)

        # build the decoder
        dec_layers = []
        feature_dims.reverse()
        in_feat = latent_dim
        for out_feat in feature_dims:
            dec_layers.append(nn.Sequential(
                nn.Linear(in_features=in_feat, out_features=out_feat),
                nn.BatchNorm1d(out_feat),
                nn.LeakyReLU()
            ))
            in_feat = out_feat
        dec_layers.append(nn.Sequential(
            nn.Linear(in_features=in_feat, out_features=latent_dim),
            nn.Sigmoid()
        ))
        self.decoder = nn.Sequential(*dec_layers)


# Two-stage VAE
class TwoStageCVAE(nn.Module):
    def __init__(self, im_size, enc_dim, vocab_size, embed_dim, hidden_size, latent_dim, channel_dims, u_dim=32, feature_dims=None):
        super(TwoStageCVAE, self).__init__()
        self.latent_dim = latent_dim
        self.cvae = CVAE(im_size, enc_dim, vocab_size, embed_dim, hidden_size, latent_dim, channel_dims)

        self.u_dim = u_dim
        self.small_vae = LinVAE(latent_dim=u_dim)

    def forward(self, images, captions, lengths):
        encoded_images = self.cvae.im_encoder(images)  # batch_size x enc_dim
        encoded_captions = self.cvae.cap_encoder(captions, lengths)  # batch_size x hidden_dim
        z, z_mu, z_logvar = self.cvae.encode(encoded_images=encoded_images, encoded_captions=encoded_captions)

        z_hat, u_mu, u_logvar = self.small_vae(z.detach())

        x_hat = self.cvae.decode(z, encoded_captions=encoded_captions)

        return (x_hat, z, z_mu, z_logvar), (z_hat, u_mu, u_logvar)

    def z_forward(self, images, captions, lengths):
        encoded_images = self.cvae.im_encoder(images)  # batch_size x enc_dim
        encoded_captions = self.cvae.cap_encoder(captions, lengths)  # batch_size x hidden_dim
        z, z_mu, z_logvar = self.cvae.encode(encoded_images=encoded_images, encoded_captions=encoded_captions)
        x_hat = self.cvae.decode(z, encoded_captions=encoded_captions)
        return (x_hat, z, z_mu, z_logvar)

    def u_forward(self, z):
        z_hat, u_mu, u_logvar = self.small_vae(z)
        return z_hat, u_mu, u_logvar

    def sample_images(self, captions, lengths):
        u = torch.randn(captions.shape[0], self.u_dim)
        z = self.small_vae.decode(u)
        encoded_captions = self.cvae.cap_encoder(captions, lengths)
        recons = self.cvae.decode(z, encoded_captions)
        return recons

    def freeze_caption_encoder(self):
        self.cvae.freeze_caption_encoder()


# conditional VAE
class CVAE(nn.Module):
    def __init__(self, im_size, enc_dim, vocab_size, embed_dim, hidden_size, latent_dim, channel_dims=None):
        super(CVAE, self).__init__()
        channel_dims = [16, 32, 64, 128, 256] if channel_dims is None else channel_dims

        self.new_size = im_size // (2**len(channel_dims))
        self.last_channel = channel_dims[-1]
        self.latent_dim = latent_dim

        self.im_encoder = ImageEncoderRes(im_size=im_size, channel_dims=channel_dims, enc_dim=enc_dim)
        self.cap_encoder = CaptionEncoderDouble(vocab_size=vocab_size, embed_dim=embed_dim, hidden_size=hidden_size)
        self.encoder = Encoder(enc_dim=enc_dim, hidden_size=2*hidden_size, latent_dim=latent_dim)

        channel_dims.reverse()
        self.fc = nn.Linear(in_features=latent_dim + 2*hidden_size, out_features=self.last_channel*self.new_size*self.new_size)
        self.decoder = DecoderRes(channel_dims=channel_dims)

    def encode(self, encoded_images, encoded_captions):
        z, mu, logvar = self.encoder(torch.cat([encoded_images, encoded_captions], dim=1))
        return z, mu, logvar

    def decode(self, z, encoded_captions):
        encoded = self.fc(torch.cat([z, encoded_captions], dim=1)).view(-1, self.last_channel, self.new_size, self.new_size)
        recons = self.decoder(encoded)
        return recons

    def forward(self, images, captions, lengths):
        # encode
        encoded_images = self.im_encoder(images)  # batch_size x enc_dim
        encoded_captions = self.cap_encoder(captions, lengths)  # batch_size x hidden_dim
        z, mu, logvar = self.encode(encoded_images, encoded_captions)   # batch_size x latent_dim

        # decode
        recons = self.decode(z, encoded_captions)
        return recons, mu, logvar

    def sample_images(self, captions, lengths):
        # captions is a Tensor of integers
        z = torch.randn((captions.shape[0], self.latent_dim))
        encoded_captions = self.cap_encoder(captions, lengths)
        encoded = self.fc(torch.cat([z, encoded_captions], dim=1)).view(-1, self.last_channel, self.new_size, self.new_size)
        recons = self.decoder(encoded)
        return recons

    def freeze_caption_encoder(self):
        for param in list(self.cap_encoder.parameters()):
            param.requires_grad = False
        print('Parameters of the captionEncoder frozen!')


class MultiClass(nn.Module):
    # A multi-class classifier that predicts the annotation vectors
    def __init__(self, im_size, channel_dims, im_enc_dim, vocab_size, embed_dim, hidden_size):
        super(MultiClass, self).__init__()
        self.im_encoder = ImageEncoder(im_size=im_size, enc_dim=im_enc_dim, channel_dims=channel_dims)
        self.cap_encoder = CaptionEncoder(vocab_size=vocab_size, embed_dim=embed_dim, hidden_size=hidden_size)

        self.fc = nn.Linear(in_features=im_enc_dim + hidden_size, out_features=40)

        self.loss = None

    def forward(self, images, captions, lengths):
        im_encoded = self.im_encoder(images)
        cap_encoded = self.cap_encoder(captions, lengths)
        return self.fc(torch.cat([im_encoded, cap_encoded], dim=1))

    def accuracy(self, preds, targets):
        # The number of
        temp = preds * targets > 0
        return torch.sum(temp, dim=1) / temp.shape[1]

    def compute_loss(self, preds, targets):
        criterion = nn.MSELoss(reduction='sum')
        return criterion(preds, targets)

    def train_step(self, images, captions, lengths, targets):
        preds = self(images, captions, lengths)
        self.loss = self.compute_loss(preds, targets)

    def train_model(self, train_loader, val_loader, epochs, lr=1e-4, ckpt=5):
        device = get_device()
        self = self.to(device)
        optimizer = torch.optim.Adam(params=self.parameters(), lr=lr)
        history = History()
        for epoch in range(1, epochs + 1):
            print('Epoch {} ...'.format(epoch))
            losses, val_losses = [], []
            for images, captions, lengths, targets in tqdm(train_loader):
                images, captions, lengths, targets = images.to(device), captions.to(device), lengths.to(device), targets.to(device)
                self.train_step(images, captions, lengths, targets)
                losses.append(float(self.loss))

                optimizer.zero_grad()
                self.loss.backward()
                optimizer.step()

            for images, captions, lengths, targets in tqdm(val_loader):
                with torch.no_grad():
                    images, captions, lengths, targets = images.to(device), captions.to(device), lengths.to( device), targets.to(device)
                    self.train_step(images, captions, lengths, targets)
                    val_losses.append(float(self.loss))

            loss, val_loss = np.mean(losses), np.mean(val_losses)
            history.save({'loss': loss, 'val_loss': val_loss})
            print('Loss: {}; val-loss: {}'.format(loss, val_loss))

            # checkpoint ?
            if epoch % ckpt == 0 or epoch == epochs:
                path = 'epoch' + str(epoch) + '-' + 'multi_task' + '.ckpt'
                torch.save({'model': self.state_dict()}, path)

        return history