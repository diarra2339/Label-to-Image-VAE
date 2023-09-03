import torch
import matplotlib.pyplot as plt
import torch.distributions as D
from sklearn import mixture
from tqdm import tqdm

from utils import get_device, targets_lengths


class GMMSampler():
    def __init__(self, model, train_loader, n_components=10):
        self.model = model
        self.components = n_components
        self.train(train_loader)

    def train(self, train_loader):
        mu = []
        device = get_device()
        self.model = self.model.to(device)
        with torch.no_grad():
            for images, captions, lengths, _ in tqdm(train_loader):
                images, captions, lengths = images.to(device), captions.to(device), lengths.to(device)
                encoded_images = self.model.cvae.im_encoder(images)  # batch_size x enc_dim
                encoded_captions = self.model.cvae.cap_encoder(captions, lengths)  # batch_size x hidden_dim
                _, _, mu_data, _ = self.model.z_forward(encoded_images, encoded_captions)
                mu.append(mu_data)
        mu = torch.cat(tensors=mu, dim=0)
        gmm = mixture.GaussianMixture(n_components=self.components, covariance_type='full', max_iter=1000, verbose=0)
        gmm.fit(mu.cpu().detach())
        self.gmm = gmm
        self.model = self.model.cpu()

    def sample(self, vocab, captions):
        captions = [torch.Tensor(vocab.id_tokenize(caption)) for caption in captions]
        targets, lengths = targets_lengths(captions)
        z = torch.tensor(self.gmm.sample(targets.shape[0])[0]).float()
        with torch.no_grad():
            enc_cap = self.model.cvae.cap_encoder(targets, lengths)
            images = self.model.cvae.decode(z, encoded_captions=enc_cap)
        rows, columns = 1, images.shape[0]
        _show_images(images=images, rows=rows, columns=columns)


def show_image(im_tensor):
    # im_tensor is a 3 x H x W Tensor image
    plt.imshow(im_tensor.permute(1, 2, 0))
    plt.axis('off')


def _show_images(images, rows, columns, figsize=10):
    fig = plt.figure(figsize=(figsize, figsize))
    for i in range(1, columns * rows + 1):
        img = images[i-1].permute(1, 2, 0)
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
        #plt.imshow((img * 255).numpy().astype(np.uint8))
        plt.axis('off')
    fig.tight_layout()
    plt.show()


def sample_2stage(model, vocab, captions, figsize=10):
    # captions is a list of strings
    device = get_device()
    model = model.to(device)
    captions = [torch.Tensor(vocab.id_tokenize(caption)) for caption in captions]
    targets, lengths = targets_lengths(captions)
    u = torch.randn(targets.shape[0], model.small_vae.latent_dim)
    with torch.no_grad():
        enc_cap = model.cvae.cap_encoder(targets, lengths)
        z_hat = model.small_vae.decode(u)
        images = model.cvae.decode(z_hat, encoded_captions=enc_cap)
    rows, columns = 1, images.shape[0]
    _show_images(images=images, rows=rows, columns=columns, figsize=figsize)


def sample_temp(model, vocab, captions, temp=1, figsize=10):
    # captions is a list of strings
    device = get_device()
    model = model.to(device)
    captions = [torch.Tensor(vocab.id_tokenize(caption)) for caption in captions]
    targets, lengths = targets_lengths(captions)
    dist = D.Normal(torch.zeros(model.latent_dim), temp * torch.ones(model.latent_dim))
    z = dist.sample(sample_shape=(targets.shape[0], ))
    with torch.no_grad():
        enc_cap = model.cvae.cap_encoder(targets, lengths)
        images = model.cvae.decode(z, encoded_captions=enc_cap)
    rows, columns = 1, images.shape[0]
    _show_images(images=images, rows=rows, columns=columns, figsize=figsize)


def show_images(model, vocab, captions):
    # captions is a list of strings
    device = get_device()
    model = model.to(device)
    with torch.no_grad():
        captions = [torch.Tensor(vocab.id_tokenize(caption)) for caption in captions]
        targets, lengths = targets_lengths(captions)
        images = model.sample_images(captions=targets, lengths=lengths)
        rows, columns = 1, images.shape[0]
        _show_images(images=images, rows=rows, columns=columns)


def show_random_recons(cvae_multi, test_loader, n_samples, figsize=10):
    with torch.no_grad():
        for images, captions, lengths, annotations in test_loader:
            break

        # just take n_samples from the batch
        images = images[:n_samples, :, :, :]
        captions = captions[:n_samples, :]
        lengths = lengths[:n_samples]
        encoded_images = cvae_multi.cvae.im_encoder(images)  # batch_size x enc_dim
        encoded_captions = cvae_multi.cvae.cap_encoder(captions, lengths)  # batch_size x hidden_dim
        recons, _, _, _ = cvae_multi.z_forward(encoded_images, encoded_captions)

        _show_images(torch.cat([images, recons], dim=0), rows=2, columns=n_samples, figsize=figsize)
