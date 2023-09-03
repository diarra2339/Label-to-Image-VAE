import os
import random

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.utils import save_image
import torch.distributions as D
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
import pickle

from tqdm import tqdm


def save_object(object, filename):
    with open(filename, 'wb') as f:
        pickle.dump(object, f)
    print(str(type(object)) + ' saved to ' + filename)


def load_object(filename):
    with open(filename, 'rb') as f:
        object = pickle.load(f)
    return object


def load_model(model, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model'])
    return model


def load_optimizer(optimizer, filename):
    checkpoint = torch.load(filename)
    optimizer.load_state_dict(checkpoint['optimizer'])
    return optimizer



def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def resnet_transform(mean=None, std=None, im_size=224):
    """
    Tranforms and normalize images for a resnet
    :param mean: the mean, by default, we use image net mean
    :param std: the std, by default, use imagenet values
    :return: a transforms object
    """
    mean = [0.485, 0.456, 0.406] if mean is None else mean
    std = [0.229, 0.224, 0.225] if std is None else std

    tr = T.Compose([
        T.Resize((im_size, im_size)),
        T.ToTensor()#,
        #T.Normalize(mean=mean,
        #            std=std)
    ])
    return tr


def targets_lengths(captions):
    # captions is a list or tuple of 1D Tensors
    lengths = [len(caption) for caption in captions]
    max_length = max(lengths)
    lengths = torch.Tensor(lengths).int()
    targets = torch.zeros(len(captions), max_length).long()
    for i, caption in enumerate(captions):
        length = lengths[i]
        targets[i, :length] = caption[:length]
    return targets, lengths


def collate_fn(tuples):
    """
    The collate_fn function for the DataLoader.
    :param tuples: a list of tuples (image (Tensor), caption (Tensor))
    :return: A tuple (images (Tensor), captions (Tensor), lengths (list))
    """
    tuples.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*tuples)

    # Change the images from tuple of 3D Tensors to a 4D Tensor (batch_size x 3 x H x W)
    images = torch.stack(images, dim=0)

    # Change the captions from tuple of 1D Tensors to 2D Tensor
    targets, lengths = targets_lengths(captions=captions)

    return images, targets, lengths


def collate_captions(tuples):
    """
    The collate_fn function for the DataLoader.
    :param tuples: a list of tuples (caption (Tensor), annotation (Tensor))
    :return: A tuple (captions (Tensor), lengths(Tensor), annotations (Tensor))
    """
    tuples.sort(key=lambda x: len(x[1]), reverse=True)
    captions, annotations = zip(*tuples)
    captions, lengths = targets_lengths(captions)
    annotations = torch.cat([x.unsqueeze(dim=0) for x in annotations], dim=0)
    return captions, lengths, annotations


def collate_annots(tuples):
    """
    The collate_fn function for the DataLoader.
    :param tuples: a list of tuples (image (Tensor), caption (Tensor), annotation (Tensor))
    :return: A tuple (Images (Tensor), captions (Tensor), lengths(Tensor), annotations (Tensor))
    """
    tuples.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, annotations = zip(*tuples)
    images = torch.stack(images, dim=0)
    captions, lengths = targets_lengths(captions)
    annotations = torch.cat([x.unsqueeze(dim=0) for x in annotations], dim=0)
    return images, captions, lengths, annotations


def fid_prep_temp(model, vocab, real_folder, fake_folder, num_samples, image_folder=None, temp=0.5):
    image_folder = 'data/images/' if image_folder is None else image_folder
    indices = random.sample(range(30000), num_samples)
    captions = [''] * num_samples
    captions = [torch.Tensor(vocab.id_tokenize(caption)) for caption in captions]
    targets, lengths = targets_lengths(captions)
    dist = D.Normal(torch.zeros(model.latent_dim), temp * torch.ones(model.latent_dim))
    z = dist.sample(sample_shape=(targets.shape[0], ))
    with torch.no_grad():
        enc_cap = model.cvae.cap_encoder(targets, lengths)
        samples = model.cvae.decode(z, encoded_captions=enc_cap)

    to_pil = T.ToPILImage()
    for i in tqdm(range(num_samples)):
        # one real image
        real_img = Image.open(os.path.join(image_folder, str(indices[i]) + '.jpg')).resize((224, 224))
        real_img.save(os.path.join(real_folder, str(i) + '.jpg'), 'JPEG')
        # one fake image
        fake_img = to_pil(samples[i])
        fake_img.save(os.path.join(fake_folder, str(i) + '.jpg'), 'JPEG')


def fid_prep_2stage(model, vocab, real_folder, fake_folder, num_samples, image_folder=None):
    image_folder = 'data/images/' if image_folder is None else image_folder
    indices = random.sample(range(30000), num_samples)
    captions = [''] * num_samples
    captions = [torch.Tensor(vocab.id_tokenize(caption)) for caption in captions]
    targets, lengths = targets_lengths(captions)
    u = torch.randn(targets.shape[0], model.small_vae.latent_dim)
    with torch.no_grad():
        enc_cap = model.cvae.cap_encoder(targets, lengths)
        z_hat = model.small_vae.decode(u)
        samples = model.cvae.decode(z_hat, encoded_captions=enc_cap)

    to_pil = T.ToPILImage()
    for i in tqdm(range(num_samples)):
        # one real image
        real_img = Image.open(os.path.join(image_folder, str(indices[i]) + '.jpg')).resize((224, 224))
        real_img.save(os.path.join(real_folder, str(i) + '.jpg'), 'JPEG')
        # one fake image
        fake_img = to_pil(samples[i])
        fake_img.save(os.path.join(fake_folder, str(i) + '.jpg'), 'JPEG')





def get_annotations(annot_file):
    # returns a dictionary and a (int) tensor shaped is 30,000 x 40
    mapping = {}
    with open(annot_file, 'r') as file:
        lines = file.readlines()  # list of strings
        features = lines[1].split(' ')  # list of str; 40 feature names

    # the names of the features
    for i in range(len(features)):
        mapping[i] = features[i]

    # get the annotations
    int_lines = [line.split()[1:] for line in lines[2:]]
    int_lines = [[int(x) for x in line] for line in int_lines]

    return mapping, torch.Tensor(int_lines)


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


def show_images_vm(model, vocab, captions, temp=1):
    # captions is a list of strings
    device = get_device()
    model = model.to(device)
    with torch.no_grad():
        captions = [torch.Tensor(vocab.id_tokenize(caption)) for caption in captions]
        targets, lengths = targets_lengths(captions)
        enc_cap = model.cvae.cap_encoder(targets, lengths)

        u = torch.randn(targets.shape[0], model.small_vae.latent_dim)
        z_hat = model.small_vae.decode(u)
        dist = D.Normal(torch.zeros(model.latent_dim), temp * torch.ones(model.latent_dim))
        # z_hat = dist.sample(sample_shape=(targets.shape[0], ))
        #z_hat = torch.randn(targets.shape[0], model.latent_dim)
        images = model.cvae.decode(z_hat, encoded_captions=enc_cap)
        rows, columns = 1, images.shape[0]
        _show_images(images=images, rows=rows, columns=columns)


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


def show_random_recons(model, test_loader, n_samples, figsize=10):
    with torch.no_grad():
        for images, captions, lengths in test_loader:
            break

        # just take n_samples from the batch
        images = images[:n_samples, :, :, :]
        captions = captions[:n_samples, :]
        lengths = lengths[:n_samples]
        if 'TwoStageCVAE' in str(type(model)):
            recons, _, _, _ = model.z_forward(images, captions, lengths)
        elif 'CVAEMulti' in str(type(model)):
            encoded_images = model.cvae.im_encoder(images)  # batch_size x enc_dim
            encoded_captions = model.cvae.cap_encoder(captions, lengths)  # batch_size x hidden_dim
            recons, _, _, _ = model.z_forward(encoded_images, encoded_captions)
        else:
            recons, _, _ = model(images, captions, lengths)
        _show_images(torch.cat([images, recons], dim=0), rows=2, columns=n_samples, figsize=figsize)


def train_second_stage(model, train_loader, epochs, lr):
    loss_func = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rec_losses, kl_losses = [], []
    for epoch in range(epochs):
        prog_bar = tqdm(train_loader)
        for images, captions, lengths, annotations in prog_bar:
            with torch.no_grad():
                encoded_images = model.cvae.im_encoder(images)  # batch_size x enc_dim
                encoded_captions = model.cvae.cap_encoder(captions, lengths)  # batch_size x hidden_dim

                _, z, z_mu, z_logvar = model.z_forward(encoded_images, encoded_captions)
            z_hat, u_mu, u_logvar = model.u_forward(z)
            u_rec_loss = loss_func(z_hat, z)
            u_kl_loss = torch.mean(-0.5 * torch.sum(1 + u_logvar - u_mu ** 2 - u_logvar.exp(), dim=1), dim=0)
            loss = u_rec_loss + u_kl_loss
            rec_losses.append(float(u_rec_loss))
            kl_losses.append(float(u_kl_loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prog_bar.set_description(desc='u_rec: {:.2f}, u_kl: {:.2f}'.format(float(u_rec_loss), float(u_kl_loss)))

        print('rec-loss: {:.3f}, kl-loss: {:.3f}'.format(np.mean(rec_losses), np.mean(kl_losses)))


