from utils import load_object, resnet_transform
from data import Data4
from vae_multi import CVAEMulti3
from sample import *
from vocab import Vocab
import random
import numpy

torch.manual_seed(0)
random.seed(0)
numpy.random.seed(0)

img_folder = 'data/images'
cap_folder = 'data/captions'
annot_file = 'data/annotations.txt'

vocab = Vocab(caption_folder=cap_folder)

if __name__ == '__main__':
    data = Data4(vocab=vocab, batch_size=128, train_test_split=[0.8, 0.1], transform=resnet_transform(im_size=224), workers=5)
    train_loader, val_loader, test_loader = data.train_loader, data.val_loader, data.test_loader

    model = CVAEMulti3(im_size=224, channel_dims=None, im_enc_dim=256, vocab_size=len(vocab), embed_dim=32,
                       hidden_size=64, latent_dim=256)
    model.train_two_stages(train_loader, epochs=20, epochs2=10, lr=1e-3, ckpt=1)

    print('Done!')
