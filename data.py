import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import os
import random
random.seed(0)

from utils import collate_fn, collate_captions, get_annotations, collate_annots


# Dataset with images and captions only, no annotation vectors.
class CelebData(Dataset):
    def __init__(self, image_folder, caption_folder, vocab, batch_size, transform, train_test_split, workers=1):
        """
        :param image_folder: the path to the folder containing the images
        :param caption_folder: the path to the folder containing the captions
        :param vocab: the vocabulary object used to change the captions into idx
        :param batch_size: the size of the batches for the dataloaders
        :param transform: the default transform performed on the images. No normalization.
        :param train_test_split: the portions of the data used for training set and test set
        :param workers:
        """
        super(CelebData, self).__init__()
        if train_test_split is None:
            train_test_split = [0.8, 0.1]

        self.image_folder = image_folder
        self.caption_folder = caption_folder
        self.num_images = len([name for name in os.listdir(image_folder)])
        self.cap_per_image = 10
        self.vocab = vocab
        self.transform = transform

        self._set_loaders(train_test_split=train_test_split, batch_size=batch_size, workers=workers)

    def __getitem__(self, index):
        # returns a pair (image (Tensor), caption (Tensor)) of an image and a corresponding caption
        assert 0 <= index < len(self)
        image_id, caption_number = self._pair_from_index(index)
        return self._get_image(image_id), self._get_caption(image_id, caption_number)

    def __len__(self):
        # The number of total datapoints in the dataset
        return self.num_images * self.cap_per_image

    def _get_image(self, image_id):
        # returns an image from the given index
        image_file = os.path.join(self.image_folder, str(image_id)+'.jpg')  # get the absolute path of the image
        with open(image_file, 'rb') as file:
            image = Image.open(file).convert('RGB')
        return self.transform(image)

    def _get_caption(self, image_id, caption_number):
        # extracts a caption for a specified image. Each image has 10.
        caption_file = os.path.join(self.caption_folder, str(image_id)+'.txt')
        with open(caption_file, 'r') as file:
            for i, cap in enumerate(file):
                if i == caption_number:
                    caption = cap
                    break
        caption = self.vocab.id_tokenize(caption)
        return torch.Tensor(caption).int()

    def _index_from_pair(self, pair):
        # pair is a (image_id, caption_number) pair and we want the index of the item in the dataset
        return 10 * pair[0] + pair[1]

    def _pair_from_index(self, index):
        # get the (image_id, caption_number) pair from the index
        caption_number = index % 10
        image_id = (index - caption_number) // 10
        return image_id, caption_number

    def _set_loaders(self, train_test_split, batch_size, workers):
        # set the dataloaders for the instance.
        train_split, val_split, test_split = train_test_split[0], 1-sum(train_test_split), train_test_split[1]
        assert train_split > 0 and val_split >= 0 and train_split + val_split <= 1

        num_samples = len(self)
        indices = list(range(num_samples))
        random.shuffle(indices)
        train_indices = indices[:int(num_samples*train_split)]
        val_indices = indices[int(num_samples*train_split) : int(num_samples*(train_split+val_split))]
        test_indices = indices[int(num_samples*(train_split+val_split)) : num_samples]

        train_data = Subset(dataset=self, indices=train_indices)
        val_data = Subset(dataset=self, indices=val_indices)
        test_data = Subset(dataset=self, indices=test_indices)

        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=collate_fn)
        self.val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=collate_fn)
        self.test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=collate_fn)


# This dataset has captions and annotations only. It allows to pretrain a caption encoder.
class CaptionData(Dataset):
    def __init__(self, vocab, cap_folder, annot_file, train_test_split=[0.8, 0.1], batch_size=128, workers=1):
        self.vocab = vocab
        self.cap_folder = cap_folder
        self.annot_file = annot_file

        self.mapping, self.annots = get_annotations(annot_file)

        self.num_features = 40
        self.captions_per_image = 10
        self.num_images = len([name for name in os.listdir(cap_folder)])

        self._set_data_loaders(train_test_split, batch_size, workers)

    def __len__(self):
        return self.num_images * self.captions_per_image

    def __getitem__(self, index):
        # returns a pair (caption (IntTensor), annotation (IntTensor))
        file_number, line_number = self._locate_caption(index)
        caption = self._get_caption(file_number, line_number)  # Tensor shaped (40)
        annotation = self.annots[file_number]  # tensor shaped (40)
        return caption, annotation

    def _locate_caption(self, index):
        # returns a pair file_number, line_number of the caption from an index
        file_number, line_number = index // 10, index % 10
        return file_number, line_number

    def _get_caption(self, file_number, line_number):
        # returns a int tensor shaped (40)
        file_path = os.path.join(self.cap_folder, str(file_number) + '.txt')
        with open(file_path, 'r') as f:
            caption = f.readlines()[line_number]
        caption = self.vocab.id_tokenize(caption=caption)
        return torch.Tensor(caption).int()

    def _set_data_loaders(self, train_test_split, batch_size, workers):
        train_split, val_split, test_split = train_test_split[0], 1-sum(train_test_split), train_test_split[1]
        assert train_split > 0 and val_split >= 0 and train_split + val_split <= 1

        num_samples = len(self)
        indices = list(range(num_samples))
        random.shuffle(indices)
        train_indices = indices[:int(num_samples*train_split)]
        val_indices = indices[int(num_samples*train_split) : int(num_samples*(train_split+val_split))]
        test_indices = indices[int(num_samples*(train_split+val_split)) : num_samples]

        train_data = Subset(dataset=self, indices=train_indices)
        val_data = Subset(dataset=self, indices=val_indices)
        test_data = Subset(dataset=self, indices=test_indices)

        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=collate_captions)
        self.val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=collate_captions)
        self.test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=collate_captions)


# This dataset has images, captions, and annotation vectors. The '4' is because the collate_fn function will
# return 4 items: images, captions, lengths, annotations.
class Data4(Dataset):
    def __init__(self, vocab, batch_size, train_test_split, transform, workers):
        super(Data4, self).__init__()
        self.image_folder = 'data/images'
        self.caption_folder = 'data/captions'
        self.annot_file = 'data/annotations.txt'
        self.num_images = len([name for name in os.listdir(self.image_folder)])
        self.cap_per_image = 10
        self.vocab = vocab
        self.transform = transform
        self.mapping, self.annots = get_annotations(self.annot_file)

        self._set_loaders(train_test_split=train_test_split, batch_size=batch_size, workers=workers)

    def __getitem__(self, index):
        # returns a tuple (image (Tensor), caption (Tensor), annot (Tensor))
        assert 0 <= index < len(self)
        image_id, caption_number = self._pair_from_index(index)
        return self._get_image(image_id), self._get_caption(image_id, caption_number), self.annots[image_id]

    def __len__(self):
        return self.num_images * self.cap_per_image

    def _get_image(self, image_id):
        image_file = os.path.join(self.image_folder, str(image_id)+'.jpg')  # get the absolute path of the image
        with open(image_file, 'rb') as file:
            image = Image.open(file).convert('RGB')
        return self.transform(image)

    def _get_caption(self, image_id, caption_number):
        caption_file = os.path.join(self.caption_folder, str(image_id)+'.txt')
        with open(caption_file, 'r') as file:
            for i, cap in enumerate(file):
                if i == caption_number:
                    caption = cap
                    break
        caption = self.vocab.id_tokenize(caption)
        return torch.Tensor(caption).int()

    def _index_from_pair(self, pair):
        # pair is a (image_id, caption_number) pair and we want the index of the item in the dataset
        return 10 * pair[0] + pair[1]

    def _pair_from_index(self, index):
        # get the (image_id, caption_number) pair from the index
        caption_number = index % 10
        image_id = (index - caption_number) // 10
        return image_id, caption_number

    def _set_loaders(self, train_test_split, batch_size, workers):
        train_split, val_split, test_split = train_test_split[0], 1-sum(train_test_split), train_test_split[1]
        assert train_split > 0 and val_split >= 0 and train_split + val_split <= 1

        num_samples = len(self)
        indices = list(range(num_samples))
        random.shuffle(indices)
        train_indices = indices[:int(num_samples*train_split)]
        val_indices = indices[int(num_samples*train_split) : int(num_samples*(train_split+val_split))]
        test_indices = indices[int(num_samples*(train_split+val_split)) : num_samples]

        train_data = Subset(dataset=self, indices=train_indices)
        val_data = Subset(dataset=self, indices=val_indices)
        test_data = Subset(dataset=self, indices=test_indices)

        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=collate_annots)
        self.val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=collate_annots)
        self.test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=collate_annots)