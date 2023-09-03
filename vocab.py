import nltk

import pickle

from collections import Counter

import os
# vocab_folder = '/Users/sokhnadiarrambacke/Project-7030/multi-modal-celeb/CelebProject1/vocab/'
# captions_folder = '/Users/sokhnadiarrambacke/Project-7030/multi-modal-celeb/mm-celebA-data/text/celeba-caption/'


class Vocab:

    def __init__(self, caption_folder, min_frequency=3):
        self.caption_folder = caption_folder

        self.word_to_id = {}
        self.id_to_word = {}
        self.max_id = 0
        self.min_frequency = min_frequency

        self.start_token = '<start>'
        self.end_token = '<end>'
        self.unk_token = '<unk>'
        self.pad_token = '<pad>'

        self._build()

    def add_word(self, word):
        if word not in self.word_to_id.keys():
            self.word_to_id[word] = self.max_id
            self.id_to_word[self.max_id] = word
            self.max_id += 1

    def get_id(self, word):
        if word in self.word_to_id.keys():
            idx = self.word_to_id[word]
        else:
            idx = self.word_to_id[self.unk_token]
        return idx

    def _build(self):
        counter = Counter()
        num_files = len([name for name in os.listdir(self.caption_folder)])
        for id in range(num_files):
            file_name = os.path.join(self.caption_folder, str(id)+'.txt')
            with open(file_name, 'r') as file:
                captions = [x[:-1].lower() for x in file.readlines()]  # Remove the \n at the end

            for caption in captions:
                tokens = nltk.tokenize.word_tokenize(caption)
                counter.update(tokens)

        # Keep words that appear at least min_frequency times in the file
        words = [word for word, freq in counter.items() if freq >= self.min_frequency]

        # Add the basic tokens
        self.add_word(self.pad_token)
        self.add_word(self.unk_token)
        self.add_word(self.start_token)
        self.add_word(self.end_token)

        # Add the remaining words
        for word in words:
            self.add_word(word)
        print('Successfully built vocabulary!')

    def __len__(self):
        return len(self.word_to_id)

    def id_tokenize(self, caption):
        # Tokenizes a caption and adds the <start> and <end> tags. Returns a list of idx (int)
        idx = []
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        idx.append(self.get_id(self.start_token))
        idx.extend([self.get_id(token) for token in tokens])
        idx.append(self.get_id(self.end_token))

        return idx

