import os
from copy import deepcopy

import torch
import torch.utils.data as data
from tqdm import tqdm

from config import load_dictionary
from model.tree import Tree
import torch.nn.functional as F
from data import constants


class SSTDataset(data.Dataset):
    """
    A wrapper class for dataset in the format of Stanford Sentiment Treebank 
    (SST) (https://nlp.stanford.edu/sentiment/)
    """

    def __init__(self, path=None, vocab=None, num_classes=None, dictionaries = []):
        super(SSTDataset, self).__init__()

        self.num_classes = num_classes
        if not path and not vocab:
            return

        self.vocab = vocab
        self.num_classes = num_classes
        skladnica_sentences, skladnica_trees, skladnica_dict = self.create_trees(path, 'sklad', dictionaries)

        reviews_sentences, reviews_trees, reviews_dict = self.create_trees(path, 'rev', dictionaries)

        test_sentences, test_trees, test_dict = self.create_trees(path, 'polevaltest', dictionaries)
        self.dict = None

        if test_trees:
            self.trees = test_trees
            self.sentences = test_sentences
            if test_dict:
                self.dict = test_dict
            self.labels = []
            for i in range(0, len(self.trees)):
                self.labels.append(self.trees[i].gold_label)
        else:
            self.trees = skladnica_trees + reviews_trees  # list concatenation
            self.sentences = skladnica_sentences + reviews_sentences
            if skladnica_dict or reviews_dict:
                self.dict = skladnica_dict + reviews_dict
            self.labels = []

            for i in range(0, len(self.trees)):
                self.labels.append(self.trees[i].gold_label)

                # shuffle
            # self.trees, self.sentences, self.labels = shuffle(self.trees,
            #                                                   self.sentences,
            #                                                   self.labels)

        self.labels = torch.Tensor(self.labels)  # let labels be tensor

    @classmethod
    def create_dataset_from_user_input(cls, sentence_path, parents_path,
                                       vocab=None, num_classes=None,dictionaries = [], wordnet=None):
        dataset = cls()
        dataset.vocab = vocab
        dataset.num_classes = num_classes
        parents_file = open(parents_path, 'r', encoding='utf-8')
        tokens_file = open(sentence_path, 'r', encoding='utf-8')
        dataset.trees = [
            dataset.read_tree(parents, 0, tokens, tokens)
            for parents, tokens in zip(parents_file.readlines(),
                                       tokens_file.readlines())
        ]
        dataset.sentences, dict = dataset.read_sentences(sentence_path, dictionaries)
        dataset.labels = torch.Tensor(len(dataset.sentences))
        if wordnet:
            dict = get_dict_sentiments(wordnet, dict)
        dataset.dict = dict
        return dataset

    def __len__(self):
        return len(self.trees)

    def __getitem__(self, index):
        tree = deepcopy(self.trees[index])
        sent = deepcopy(self.sentences[index])
        label = deepcopy(self.labels[index])

        dict = deepcopy(self.dict[index]) if self.dict else None
        return tree, sent, dict, label

    def create_trees(self, path, file_type, dictionaries):
        if os.path.isfile(os.path.join(path, file_type + '_sentence.txt')):
            sentences, dictionary_sentiments = self.read_sentences(
                os.path.join(path, file_type + '_sentence.txt'), dictionaries
            )
            trees = self.read_trees(
                filename_parents=os.path.join(path, file_type + '_parents.txt'),
                filename_labels=os.path.join(path, file_type + '_labels.txt'),
                filename_tokens=os.path.join(path, file_type + '_sentence.txt'),
                filename_relations=os.path.join(path, file_type + '_rels.txt'),
            )
            wordnet_filename = os.path.join(path, file_type + '_wordnet.txt')

            if os.path.exists(wordnet_filename):
                sentiments = get_dict_sentiments(wordnet_filename, dictionary_sentiments)
                return sentences, trees, sentiments
            else:
                return sentences, trees, dictionary_sentiments
        else:
            return None, None, None

    def read_sentences(self, filename, dictionaries):
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            sentences = [self.read_sentence(line) for line in tqdm(lines)]

            sentiments = [self.get_sentiments(line, dictionaries) for line in tqdm(lines)] if dictionaries else None


        return sentences, sentiments

    def get_sentiments(self, line, dictionaries):
        sentiments = []
        for dictionary in dictionaries:
            values = [dictionary.get(word,0) for word in line.split()]
            values = F.torch.unsqueeze(torch.FloatTensor(values), 1)
            sentiments.append(values)
        return torch.cat(sentiments, 1)

    def read_sentence(self, line):
        indices = self.vocab.convert_to_idx(line.split(), constants.UNK_WORD)
        return torch.LongTensor(indices)

    def read_trees(self, filename_parents, filename_labels, filename_tokens,
                   filename_relations):
        parents_file = open(filename_parents, 'r', encoding='utf-8')
        tokens_file = open(filename_tokens, 'r', encoding='utf-8')
        relations_file = open(filename_relations, 'r', encoding='utf-8')
        if filename_labels:
            labels_file = open(filename_labels, 'r', encoding='utf-8')
            iterator = zip(parents_file.readlines(), labels_file.readlines(),
                           tokens_file.readlines(), relations_file.readlines())
            trees = [self.read_tree(parents, labels, tokens, relations)
                     for parents, labels, tokens, relations in tqdm(iterator)]
        else:
            iterator = zip(parents_file.readlines(), tokens_file.readlines(), relations_file.readlines())
            trees = [self.read_tree(parents, None, tokens, relations)
                     for parents, tokens, relations in tqdm(iterator)]

        return trees

    def parse_label(self, label):
        return int(label) + 1

    def read_tree(self, line_parents, line_label, line_words, line_relations):
        parents = list(map(int, line_parents.split()))
        if line_label:
            labels = list(map(self.parse_label, line_label.split()))
        else:
            labels = None
        words = line_words.split()
        relations = line_relations.split()
        trees = dict()
        root = None

        for i in range(1, len(parents) + 1):
            if i not in trees.keys():
                idx = i
                prev = None
                while True:
                    parent = parents[idx - 1]
                    tree = Tree()
                    if prev:
                        tree.add_child(prev)
                    trees[idx] = tree
                    tree.idx = idx
                    if labels:
                        tree.gold_label = labels[idx - 1]
                    else:
                        tree.gold_label = None
                    tree.word = words[idx - 1]
                    tree.relation = relations[idx - 1]
                    if parent in trees.keys():
                        trees[parent].add_child(tree)
                        break
                    elif parent == 0:
                        root = tree
                        break
                    else:
                        prev = tree
                        idx = parent
        # helper for visualization
        root._viz_all_children = trees
        root._viz_sentence = words
        root._viz_relations = relations
        root._viz_labels = labels
        return root

def get_dict_sentiments(wordnet_filename, dictionaries):
    sentiments = []
    with open(wordnet_filename, 'r', encoding='utf-8') as f:
        wordnet_sentiments = [F.torch.unsqueeze(torch.FloatTensor(list(map(int, line.split()))), 1) for line in
                              f.readlines()]
    if dictionaries:
        for idx, _ in enumerate(wordnet_sentiments):
            sentiments.append(torch.cat((dictionaries[idx], wordnet_sentiments[idx]), 1))
    else:
        sentiments = wordnet_sentiments
    return sentiments
