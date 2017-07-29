from __future__ import print_function

import os
import torch.optim as optim
import gc
import subprocess
import numpy as np

from model import *
from utils import load_word_vectors
from sentiment_trainer import SentimentTrainer


def choose_optimizer(args, model):

    if args.optim =='adam':
        return optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)
    elif args.optim=='adagrad':
        # optimizer   = optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)
        return optim.Adagrad([
                {'params': model.parameters(), 'lr': args.lr}
            ], lr=args.lr, weight_decay=args.wd)


def apply_not_known_words(emb,args, not_known,vocab):
    new_words = 'new_words.txt'
    f = open(new_words, 'w')
    for item in not_known:
        f.write("%s\n" % item)
    cmd = " ".join(["./../fastText/fasttext", "print-word-vectors",
                    args.emb_dir + "/" + args.emb_file + ".bin", "<", new_words])
    print(cmd)
    ps = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = ps.communicate()[0]
    new_words_embeddings = [x.split(" ")[:-1] for x in output.decode("utf-8").split("\n")]
    for word in new_words_embeddings:
        if args.input_dim == len(word[1:]):
            emb[vocab.get_index(word[0])] = torch.from_numpy(np.asarray(list(map(float, word[1:]))))
        else:
            print('Word embedding from subproccess has different length than expected')
    os.remove(new_words)
    return emb


def load_embedding_model(args,vocab):
    embedding_model = nn.Embedding(vocab.size(), args.input_dim)

    if args.cuda:
        embedding_model = embedding_model.cuda()
    # for words common to dataset vocab and GLOVE, use GLOVE vectors
    # for other words in dataset vocab, use random normal vectors
    emb_file = os.path.join(args.data, args.emb_dir.split("/")[-1]+"_"+args.emb_file + '_emb.pth')
    if os.path.isfile(emb_file) and torch.load(emb_file).size()[1] == args.input_dim:
        emb = torch.load(emb_file)
    else:
        # load glove embeddings and vocab
        # args.glove = "data/glove"
        # glove_vocab, glove_emb = load_word_vectors(os.path.join(args.glove,'glove.twitter.27B.25d'))
        glove_vocab, glove_emb = load_word_vectors(os.path.join(args.emb_dir,args.emb_file))
        print('==> GLOVE vocabulary size: %d ' % glove_vocab.size())

        emb = torch.zeros(vocab.size(), glove_emb.size(1))
        not_known = []
        for word in vocab.label_to_idx.keys():
            if glove_vocab.get_index(word):
                emb[vocab.get_index(word)] = glove_emb[glove_vocab.get_index(word)]
            else:
                not_known.append(word)
                emb[vocab.get_index(word)] = torch.Tensor(emb[vocab.get_index(word)].size()).normal_(-0.05, 0.05)
        if args.calculate_new_words:
            emb = apply_not_known_words(emb,args, not_known,vocab)

        torch.save(emb, emb_file)

    # plug these into embedding matrix inside model
    if args.cuda:
        emb = emb.cuda()

    embedding_model.state_dict()['weight'].copy_(emb)
    return embedding_model


def train(train_dataset, dev_dataset, vocab, args):
    # Optionally reweight loss per class to the distribution of classes in
    # the public dataset
    weight = torch.Tensor([1/0.024, 1/0.820, 1/0.156]) if args.reweight else None
    criterion = nn.NLLLoss(weight=weight)

    # initialize model, criterion/loss_function, optimizer

    model = TreeLSTMSentiment(
        cuda=args.cuda,
        in_dim=args.input_dim,
        mem_dim=args.mem_dim,
        num_classes=args.num_classes,
        criterion=criterion
    )

    if args.cuda:
        model.cuda()
        criterion.cuda()

    optimizer = choose_optimizer(args,model)

    embedding_model = load_embedding_model(args,vocab)

    # create trainer object for training and testing
    trainer = SentimentTrainer(args, model, embedding_model ,criterion, optimizer)

    max_dev = 0
    max_dev_epoch = 0
    filename = args.name + '.pth'
    for epoch in range(args.epochs):
        train_loss = trainer.train(train_dataset)
        dev_loss, dev_acc, _ = trainer.test(dev_dataset)
        dev_acc = torch.mean(dev_acc)
        print('==> Train loss   : %f \t' % train_loss, end="")
        print('Epoch ', epoch, 'dev percentage ', dev_acc)
        torch.save(model, args.saved + str(epoch) + '_model_' + filename)
        torch.save(embedding_model, args.saved + str(epoch) + '_embedding_' + filename)
        if dev_acc > max_dev:
            max_dev = dev_acc
            max_dev_epoch = epoch
        gc.collect()
    print('epoch ' + str(max_dev_epoch) + ' dev score of ' + str(max_dev))

    return max_dev_epoch, max_dev

