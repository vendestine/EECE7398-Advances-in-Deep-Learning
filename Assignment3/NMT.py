# -------------------------- import ----------------------------------
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from collections import Counter

# system
import os
import sys
import glob

# print time
import time
import math

# plot
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# BLEU tests
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# ------------------------------------ parameters -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

SOS_token = 0
EOS_token = 1

ENG_train = '../ev_dataset/train.en.txt'
ENG_test = '../ev_dataset/tst2013.en.txt'
VI_train = '../ev_dataset/train.vi.txt'
VI_test = '../ev_dataset/tst2013.vi.txt'
ENG_vocab = '../ev_dataset/vocab.en.txt'
VI_vocab = '../ev_dataset/vocab.vi.txt'

MAX_LENGTH = 10

teacher_forcing_ratio = 0.5

plt.switch_backend('agg')


# ---------------------------------------- prepare the data ---------------------------------
class Lang:
    def __init__(self, name, vocab_path):
        self.name = name
        self.word2index = {'<unk>': 2}
        self.index2word = {0: "SOS", 1: "EOS", 2: '<unk>'}
        self.n_words = 3  # Count <unk>, SOS and EOS
        self.vocab_path = vocab_path

    def loadVocab(self):
        with open(self.vocab_path, encoding="utf-8") as f:
            data = f.readlines()
            vocab = list(map(lambda word: word[:-1], data))
        vocab_norm = [normalizeString(word) for word in vocab]
        unique_vocab_norm = list(Counter(vocab_norm).keys())
        for word in unique_vocab_norm:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def readLangs(lang1, lang2, reverse=False, vocab1=None, vocab2=None):
    print("Reading lines...")

    # Read the file and split into lines
    eng_lines = open(lang1, encoding='utf-8').read().strip().split('\n')
    vi_lines = open(lang2, encoding='utf-8').read().strip().split('\n')

    lines = [eng_lines + '\t' + vi_lines for eng_lines, vi_lines in zip(eng_lines, vi_lines)]

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2, vocab2)
        output_lang = Lang(lang1, vocab1)
    else:
        input_lang = Lang(lang1, vocab1)
        output_lang = Lang(lang2, vocab2)

    return input_lang, output_lang, pairs


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
           len(p[1].split(' ')) < MAX_LENGTH


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang1, lang2, reverse=False, vocab1=None, vocab2=None):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse, vocab1=vocab1, vocab2=vocab2)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    input_lang.loadVocab()
    output_lang.loadVocab()
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


# ------------------------- The Seq2Seq Model ---------------------------------
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# ------------------------- Preparing Training Data ---------------------------------
def indexesFromSentence(lang, sentence):
    indexes = []
    for word in sentence.split(' '):
        if word in lang.word2index:
            indexes.append(lang.word2index[word])
        else:
            indexes.append(lang.word2index['<unk>'])
    return indexes


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


# ---------------------------------- helper function ---------------------------------------

# print time elapsed and estimated time
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


# save and load model
def save_model(model, path):
    # creat model directory if not exist
    if not os.path.isdir('./model'):
        os.mkdir('./model')

    # save new model
    torch.save(model.state_dict(), path)
    print("model saved in file: %s" % path)


def load_model(model, path):
    # print("loading Model %s" % path)
    model.load_state_dict(torch.load(path))


# ---------------------------- Training the Model --------------------------------------
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('Time: %s  Loss: (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                                      iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

        if iter == n_iters:
            save_model(encoder, './model/encoder.pth')
            save_model(decoder, './model/decoder.pth')


# ------------------------------------ Evaluation -----------------------------------------
def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


# ------------------------------------- calculate BLEU score ---------------------------------
def calculateBleu(encoder, decoder, pairs):
    bleu = 0
    for pair in pairs:
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        bleu += sentence_bleu([output_sentence], pair[1], smoothing_function=SmoothingFunction().method1)
    return bleu / len(pairs)


def translation(encoder, decoder):
    print("Loading Model...")
    load_model(encoder, './model/encoder.pth')
    load_model(decoder, './model/decoder.pth')

    while True:
        print(">", end=' ')
        sentence = str(input())
        decoded_words, _ = evaluate(encoder, decoder, sentence)
        output_sentence = ' '.join(decoded_words)
        print(output_sentence)


if __name__ == '__main__':
    input_lang, output_lang, pairs = prepareData(ENG_train, VI_train, False, ENG_vocab, VI_vocab)
    print(random.choice(pairs))

    hidden_size = 256
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

    option = sys.argv[1]
    if option == "train":
        trainIters(encoder1, attn_decoder1, 75000, print_every=5000)
        # evaluateRandomly(encoder1, attn_decoder1)
    elif option == "test":
        print("Loading Model...")
        load_model(encoder1, './model/encoder.pth')
        load_model(attn_decoder1, './model/decoder.pth')
        _, _, test_pairs = prepareData(ENG_test, VI_test, False, ENG_vocab, VI_vocab)
        print("Average BLEU: %s" % str(calculateBleu(encoder1, attn_decoder1, test_pairs)))
    elif option == "translate":
        translation(encoder1, attn_decoder1)
