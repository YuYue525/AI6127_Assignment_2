from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
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

def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')[:2]] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

MAX_LENGTH = 15

eng_prefixes = (
    "i am", "i m",
    "he is", "he s",
    "she is", "she s",
    "you are", "you re",
    "we are", "we re",
    "they are", "they re"
)


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
#print(random.choice(pairs))

from sklearn.model_selection import train_test_split

X = [i[0] for i in pairs]
y = [i[1] for i in pairs]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

train_pairs = list(zip(X_train,y_train))
test_pairs = list(zip(X_test,y_test))

import math
class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p = 0, max_len = MAX_LENGTH):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)
        
        # Info
        self.dropout = nn.Dropout(dropout_p)
        
        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
        
        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        
        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding",pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.positional_encoder = PositionalEncoding(dim_model=hidden_size)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)


    def forward(self, input_tensor):
        input_length = input_tensor.size(0)
        embedded = self.embedding(input_tensor).view(input_length, 1, -1)
        embedded = self.positional_encoder(embedded)
        output = self.transformer_encoder(embedded)
        return torch.mean(output, dim = 0, keepdim = True)

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):

        # Your code here #
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

teacher_forcing_ratio = 0.5

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    loss = 0

    encoder_output = encoder(input_tensor)

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_output

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

import time
import math


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

import matplotlib.pyplot as plt

def trainIters(encoder, decoder, epochs, print_every=1000, plot_every=1000, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr = learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr = learning_rate)

    encoder_scheduler = optim.lr_scheduler.CosineAnnealingLR(encoder_optimizer, T_max=20)
    decoder_scheduler = optim.lr_scheduler.CosineAnnealingLR(decoder_optimizer, T_max=20)

    criterion = nn.NLLLoss()

    iter = 1
    n_iters = len(train_pairs) * epochs

    for epoch in range(epochs):
        print("Epoch: %d/%d" % (epoch, epochs))
        for training_pair in train_pairs:
            training_pair = tensorsFromPair(training_pair)

            input_tensor = training_pair[0]
            target_tensor = training_pair[1]

            loss = train(input_tensor, target_tensor, encoder,
                        decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                            iter, iter / n_iters * 100, print_loss_avg))
            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_loss_total = 0
                plot_losses.append(plot_loss_avg)

            iter +=1

        encoder_scheduler.step()
        decoder_scheduler.step()

    plt.figure()
    plt.title("loss curves")
    plt.plot([x * plot_every for x in range(1, len(plot_losses) + 1)], plot_losses, label="loss")
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig("loss_task_5.png")

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_output = encoder(input_tensor)

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_output

        decoded_words = []

        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words

def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

import numpy as np
from torchmetrics.text.rouge import ROUGEScore
rouge = ROUGEScore()

def test(encoder, decoder, testing_pairs):
    input = []
    gt = []
    predict = []
    metric_score = {
        "rouge1_fmeasure":[],
        "rouge1_precision":[],
        "rouge1_recall":[],
        "rouge2_fmeasure":[],
        "rouge2_precision":[],
        "rouge2_recall":[]
    }

    for pair in testing_pairs:
        output_words = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)

        input.append(pair[0])
        gt.append(pair[1])
        predict.append(output_sentence)
        metric_score["rouge1_fmeasure"].append(rouge(output_sentence, pair[1])['rouge1_fmeasure'])
        metric_score["rouge1_precision"].append(rouge(output_sentence, pair[1])['rouge1_precision'])
        metric_score["rouge1_recall"].append(rouge(output_sentence, pair[1])['rouge1_recall'])
        metric_score["rouge2_fmeasure"].append(rouge(output_sentence, pair[1])['rouge2_fmeasure'])
        metric_score["rouge2_precision"].append(rouge(output_sentence, pair[1])['rouge2_precision'])
        metric_score["rouge2_recall"].append(rouge(output_sentence, pair[1])['rouge2_recall'])
        
    metric_score["rouge1_fmeasure"] = np.array(metric_score["rouge1_fmeasure"]).mean()
    metric_score["rouge1_precision"] = np.array(metric_score["rouge1_precision"]).mean()
    metric_score["rouge1_recall"] = np.array(metric_score["rouge1_recall"]).mean()
    metric_score["rouge2_fmeasure"] = np.array(metric_score["rouge2_fmeasure"]).mean()
    metric_score["rouge2_precision"] = np.array(metric_score["rouge2_precision"]).mean()
    metric_score["rouge2_recall"] = np.array(metric_score["rouge2_recall"]).mean()

    print("=== Evaluation score - Rouge score ===")
    print("Rouge1 fmeasure:\t",metric_score["rouge1_fmeasure"])
    print("Rouge1 precision:\t",metric_score["rouge1_precision"])
    print("Rouge1 recall:  \t",metric_score["rouge1_recall"])
    print("Rouge2 fmeasure:\t",metric_score["rouge2_fmeasure"])
    print("Rouge2 precision:\t",metric_score["rouge2_precision"])
    print("Rouge2 recall:  \t",metric_score["rouge2_recall"])
    print("=====================================")
    return input,gt,predict,metric_score

hidden_size = 512

encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
decoder1 = Decoder(hidden_size, output_lang.n_words).to(device)

trainIters(encoder1, decoder1, 20, print_every=5000)

input,gt,predict,score = test(encoder1, decoder1, train_pairs)

input,gt,predict,score = test(encoder1, decoder1, test_pairs)


'''
hidden_size = 512
test_num = 5

train_r1 = [0, 0, 0]
train_r2 = [0, 0, 0]
test_r1 = [0, 0, 0]
test_r2 = [0, 0, 0]

for i in range(test_num):

    encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    decoder1 = Decoder(hidden_size, output_lang.n_words).to(device)

    trainIters(encoder1, decoder1, 10, print_every=5000)

    input,gt,predict,score = test(encoder1, decoder1, train_pairs)

    train_r1[0] += score["rouge1_fmeasure"]
    train_r1[1] += score["rouge1_precision"]
    train_r1[2] += score["rouge1_recall"]
    train_r2[0] += score["rouge2_fmeasure"]
    train_r2[1] += score["rouge2_precision"]
    train_r2[2] += score["rouge2_recall"]

    input,gt,predict,score = test(encoder1, decoder1, test_pairs)

    test_r1[0] += score["rouge1_fmeasure"]
    test_r1[1] += score["rouge1_precision"]
    test_r1[2] += score["rouge1_recall"]
    test_r2[0] += score["rouge2_fmeasure"]
    test_r2[1] += score["rouge2_precision"]
    test_r2[2] += score["rouge2_recall"]

print("\n=======final average scores=======")
print("train Rouge1 fmeasure:\t", train_r1[0] / test_num)
print("train Rouge1 precision:\t", train_r1[1] / test_num)
print("train Rouge1 recall:  \t", train_r1[2] / test_num)
print("train Rouge2 fmeasure:\t", train_r2[0] / test_num)
print("train Rouge2 precision:\t", train_r2[1] / test_num)
print("train Rouge2 recall:  \t", train_r2[2] / test_num)
print("------------------------------------")
print("test Rouge1 fmeasure:\t", test_r1[0] / test_num)
print("test Rouge1 precision:\t", test_r1[1] / test_num)
print("test Rouge1 recall:  \t", test_r1[2] / test_num)
print("test Rouge2 fmeasure:\t", test_r2[0] / test_num)
print("test Rouge2 precision:\t", test_r2[1] / test_num)
print("test Rouge2 recall:  \t", test_r2[2] / test_num)
print("====================================")
'''
