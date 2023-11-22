import collections
import  math

import numpy as np
import torch
from torch import nn

class attention(nn.Module):
    def __init__(self, hidden_size):
        super(attention, self).__init__()
        self.linear = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, decoder_hidden, encoder_outputs):
        attn_scores = torch.bmm(encoder_outputs.permute(1, 0, 2), decoder_hidden.unsqueeze(2)).squeeze(2)
        attn_weights = nn.functional.softmax(attn_scores, dim=0)

        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs.permute(1, 0, 2)).squeeze(1)

        attended_input = torch.cat([context, decoder_hidden], dim=1)
        attention_output = torch.tanh(self.linear(attended_input))

        return attention_output

class encoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, emb_size, layer_num, dropout=0):
        super(encoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.gru = nn.GRU(emb_size, hidden_size, layer_num, dropout=dropout)

    def forward(self, input_seq):
        if len(input_seq.shape) == 3:
            input_seq = input_seq.view(-1, input_seq.size(2))
        # print(input_seq)
        seq = self.embedding(input_seq.long().clone().detach())
        seq = seq.permute(1, 0, 2)
        output, hidden = self.gru(seq)
        # print(output.shape)
        # print(hidden.shape)
        return output, hidden
        
class decoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, emb_size, layer_num, dropout=0):
        super(decoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.gru = nn.GRU(emb_size + hidden_size, hidden_size, layer_num, dropout=dropout)
        self.com = nn.Linear(hidden_size, vocab_size)
        self.attention = attention(hidden_size)

    def init_state(self, enc_output):
        return enc_output[1]

    def forward(self, input_seq, state, encoder_outputs, word2vec_model):
        if len(input_seq.shape) == 3:
            input_seq = input_seq.view(-1, input_seq.size(2))

        seq = self.embedding(input_seq.long().clone().detach())
        seq = seq.permute(1, 0, 2)

        attention_output = self.attention(state[-1], encoder_outputs)
        attention_output = attention_output.unsqueeze(0)
        # print("seq size:", seq.size())
        # print("attention_output size:", attention_output.size())
        # print("seq.size(0):", seq.size(0))
        seq = torch.cat([seq, attention_output.repeat(seq.size(0), 1, 1)], dim=2)
        output, hidden = self.gru(seq, state)
        output = self.com(output).permute(1, 0, 2)
        pos_word = nn.functional.softmax(output, dim=2)

        batch_size, max_length, vocab_size = pos_word.shape
        words = []

        for i in range(batch_size):
            sentence_words = []
            for j in range(max_length):
                word_index = np.random.choice(vocab_size, p=pos_word[i, j, :].detach().numpy())
                word = word2vec_model.index_to_key[word_index]
                sentence_words.append(word)

            words.append(sentence_words)
            sentence = [" ".join(word) for word in words]

        return output, hidden, pos_word, sentence
      
