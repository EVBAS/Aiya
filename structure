import torch
import torch.nn as nn
import torch.nn.functional as F

class HanAttentionEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(HanAttentionEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.self_attention = nn.MultiheadAttention(hidden_size, num_heads=1)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, encoder_input, gpu):
        encoder_input = encoder_input.to(gpu)
        embedded = self.embedding(encoder_input).view(1, -1, self.hidden_size)
        embedded = F.relu(embedded)
        attn_output, _ = self.self_attention(embedded, embedded, embedded)
        output = F.relu(attn_output)
        output, hidden = self.gru(output)

        return output, hidden

class HanAttentionDecoder(nn.Module):
    def __init__(self, output_size, hidden_size):
        super(HanAttentionDecoder, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, decoder_input, hidden, encoder_output, gpu):
        decoder_input = decoder_input.to(gpu)
        hidden = hidden.to(gpu)
        encoder_output = encoder_output.to(gpu)
        embedded = self.embedding(decoder_input).view(1, -1, self.hidden_size)
        attn_weights = torch.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_output.transpose(0, 1))
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = torch.sigmoid(self.fc(output[0]))

        return output, hidden, attn_weights

class Seq2Seq(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Seq2Seq, self).__init__()
        self.encoder = HanAttentionEncoder(input_size, hidden_size)
        self.decoder = HanAttentionDecoder(output_size, hidden_size)

    def forward(self, input_seq, target_seq, gpu):
        encoder_output, encoder_hidden = self.encoder(input_seq, gpu)
        decoder_input = target_seq[:, 0].unsqueeze(1)
        decoder_hidden = encoder_hidden
        loss = 0
        for di in range(target_seq.size(1)):
            decoder_output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_output, gpu)
            loss += F.cross_entropy(decoder_output.squeeze(0), target_seq[:, di])
            decoder_input = target_seq[:, di].unsqueeze(1)

        return loss
      
