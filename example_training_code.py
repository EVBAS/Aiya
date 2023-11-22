import torch
import torch.nn as nn
import torch.optim as optim
from gensim.models import KeyedVectors
import dataset
import structure

dataset_path = "awa.json"
seq_model = model = KeyedVectors.load_word2vec_format('seq.bin', binary=True)
ex_dataset = dataset.dataset(dataset_path,seq_model)
dataloader = torch.utils.data.DataLoader(ex_dataset,batch_size=64,shuffle=True,collate_fn=dataset.collate_fn)
vocab_size = 64
hidden_size = 128
emb_size = 64
layer_num = 2

encoder = structure.encoder(vocab_size,hidden_size,emb_size,layer_num)
decoder = structure.decoder(vocab_size,hidden_size,emb_size,layer_num)

criterion = nn.CrossEntropyLoss()
par = list(encoder.parameters())+list(decoder.parameters())
optim = optim.Adam(par,lr=0.001)

epoch_num = 1000
total_loss = 0
for i in range(epoch_num):
    for batch_index,(q,a) in enumerate(dataloader):
        optim.zero_grad()
        # print(q.shape,a.shape)
        en_output,en_hidden = encoder((q))
        # print(en_output.shape)
        de_state = decoder.init_state((en_output,en_hidden))
        de_output,de_hidden,tensor,text_output = decoder(a,de_state,en_output,seq_model)
        de_output = de_output.permute(0, 2, 1)

        tar = a.long()
        # print(tar.shape,q.shape)
        # print(de_output.shape,tar.shape)
        loss = criterion(de_output.reshape(-1,vocab_size), tar.reshape(-1))
        total_loss += loss
        loss.backward()
        optim.step()
        # print(batch_index)
    if i % 10  == 0:
        print(f'Epoch [{i}/{epoch_num}],Loss: {loss}',text_output)
