from gensim.models import Word2Vec, KeyedVectors
import json

with open('awa.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

question = data[0]["questions"]
answer = data[0]["answers"]
s = [q.split() + a.split() for q, a in zip(question, answer)]
model = Word2Vec(s, vector_size=20, window=5, min_count=1, workers=4)
model.wv.save_word2vec_format('seq.bin', binary=True)
model = KeyedVectors.load_word2vec_format('seq.bin', binary=True)
# word_vector = model['kitchen']
# print("Vector for 'some_word':", word_vector)




