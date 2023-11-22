import torch
from gensim.models import Word2Vec, KeyedVectors
from torch.utils.data import Dataset
import json
import numpy as np

class dataset(Dataset): #for a example awa
    def __init__(self, json_path, model):
        self.model = model

        with open(json_path, 'r', encoding='utf-8') as file:
            self.data = json.load(file)

        self.questions = self.data[0]["questions"]
        self.answers = self.data[0]["answers"]

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        answer = self.answers[idx]

        question_vector = [self.model[word] for word in question.split()]
        answer_vector = [self.model[word] for word in answer.split()]

        question_tensor = torch.tensor(np.array(question_vector),dtype=torch.float32)
        answer_tensor = torch.tensor(np.array(answer_vector), dtype=torch.float32)

        # print(question_tensor)
        return question_tensor, answer_tensor

def collate_fn(batch):
    questions, answers = zip(*batch)

    max_len_question = max(q.size(0) for q in questions)
    max_len_answer = max(a.size(0) for a in answers)
    max_len = max(max_len_question, max_len_answer)

    padded_questions = [torch.cat([q, torch.zeros(max_len - q.size(0), q.size(1))]) for q in questions]
    padded_answers = [torch.cat([a, torch.zeros(max_len - a.size(0), a.size(1))]) for a in answers]

    padded_questions = torch.stack(padded_questions)
    padded_answers = torch.stack(padded_answers)
    return padded_questions, padded_answers
#
# model = KeyedVectors.load_word2vec_format('seq.bin', binary=True)
#
# json_path = 'awa.json'
# custom_dataset = dataset(json_path,model)
# dataloader = torch.utils.data.DataLoader(custom_dataset,16,collate_fn=collate_fn)
# for batch in dataloader:
#         questions, answers = batch
#         print(questions.shape)
#         print(answers.shape)
