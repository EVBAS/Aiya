import torch
import os
from PIL import Image

from torch.utils.data import Dataset
class dataset(Dataset):
    def __init__(self,data_root,transform):
        super(dataset,self).__init__()

        self.root = data_root
        self.transform = transform
        self.class_index,self.label = self.classify_class()
        self.img = self.load()

    def classify_class(self):
        class_index = [d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))]
        class_index.sort()
        label = {class_index[i]:i for i in range(len(class_index))}
        return class_index,label
# datasetw = dataset("C:\pytorchawa/face_emotion/train",0)
# awa,aka = datasetw.classify_class()
# print(awa) #['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
# print(aka) #{'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}
    def load(self):
        img = []
        for class_ in self.class_index:
            class_dir = os.path.join(self.root,class_) #['C:\\pytorchawa/face_emotion/train\\angry', 'C:\\pytorchawa/face_emotion/train\\disgust', 'C:\\pytorchawa/face_emotion/train\\fear', 'C:\\pytorchawa/face_emotion/train\\happy', 'C:\\pytorchawa/face_emotion/train\\neutral', 'C:\\pytorchawa/face_emotion/train\\sad', 'C:\\pytorchawa/face_emotion/train\\surprise']
            for root, _, fnames in sorted(os.walk(class_dir)):
                for i in fnames:
                    path = os.path.join(root,i)
                    item = (path,self.label[class_])
                    img.append(item)

        return img

    def __len__(self):
        return len(self.img)

    def __getitem__(self,index):
        path, target = self.img[index]
        convert = Image.open(path).convert("RGB")
        target = torch.tensor(target)
        if self.transform is not None:
            img = self.transform(convert)

        return img, target



# datasetw = dataset("C:\pytorchawa/face_emotion/train",0)
# awa,aka = datasetw.__getitem__(2133)
# print(awa.shape,aka)
