import torch
from sklearn.metrics import accuracy_score
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
import dataset
from face_emotion.structure import cnn

transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
])
train_dataset = dataset.dataset("C:\pytorchawa/face_emotion/train", transform)
test_dataset = dataset.dataset("C:\pytorchawa/face_emotion/test", transform)

model = cnn(class_num=len(train_dataset.class_index))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

num_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
running_loss = 0
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    average_loss = running_loss / len(train_loader)

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

    accuracy = correct / total
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}")

torch.save(model.state_dict(), "emotion.pth")
