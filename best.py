import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import pandas as pd
from PIL import Image
from torchvision import transforms
import numpy as np
import time
from statistics import mode
import random
import torch.nn as nn

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class VQADataset(Dataset):
    def __init__(self, df_path, image_dir, transform=None, answer=True):
        self.transform = transform
        self.image_dir = image_dir
        self.df = pd.read_json(df_path)
        self.answer = answer

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

        self.answer2idx = {}
        self.idx2answer = {}

        if self.answer:
            for answers in self.df["answers"]:
                for answer in answers:
                    word = answer["answer"].lower()
                    if word not in self.answer2idx:
                        self.answer2idx[word] = len(self.answer2idx)
            self.idx2answer = {v: k for k, v in self.answer2idx.items()}

    def update_dict(self, dataset):
        self.answer2idx = dataset.answer2idx
        self.idx2answer = dataset.idx2answer

    def __getitem__(self, idx):
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}")
        image = self.transform(image)

        question_text = self.df["question"][idx]
        inputs = self.tokenizer(question_text, return_tensors='pt', padding='max_length', max_length=20, truncation=True)
        question = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)

        if self.answer:
            answers = [self.answer2idx[answer["answer"].lower()] for answer in self.df["answers"][idx]]
            mode_answer_idx = mode(answers)
            return image, question, attention_mask, torch.tensor(answers), torch.tensor(mode_answer_idx)

        else:
            return image, question, attention_mask

    def __len__(self):
        return len(self.df)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, layers[0], 64)
        self.layer2 = self._make_layer(block, layers[1], 128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], 256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 512)

    def _make_layer(self, block, blocks, out_channels, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet50():
    return ResNet(BottleneckBlock, [3, 4, 6, 3])


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def VQA_criterion(batch_pred: torch.Tensor, batch_answers: torch.Tensor):
    total_acc = 0.
    for pred, answers in zip(batch_pred, batch_answers):
        acc = 0.
        for i in range(len(answers)):
            num_match = 0
            for j in range(len(answers)):
                if i == j:
                    continue
                if pred == answers[j]:
                    num_match += 1
            acc += min(num_match / 3, 1)
        total_acc += acc / 10
    return total_acc / len(batch_pred)

class VQAModel(nn.Module):
    def __init__(self, n_answer: int):
        super().__init__()
        self.resnet = ResNet18()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.fc = nn.Sequential(
            nn.Linear(768 + 512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_answer)
        )

    def forward(self, image, question, attention_mask):
        image_feature = self.resnet(image)
        question_feature = self.bert(question, attention_mask=attention_mask).pooler_output

        x = torch.cat([image_feature, question_feature], dim=1)
        x = self.fc(x)

        return x

def train(model, dataloader, optimizer, criterion, device):
    model.train()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for image, question, attention_mask, answers, mode_answer in dataloader:
        image, question, attention_mask, answers, mode_answer = \
            image.to(device), question.to(device), attention_mask.to(device), answers.to(device), mode_answer.to(device)

        pred = model(image, question, attention_mask)
        loss = criterion(pred, mode_answer)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start

def eval(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    with torch.no_grad():
        for image, question, attention_mask, answers, mode_answer in dataloader:
            image, question, attention_mask, answers, mode_answer = \
                image.to(device), question.to(device), attention_mask.to(device), answers.to(device), mode_answer.to(device)

            pred = model(image, question, attention_mask)
            loss = criterion(pred, mode_answer)

            total_loss += loss.item()
            total_acc += VQA_criterion(pred.argmax(1), answers)
            simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start

def main():
    set_seed(32)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor()
    ])

    train_dataset = VQADataset(df_path="./data/train.json", image_dir="./data/train", transform=transform)
    test_dataset = VQADataset(df_path="./data/valid.json", image_dir="./data/valid", transform=transform, answer=False)
    test_dataset.update_dict(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = VQAModel(n_answer=len(train_dataset.answer2idx)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    num_epoch = 1
    for epoch in range(num_epoch):
        train_loss, train_acc, train_simple_acc, train_time = train(model, train_loader, optimizer, criterion, device)
        print(f"【{epoch + 1}/{num_epoch}】\n"
              f"train time: {train_time:.2f} [s]\n"
              f"train loss: {train_loss:.4f}\n"
              f"train acc: {train_acc:.4f}\n"
              f"train simple acc: {train_simple_acc:.4f}")

    model.eval()
    submission = []
    for image, question, attention_mask in test_loader:
        image, question, attention_mask = image.to(device), question.to(device), attention_mask.to(device)
        pred = model(image, question, attention_mask)
        pred = pred.argmax(1).cpu().item()
        submission.append(pred)

    submission = [train_dataset.idx2answer[id] for id in submission]
    submission = np.array(submission)
    torch.save(model.state_dict(), "model.pth")
    np.save("submission.npy", submission)

if __name__ == "__main__":
    main()
