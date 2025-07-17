# ------------------------------------------
# 0. Imports
# ------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pickle, re
from collections import Counter
from datasets import load_dataset
import numpy as np
from sklearn.metrics import classification_report

# ------------------------------------------
# 1. Load Dataset
# ------------------------------------------
dataset = load_dataset("dair-ai/emotion")

# ------------------------------------------
# 2. Clean Text
# ------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()

dataset = dataset.map(lambda x: {"clean_text": clean_text(x["text"])})

# ------------------------------------------
# 3. Vocabulary Building
# ------------------------------------------
max_vocab_size = 10000
counter = Counter()

for example in dataset['train']:
    counter.update(example['clean_text'].split())

most_common = counter.most_common(max_vocab_size - 2)
word2idx = {'<PAD>': 0, '<OOV>': 1}
for i, (word, _) in enumerate(most_common, start=2):
    word2idx[word] = i

# ------------------------------------------
# 4. Load GloVe 200d Embeddings
# ------------------------------------------
def load_glove_embeddings(filepath, word2idx, embedding_dim=200):
    embeddings = np.random.uniform(-0.25, 0.25, (len(word2idx), embedding_dim))
    found = 0
    with open(filepath, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            if word in word2idx:
                found += 1
                vector = np.array(values[1:], dtype='float32')
                embeddings[word2idx[word]] = vector
    print(f"Loaded {found} pretrained embeddings out of {len(word2idx)}")
    return torch.tensor(embeddings, dtype=torch.float)

embedding_matrix = load_glove_embeddings("glove.6B.200d.txt", word2idx, embedding_dim=200)

# ------------------------------------------
# 5. Tokenization + Padding
# ------------------------------------------
def text_to_sequence(text):
    return [word2idx.get(w, word2idx['<OOV>']) for w in text.split()]

def tokenize_and_pad(dataset, column='clean_text', max_len=None):
    def process(split):
        seqs = [torch.tensor(text_to_sequence(x[column]), dtype=torch.long) for x in dataset[split]]
        nonlocal max_len
        if max_len is None:
            max_len = max(len(s) for s in seqs)
        padded = pad_sequence(seqs, batch_first=True, padding_value=word2idx['<PAD>'])
        return padded[:, :max_len] if padded.size(1) > max_len else \
               F.pad(padded, (0, max_len - padded.size(1)), value=word2idx['<PAD>'])

    return (*[process(split) for split in ['train', 'validation', 'test']], max_len)

train_x, val_x, test_x, max_len = tokenize_and_pad(dataset)

# ------------------------------------------
# 6. Labels
# ------------------------------------------
def extract_labels(dataset):
    return tuple(torch.tensor([x['label'] for x in dataset[split]]) for split in ['train', 'validation', 'test'])

train_y, val_y, test_y = extract_labels(dataset)

# ------------------------------------------
# 7. Dataset and DataLoader
# ------------------------------------------
class EmotionDataset(Dataset):
    def __init__(self, x, y): self.x, self.y = x, y
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]

train_loader = DataLoader(EmotionDataset(train_x, train_y), batch_size=32, shuffle=True)
val_loader   = DataLoader(EmotionDataset(val_x, val_y), batch_size=32)
test_loader  = DataLoader(EmotionDataset(test_x, test_y), batch_size=32)

# ------------------------------------------
# 8. TextCNN Model
# ------------------------------------------
class TextCNN(nn.Module):
    def __init__(self, embed_matrix, num_classes, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embed_matrix, freeze=False, padding_idx=pad_idx)
        self.conv1 = nn.Conv1d(200, 200, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(200, 200, kernel_size=4, padding=2)
        self.conv3 = nn.Conv1d(200, 200, kernel_size=5, padding=2)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(600, num_classes)

    def forward(self, x):
        x = self.embedding(x)        # [B, L, 200]
        x = x.permute(0, 2, 1)       # [B, 200, L]
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x3 = F.relu(self.conv3(x))
        x1 = F.max_pool1d(x1, x1.size(2)).squeeze(2)
        x2 = F.max_pool1d(x2, x2.size(2)).squeeze(2)
        x3 = F.max_pool1d(x3, x3.size(2)).squeeze(2)
        out = torch.cat((x1, x2, x3), dim=1)
        return self.fc(self.dropout(out))

# ------------------------------------------
# 9. Training & Evaluation Functions
# ------------------------------------------
def calculate_accuracy(preds, labels):
    return (preds.argmax(1) == labels).float().mean()

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train(); total_loss, total_acc = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward(); optimizer.step()
        total_loss += loss.item()
        total_acc += calculate_accuracy(out, y).item()
    return total_loss / len(loader), total_acc / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval(); total_loss, total_acc = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            total_loss += criterion(out, y).item()
            total_acc += calculate_accuracy(out, y).item()
    return total_loss / len(loader), total_acc / len(loader)

# ------------------------------------------
# 10. Train Model
# ------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TextCNN(embed_matrix=embedding_matrix,
                num_classes=len(dataset['train'].features['label'].names),
                pad_idx=word2idx['<PAD>']).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(20):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    print(f"Epoch {epoch+1}: Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")

# ------------------------------------------
# 11. Final Evaluation
# ------------------------------------------
def predict_all(model, loader, device):
    model.eval(); preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds.extend(out.argmax(1).cpu().numpy())
            labels.extend(y.cpu().numpy())
    return preds, labels

preds, labels = predict_all(model, test_loader, device)
print(classification_report(labels, preds, target_names=dataset['train'].features['label'].names))

# Save model and vocab
torch.save(model.state_dict(), "model_glove200.pth")
with open("word2idx.pkl", "wb") as f:
    pickle.dump(word2idx, f)
