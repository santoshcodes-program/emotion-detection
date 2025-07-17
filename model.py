import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, pad_idx):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        self.conv1 = nn.Conv1d(embed_dim, 200, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, 200, kernel_size=4, padding=2)
        self.conv3 = nn.Conv1d(embed_dim, 200, kernel_size=5, padding=2)

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(600, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x3 = F.relu(self.conv3(x))

        x1 = F.max_pool1d(x1, x1.size(2)).squeeze(2)
        x2 = F.max_pool1d(x2, x2.size(2)).squeeze(2)
        x3 = F.max_pool1d(x3, x3.size(2)).squeeze(2)

        out = torch.cat((x1, x2, x3), dim=1)
        out = self.dropout(out)
        return self.fc(out)
