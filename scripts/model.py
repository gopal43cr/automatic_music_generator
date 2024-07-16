from preprocess import *

class MIDIDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

train_dataset = MIDIDataset(x_train, y_train)
test_dataset = MIDIDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, timesteps, embed_dim))
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout) 
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(embed_dim, 20)
        self.fc2 = nn.Linear(20, vocab_size)

    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding
        for layer in self.transformer_layers:
            x = layer(x)
        x = torch.mean(x, dim=1)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

vocab_size = len(note2ind)
embed_dim = 256
num_heads = 8
ff_dim = 512
num_layers = 2

model = TransformerModel(vocab_size, embed_dim, num_heads, ff_dim, num_layers).to(device)

