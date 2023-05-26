import math
import numpy as np
import torch
import torch.nn as nn

# set random seed
torch.manual_seed(0)
np.random.seed(0)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
frame_num = 300


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, vocab_size=500, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(vocab_size, d_model)
        position = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TFNet(nn.Module):
    def __init__(
        self,
        nhead=4,
        d_model=128,
        dim_feedforward=256,
        num_layers=4,
        dropout=0,
    ):
        super().__init__()

        self.embed_mlp = nn.Sequential(
            nn.Linear(85, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        assert d_model % nhead == 0, "nheads must divide evenly into d_model"

        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            dropout=dropout,
            vocab_size=frame_num + 10,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 16),
        )
        self.d_model = d_model

    def forward(self, x):
        x = self.embed_mlp(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x


class TFNetV2(nn.Module):
    def __init__(
        self,
        nhead=4,
        d_model=128,
        dim_feedforward=256,
        num_layers=4,
        dropout=0,
    ):
        super().__init__()

        self.embed_mlp = nn.Sequential(
            nn.Linear(85, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        assert d_model % nhead == 0, "nheads must divide evenly into d_model"

        self.cls_token = torch.zeros(1, 1, d_model).to(device)

        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            dropout=dropout,
            vocab_size=frame_num + 10,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 16),
        )
        self.d_model = d_model

    def forward(self, x):
        x = self.embed_mlp(x)
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x[:, 0, :]
        x = self.classifier(x)
        return x


class MLPNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(85, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(16 * frame_num, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
        )

    def forward(self, x):
        x = self.mlp1(x)
        x = x.reshape(x.shape[0], -1)
        x = self.mlp2(x)
        return x


class MLPNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(85, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
        )

    def forward(self, x):
        x = self.mlp1(x)
        x = x.mean(dim=1)
        x = self.mlp2(x)
        return x


if __name__ == "__main__":
    # load data
    data = np.load("./data/qpos_trajectory2023-02-04--15-07-37.npy", allow_pickle=True)
    batch_size = 8

    states = [seq["state"] for seq in data]
    clipped_states = np.stack([state[:frame_num] for state in states], axis=0)
    labels = np.asarray([seq["obj"][0, 0] for seq in data])
    # labels = np.random.randint(0, 16, labels.shape)

    # convert to torch
    clipped_states = torch.from_numpy(clipped_states).float().to(device)
    labels = torch.from_numpy(labels).long().to(device)

    # tensor_dataset
    dataset = torch.utils.data.TensorDataset(clipped_states, labels)

    # split data
    train_size = int(0.8 * len(clipped_states))
    test_size = len(clipped_states) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # model
    model = MLPNetV2().to(device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    # loss
    criterion = nn.CrossEntropyLoss()

    # train
    for epoch in range(300):
        model.train()

        epoch_loss = 0
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()

            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, Loss: {epoch_loss / len(train_loader):.4f}")

        # test
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for i, (x, y) in enumerate(test_loader):
                    y_pred = model(x)
                    _, predicted = torch.max(y_pred.data, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()
                print(f"Epoch: {epoch}, Accuracy: {correct / total:.4f}")
