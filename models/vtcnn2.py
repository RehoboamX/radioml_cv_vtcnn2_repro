import torch
from torch import nn


class VTCNN2(nn.Module):
    """PyTorch translation of the official RadioML VTCNN2 notebook."""

    def __init__(self, num_classes=11, dropout=0.5):
        super().__init__()
        self.pad1 = nn.ZeroPad2d((2, 2, 0, 0))
        self.conv1 = nn.Conv2d(1, 256, kernel_size=(1, 3))
        self.drop1 = nn.Dropout(dropout)
        self.pad2 = nn.ZeroPad2d((2, 2, 0, 0))
        self.conv2 = nn.Conv2d(256, 80, kernel_size=(2, 3))
        self.drop2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(80 * 1 * 132, 256)
        self.drop3 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="linear")
        for layer in (self.conv1, self.conv2, self.fc1, self.fc2):
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.drop1(torch.relu(self.conv1(self.pad1(x))))
        x = self.drop2(torch.relu(self.conv2(self.pad2(x))))
        x = torch.flatten(x, start_dim=1)
        x = self.drop3(torch.relu(self.fc1(x)))
        return self.fc2(x)
