import torch


class SimpleConvRelu(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 3, stride=1, bias=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


class MultiOutputModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 3, stride=1, bias=True)

    def forward(self, x):
        y = self.conv(x)
        return y, y.mean()


class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 32)

    def forward(self, x):
        return self.fc(x)


class AddModel(torch.nn.Module):
    def forward(self, x, y):
        return x + y


class ConvSigmoidConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)

    def forward(self, x):
        x = self.conv1(x).relu()
        x = torch.sigmoid(x)
        x = self.conv2(x).relu()
        return x
