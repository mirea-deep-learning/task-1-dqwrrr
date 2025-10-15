import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


x = torch.randn(1, 3)

net_seq = nn.Sequential(
    nn.Linear(in_features=3, out_features=5),
    nn.Sigmoid(),
    nn.Linear(in_features=5, out_features=2),
)
net_seq.to(device)
print("net_seq:")
print(net_seq)
print(net_seq(x.to(device)))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=3, out_features=5)
        self.fc2 = nn.Linear(in_features=5, out_features=2)

    def forward(self, x):
        x = self.fc1(x)
        return F.sigmoid(self.fc2(x))


net_model = Model()
net_model.to(device)
print("\nnet_model:")
print(net_model)
print(net_model(x.to(device)))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f"\nПараметров в net_seq: {count_parameters(net_seq)}")
print(f"Параметров в net_model: {count_parameters(net_model)}")

print("\nДетальный подсчет:")
print("net_seq:")
for name, param in net_seq.named_parameters():
    print(f"  {name}: {param.numel()} параметров")

print("\nnet_model:")
for name, param in net_model.named_parameters():
    print(f"  {name}: {param.numel()} параметров")
