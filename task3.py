import torch
import torch.nn as nn


class ANDNeuron(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        return torch.heaviside(self.fc(x), torch.tensor([0.0]))


neuron = ANDNeuron()

neuron.fc.weight.data = torch.tensor([[1.0, 1.0]])
neuron.fc.bias.data = torch.tensor([-1.5])

print("Веса:", neuron.fc.weight.data)
print("Смещение:", neuron.fc.bias.data)

# Тестируем все комбинации
test_inputs = [
    torch.tensor([0.0, 0.0]),
    torch.tensor([0.0, 1.0]),
    torch.tensor([1.0, 0.0]),
    torch.tensor([1.0, 1.0]),
]

print("\nТестирование операции И:")
for x in test_inputs:
    output = neuron(x)
    print(f"AND({x[0].item()}, {x[1].item()}) = {output.item()}")
