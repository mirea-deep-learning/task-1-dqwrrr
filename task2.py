import numpy as np
import torch


class Neuron(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(1, 1)

    def forward(self, x):
        return torch.heaviside(self.fc(x), torch.tensor([0.0]))


neuron = Neuron()

neuron.fc.weight.data = torch.tensor([[-1.0]])
neuron.fc.bias.data = torch.tensor([0.5])

print("Веса:", neuron.fc.weight.data)
print("Смещение:", neuron.fc.bias.data)

test_inputs = [torch.tensor([0.0]), torch.tensor([1.0])]

print("\nТестирование операции НЕ:")
for x in test_inputs:
    output = neuron(x)
    print(f"NOT({x.item()}) = {output.item()}")
