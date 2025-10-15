import torch
import torch.nn as nn


class Neuron(nn.Module):
    def __init__(self, input_size=1):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        return torch.heaviside(self.fc(x), torch.tensor([0.0]))


neuron1 = Neuron(input_size=2)
neuron2 = Neuron(input_size=2)
neuron3 = Neuron(input_size=2)

neuron1.fc.weight.data = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
neuron1.fc.bias.data = torch.tensor([-0.5], dtype=torch.float32)

neuron2.fc.weight.data = torch.tensor([[-1.0, -1.0]], dtype=torch.float32)
neuron2.fc.bias.data = torch.tensor([1.5], dtype=torch.float32)

neuron3.fc.weight.data = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
neuron3.fc.bias.data = torch.tensor([-1.5], dtype=torch.float32)


def xor_with_3_neurons(x1, x2):
    input_tensor = torch.tensor([[float(x1), float(x2)]], dtype=torch.float32)

    hidden_output = torch.cat([neuron1(input_tensor), neuron2(input_tensor)], dim=1)

    xor_output = neuron3(hidden_output)

    return xor_output.item()


test_cases = [(0, 0), (0, 1), (1, 0), (1, 1)]

print("XOR с 3 нейронами:")
for x1, x2 in test_cases:
    result = xor_with_3_neurons(x1, x2)
    print(f"XOR({x1}, {x2}) = {result}")
