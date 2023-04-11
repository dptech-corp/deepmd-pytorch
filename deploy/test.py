import torch

class MyModule(torch.nn.Module):
    def __init__(self, M, N):
        super(MyModule, self).__init__()
        self.weight = torch.nn.Parameter(torch.rand((M, N), dtype=torch.float64))

    def forward(self, input):
        lst: List[torch.Tensor] = []
        output = torch.matmul(self.weight, input)
        output = output.sum()
        force = torch.autograd.grad([output], [input])[0]
        assert force is not None
        lst.append(output)
        lst.append(force)
        return lst

my_module = MyModule(10, 10)
x = torch.rand((10, 3), dtype=torch.float64)
x.requires_grad_(True)
sm = torch.jit.script(my_module)
output, force = sm(x)
print(output, force)
sm.save("model.pt")