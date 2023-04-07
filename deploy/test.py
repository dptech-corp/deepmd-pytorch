import torch

class MyModule(torch.nn.Module):
    def __init__(self, M, N):
        super(MyModule, self).__init__()
        self.weight = torch.nn.Parameter(torch.rand((M, N), dtype=torch.float64))

    def forward(self, input):
        output = torch.matmul(self.weight, input)
        return output

my_module = MyModule(10, 10)
x = torch.rand((10, 3), dtype=torch.float64)
print(my_module(x))
sm = torch.jit.script(my_module)
sm.save("model.pt")