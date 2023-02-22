import torch

class MyModule(torch.nn.Module):
    def __init__(self, N):
        super(MyModule, self).__init__()
        self.weight = torch.nn.Parameter(torch.rand(N,))

    def forward(self, input):
        output = self.weight + input
        return output

my_module = MyModule(10)
sm = torch.jit.script(my_module)
sm.save("model.pt")