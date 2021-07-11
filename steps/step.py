# Library
import numpy as np

# 01
class Variable:
    def __init__(self, data):
        if type(data) != np.ndarray:
            data = np.array(data)
        self.data = data

# 02
class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x) # 구체적인 계산은 forward 메서드에서 한다
        output = Variable(y)
        return output

    def forward(self, x):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        return x ** 2

# 03
class Exp(Function):
    def forward(self, x):
        return np.exp(x)

# 04 
def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)