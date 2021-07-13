# Library
import numpy as np

# 01
class Variable:
    def __init__(self, data):
        if type(data) != np.ndarray:
            data = np.array(data)
        self.data = data
        # 06
        self.grad = None

# 02
class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x) # 구체적인 계산은 forward 메서드에서 한다
        output = Variable(y)
        # 06
        self.input = input # 입력 변수를 기억(보관)한다.
        return output
    def forward(self, x):
        raise NotImplementedError()
    def backward(self, gy):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        y = np.power(x, 2.)
        return y
    # 06
    def backward(self, gy):
        x = self.input.data
        gx = (2. * x) * gy
        return gx

# 03
class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y
    # 06
    def backward(self, gy):
        x = self.input.data
        gx = (np.exp(x)) * gy
        return gx

# 04 
def numerical_diff(f, x:Variable, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

# 06
