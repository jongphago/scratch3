# Library
import numpy as np

# 01
class Variable:
    def __init__(self, data):
        # 09
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{}은(는) 지원하지 않습니다.'.format(type(data)))
        self.data = data
        # 06
        self.grad = None
        # 07
        self.creator = None
    def set_creator(self, func):
        self.creator = func
    # 07
    def backward(self):
        '재귀를 이용한 구현'
        # f = self.creator # 1. 함수를 가져온다.
        # if f is not None:
            # x = f.input # 2. 함수의 입력을 가져온다.
            # x.grad = f.backward(self.grad) # 3. 함수의 backward 메서드를 호출
            # x.backward() # 하나 앞 변수의 backward 메서드를 호출(재귀)
        # 08
        '반복문을 이용한 구현'
        # 09
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        func = [self.creator]
        while func:
            f = func.pop()  # 함수를 가져온다.
            x, y = f.input, f.output  # 함수의 입력과 출력을 가져온다
            x.grad = f.backward(y.grad) # backward 메서드를 호출
            if x.creator is not None:
                func.append(x.creator)  # 하나 앞의 함수를 리스트에 추가

# 02
class Function:
    def __call__(self, inputs):
        # x = input.data
        # y = self.forward(x) # 구체적인 계산은 forward 메서드에서 한다
        # 11
        xs = [x.data for x in inputs]
        ys = self.forward(xs)
        outputs = [Variable(as_array(y)) for y in ys]
        # 06
        self.inputs = inputs # 입력 변수를 기억(보관)한다.
        # output = Variable(as_array(y))
        # 07
        # output.set_creator(self) # 출력 변수에 창조자를 설정한다.
        for output in outputs:
            output.set_creator(self)
        self.outputs = outputs # 출력도 저장한다.
        return outputs
    def forward(self, x):
        raise NotImplementedError()
    # 06
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

# 09
def square(x):
    f = Square()
    return f(x)

def exp(x):
    f = Exp()
    return f(x)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

# 10
import unittest

class SquareTest(unittest.TestCase):
    def test_toward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(3.))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)

    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)

# 11
class Add(Function):
    def forward(self, xs):
        x0, x1 = xs
        y = x0 + x1
        return (y,)
        
