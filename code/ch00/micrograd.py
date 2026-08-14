"""micrograd.py — 极简自动微分（教学版，约 60 行）

第 0 章热身代码：用 60 行实现「反向传播」的核心。
看完这段代码，你就懂了 PyTorch 的 .backward() 在背后做什么。

核心思想：把每个数值包成 Value，记录它的算式来源（op 和子节点），
求梯度时从最终结果出发，沿算式反推，把「锅」按链式法则分给每个输入。

用法：
    python code/ch00/micrograd.py
"""

from __future__ import annotations


class Value:
    """一个既存数值、又记得自己怎么算出来的节点。"""

    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)      # 数值本身
        self.grad = 0.0              # 梯度，初始为 0
        self._backward = lambda: None  # 反向传播时执行的函数（默认无操作）
        self._prev = set(_children)    # 这个值由哪些子节点算出来
        self._op = _op                 # 记录运算符，方便调试打印

    # ---- 前向运算：构建算式图 ----
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # y = a + b  =>  dy/da = 1, dy/db = 1（加法把梯度原样传回去）
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # y = a * b  =>  dy/da = b, dy/db = a（乘法把对方的数值当系数传回去）
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float))
        out = Value(self.data ** exponent, (self,), f"**{exponent}")

        def _backward():
            # y = x^n  =>  dy/dx = n * x^(n-1)
            self.grad += exponent * (self.data ** (exponent - 1)) * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self + other

    # ---- 反向传播：从输出往输入分配「锅」 ----
    def backward(self):
        # 拓扑排序：先算离输出近的节点，保证梯度沿正确顺序传播
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0  # 输出对自己的梯度是 1
        for v in reversed(topo):  # 从输出往前逐个执行反向函数
            v._backward()

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"


def demo():
    """演示：训练一个 y = 3x + 2 的小模型（就一个参数 w）。"""
    print("=" * 50)
    print("演示：用梯度下降让 w 学会 y ≈ 3x + 2")
    print("=" * 50)

    # 训练数据（x, 真实y）
    data = [(1.0, 5.0), (2.0, 8.0), (3.0, 11.0), (4.0, 14.0)]
    w = Value(0.0)   # 从 w=0 开始
    lr = 0.01

    for epoch in range(1, 201):
        # 前向：算预测，算损失（均方误差）
        losses = []
        for x, y_true in data:
            y_pred = w * x + 2          # 我们的模型
            loss = (y_pred - y_true) ** 2
            losses.append(loss)
        total = sum(losses)             # Value 支持加法

        # 反向：自动求出 dLoss/dw
        total.backward()

        # 更新：沿负梯度方向走一步
        w.data -= lr * w.grad
        w.grad = 0.0                    # 清梯度（PyTorch 里也是这个套路）

        if epoch % 50 == 0:
            print(f"epoch {epoch:3d}: w = {w.data:+.4f}  loss = {total.data:8.4f}")

    print(f"\n训练完成：w = {w.data:.4f}（真实答案：3）")
    print("你刚才目睹了『反向传播 + 梯度下降』的完整一轮——")
    print("第 3 章预训练一个大语言模型，用的就是这同一个机制，只是规模大了几百万倍。")


if __name__ == "__main__":
    demo()
