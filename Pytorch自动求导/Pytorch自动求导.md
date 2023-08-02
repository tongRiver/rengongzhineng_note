## PyTorch中的自动求导

PyTorch中的自动求导（Autograd）是其核心功能之一，它使得神经网络的训练和优化过程更加便捷和高效。自动求导功能可以自动地计算张量（Tensor）的梯度，即导数，相对于某个标量值的梯度。这对于优化算法，特别是梯度下降算法，非常重要，因为它可以帮助我们找到损失函数的最小值或最大值。

在深度学习中，模型通常由一系列参数组成（如权重和偏置项），而<font color=orange>**优化的目标是通过调整这些参数来最小化（或最大化）某个损失函数的值**</font>。**梯度表示了损失函数对参数的变化率**，告诉我们在当前参数值下，应该朝着哪个方向调整参数才能使损失函数减小。

PyTorch的自动求导功能通过构建动态计算图来实现。计算图是一个由张量操作构成的有向无环图（DAG），其中张量是图中的节点，操作是图中的边。当我们进行张量操作时，PyTorch会自动跟踪这些操作，并在后台构建计算图。

在训练过程中，**当我们计算损失函数的值后**，我们可以调用backward()方法在损失张量上执行反向传播。在反向传播过程中，**PyTorch会根据链式法则自动计算每个参数（张量）的梯度，并将这些梯度保存在相应的参数张量中**。这样，我们就可以在优化器中使用这些梯度来更新模型的参数，从而不断迭代优化模型，使其更接近最优解。

总结一下，PyTorch中的自动求导功能使得我们在构建和训练神经网络时无需手动计算梯度，而是能够自动地在后台进行梯度计算和更新参数。这大大简化了深度学习模型的开发过程，提高了代码的可读性和可维护性。



自动求导的例子：

```python
import torch

# 创建两个需要求导的张量
x = torch.tensor(2.0, requires_grad=True)  # 只有一个元素的张量
y = torch.tensor(3.0, requires_grad=True)  # 只有一个元素的张量

# 定义一个计算图：z = 3x^2 + 2y
z = 3 * x**2 + 2 * y

"""进行前向传播计算，自动构建计算图
   在这个过程中，PyTorch会跟踪所有涉及requires_grad=True的张量操作
   并构建一个计算图，将操作和张量连接起来"""

# 计算z对x的梯度
z.backward()

# 计算结果
print("x:", x)  # 输出 x: tensor(2., requires_grad=True)
print("y:", y)  # 输出 y: tensor(3., requires_grad=True)
print("z:", z)  # 输出 z: tensor(18., grad_fn=<AddBackward0>)

# 计算梯度
print("dz/dx:", x.grad)  # 输出 dz/dx: tensor(12.)
print("dz/dy:", y.grad)  # 输出 dz/dy: tensor(2.)
```



## 关闭自动求导

为啥要关闭自动求导？

在PyTorch中，使用`with torch.no_grad()`上下文管理器可以临时关闭自动求导功能。这个功能通常用于以下两种情况：

1. **节省内存和计算资源：** 在模型的评估阶段，我们通常不需要计算梯度，因为梯度只用于模型的训练过程。关闭自动求导功能可以避免计算和存储不必要的梯度信息，从而节省内存和计算资源。
2. **避免参数更新：** 在某些情况下，我们可能需要手动更新模型的参数，而不依赖于自动求导功能。在这种情况下，我们可以在`with torch.no_grad()`块中执行参数更新操作，避免被自动求导记录。例如，当我们加载预训练模型时，通常希望保持模型参数不变，而不希望在加载过程中更新这些参数。

让我们通过一个例子来演示如何使用`with torch.no_grad()`来关闭自动求导：

```python
import torch

# 创建需要求导的张量
x = torch.tensor(2.0, requires_grad=True)

# 定义一个计算图：y = x^2
y = x**2

# 进行前向传播计算，自动构建计算图，并计算梯度
y.backward()

# 输出梯度
print(x.grad)     # tensor(4.)

# 使用torch.no_grad()上下文管理器关闭自动求导功能
with torch.no_grad():
    y = x**3
    print(y.grad)  # None
    print(x.grad)  # tensor(4.)

print(y.grad)      # None
print(x.grad)      # tensor(4.)
```

可以看到，在`with torch.no_grad()`块中，**`y`的梯度没了，但是`x`的梯度保持不变**。

再看这些例子：

```python
a=torch.tensor([1.1])
print(a.requires_grad)
#答案是False

a=torch.tensor([1.1],requires_grad=True)
b=a*2
print(b.requires_grad)
#答案是True

a=torch.tensor([1.1],requires_grad=True)
with torch.no_grad():
    b=a*2
print(a.requires_grad)
print(b.requires_grad)
#答案是 True False
```



## 梯度归零

在PyTorch中，`grad.zero_()`是一个用于梯度归零的方法。它是应用在张量的梯度属性上的，通常用于模型优化的迭代过程中。

当你进行反向传播（Backpropagation）并计算梯度时，PyTorch会自动在张量的`.grad`属性中保存计算得到的梯度值。然而，如果你多次调用反向传播，这些梯度值会累积，可能导致不正确的梯度更新。

为了避免梯度累积的问题，在每次参数更新之前，你通常会调用`grad.zero_()`方法来将张量的梯度归零。这样做的目的是确保在下一次反向传播时，只计算当前迭代步骤的梯度，而不受之前迭代步骤的影响。

以下是使用`grad.zero_()`的示例：

```python
import torch

# 创建一个需要梯度计算的张量
x = torch.tensor([2.0], requires_grad=True)

# 假设进行了一次反向传播，计算梯度
loss = x**2
loss.backward()

# 打印梯度
print(x.grad)  # 输出: tensor([4.])

# 再次进行一次反向传播，但不调用grad.zero_()方法
loss = 2 * x
loss.backward()

# 打印梯度，这次梯度会累积
print(x.grad)  # 输出: tensor([8.])

# 使用grad.zero_()方法将梯度归零
x.grad.zero_()

# 进行反向传播，但梯度已经被归零
loss = 3 * x
loss.backward()

# 打印梯度，这次只计算当前迭代步骤的梯度
print(x.grad)  # 输出: tensor([6.])
```

在训练深度学习模型时，通常会在每个训练迭代步骤之前调用`grad.zero_()`，以确保梯度不会累积并且能够正确地进行参数更新。



## 实例

```python
def sgd(params, lr, batch_size):  #@save
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
```

当进行反向传播计算梯度后，参数（张量）的`.grad`属性会保存计算得到的梯度值。在小批量随机梯度下降（SGD）等优化算法中，我们在每次参数更新之前都需要将梯度清零，以确保在下一次反向传播时只计算当前迭代步骤的梯度，而不受之前迭代步骤的影响。这就是`param.grad.zero_()`的作用。

让我们逐步解释这段代码：

1. `with torch.no_grad():`：这是一个上下文管理器，用于在接下来的代码块中禁止梯度的自动跟踪和计算。这样做是为了避免在梯度更新时浪费不必要的计算资源。
2. `for param in params:`：这里`params`是一个包含需要优化的参数张量的列表。通常，这些参数是模型的权重和偏置。
3. `param -= lr * param.grad / batch_size`：这是小批量随机梯度下降算法的一步更新。`lr`是学习率，`param.grad`包含了参数`param`的梯度信息。这行代码将使用梯度下降的更新规则来更新参数的值。
4. `param.grad.zero_()`：这一行的作用是将参数的梯度归零，即将`.grad`属性中的值清零。这是为了确保在下一次反向传播时，只计算当前迭代步骤的梯度，并不会受到之前迭代步骤的影响。

总结起来，`param.grad.zero_()`用于在每次参数更新之前将梯度归零，以确保在小批量随机梯度下降等优化算法中，每次梯度计算都是基于当前迭代步骤的损失值，而不会受到之前迭代步骤的梯度累积影响。



## 更多资料

[no_grad — PyTorch 2.0 documentation](https://pytorch.org/docs/stable/generated/torch.no_grad.html)

[python - What is the purpose of with torch.no_grad(): - Stack Overflow](https://stackoverflow.com/questions/72504734/what-is-the-purpose-of-with-torch-no-grad)

这一篇问的很好，就是答得一般。

> Q：
>
> 考虑使用PyTorch实现的线性回归的以下代码：
>
> X是输入，Y是训练集的输出，w是需要优化的参数
>
> ```python
> import torch
> 
> X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
> Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)
> 
> w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
> 
> def forward(x):
>     return w * x
> 
> def loss(y, y_pred):
>     return ((y_pred - y)**2).mean()
> 
> print(f'Prediction before training: f(5) = {forward(5).item():.3f}')
> 
> learning_rate = 0.01
> n_iters = 100
> 
> for epoch in range(n_iters):
>     # predict = forward pass
>     y_pred = forward(X)
> 
>     # loss
>     l = loss(Y, y_pred)
> 
>     # calculate gradients = backward pass
>     l.backward()
> 
>     # update weights
>     #w.data = w.data - learning_rate * w.grad
>     with torch.no_grad():
>         w -= learning_rate * w.grad
>     
>     # zero the gradients after updating
>     w.grad.zero_()
> 
>     if epoch % 10 == 0:
>         print(f'epoch {epoch+1}: w = {w.item():.3f}, loss = {l.item():.8f}')
> ```
>
> “with”块的作用是什么？w的requires_grad参数已设置为True。为什么它会被放在一个带 torc.no_grad（）的块下？
>
> A：
>
> The requires_grad argument tells PyTorch that we want to be able to calculate the gradients for those values. However, the with torch.no_grad() tells PyTorch to not calculate the gradients, and the program explicitly uses it here (as with most neural networks) in order to not update the gradients when it is updating the weights as that would affect the back propagation.
>
> requires_grad参数告诉PyTorch，我们希望能够计算这些值的渐变。然而，with torch.no_grad（）告诉PyTorch不要计算渐变，程序在这里显式使用它（与大多数神经网络一样），以便在更新权重时不更新渐变，因为这会影响反向传播。
>
> There is no reason to track gradients when updating the weights; that is why you will find a decorator (@torch.no_grad()) for the step method in any implementation of an optimizer.
>
> 在更新权重时没有理由跟踪梯度；这就是为什么在优化器的任何实现中都会为step方法找到一个装饰器（@torch.no.grad（））。
>
> "With torch.no_grad" block means doing these lines without keeping track of the gradients.
>
> “使用torc.no.grad”块意味着在不跟踪坡度的情况下执行这些行。







