import torch
### initializing tensor ###
### how to create a tensor
##people often write
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
my_tensor = torch.tensor([[1,2,3],[4,5,6]],dtype = torch.float,
                         device = 'cpu',requires_grad= True)
print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.shape)

# other common initialization methods
x = torch.empty(size = (3,3))
x = torch.zeros((3,3))
x= torch.ones((3,3))
x = torch.arange(start = 0,end = 5, step = 1)
x = torch.linspace(start = 0.1, end = 1, steps = 10)
x = torch.diag(torch.ones(3))
print(x)

#how to initiate and oncert tensors to other types
tensor = torch.arange(4)
print(tensor.bool())
print(tensor.short())##int 16
print(tensor.long())##int 64
print(tensor.half())#float 16

# array to tensor conversion and vice-versa
import numpy as np
np_array = np.zeros((5,5))
tensor = torch.from_numpy(np_array)
np_array = tensor.numpy()

# tensor math operation
x = torch.tensor([1,2,3])
y = torch.tensor([4,5,6])
a = torch.add(x,y)
print(a)
a = x + y
print(a)

#division
z = torch.true_divide(x,y)
print(z)
# inplace operation
t = torch.zeros(3)
t.add_(x)
print(x)
t += x
#expoetiation
z = x.pow(2)
z = x **2
##matrix multiplication
x1 = torch.rand((2,4))
x2 = torch.rand((4,3))
x3 = torch.mm(x1,x2)
print(x3)
# other operations
sum_x = torch.sum(x,dim = 0)
values,indices = torch.max(x,dim = 0)
abs_x = torch.abs(x)
z = torch.argmax(x, dim = 0)
z = torch.eq(x,y)
print(z)
z = torch.any(x)
z = torch.all(x)
## reshape the data
x= torch.arange(9)
x = x.view(3,3)
x = x.reshape(3,3)
print(x)
print(x.t())
#print(torch.cat((x,y)),dim = 0)
#flattene the things
z = x1.view(-1)
print(z)
x = torch.arange(10)
print(x.unsqueeze(1).shape)

