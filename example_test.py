import torch
import torch.nn as nn
input=torch.ones(1, 1, 5, 5)
m = nn.UpsamplingNearest2d(scale_factor=2)
print(m(input))

# (1) Log the scalar values
info = {
    'loss': 20,
    'accuracy': 10
}
x = {
           "epoch": 23,
           "iter": 31
       }
info = {**x, **info}
for tag, value in info.items():
    print(tag,value)