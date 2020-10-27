# Cuda implementation of per-pixel kernel evaluation for PyTorch

## Introduction

This folder represents a plugin layers for PyTorch. The layer is implemented with 
CUDA kernels and are thus more performant than the PyTorch equivalent. The layer is a regular `torch.nn.Module` primitive, behaves very similar to PyTorch stock layers (such as `Conv2d`), and supports both forward evaluation and back-propagation.

### WeightedFilter
The weighted filter primitive is a two dimensional convolution filter that accepts a per-pixel weight matrix, rather than a set of trainable weights. This primitive is intended to be used for the final layer of a kernel predicting network. It has no trainable parameters, but supports back-propagation. Given an input tensor $`\mathbf{x}`$ 
and a weight tensor $`\mathbf{w}`$, the output activations are compute as:

```math
out[c,y,x] =  \sum_{i,i^+}^{N} \sum_{j,j^+}^{N} w[i^+\cdot N + j^+,y,x] x[c,y+i,x+j]
```

Note that the weights in $`\mathbf{w}`$ are applied equally to all feature channels of $`\mathbf{x}`$, producing an output with as many
feature channels as $`\mathbf{x}`$. It is assumed that the input and weight tensors have the same *height* and *width* dimensions, and
border conditions are handled by zero padding.

**Splatting** The weighted filter also supports splat kernels. Instead of gathering the output activation as a nested sum, the contribution 
of each activation in the input tensor is scattered according to the pseudo code below.
```
for i in range(0, N):
    for j in range(0, N):
        out[c, y + i - N/2, x + j - N/2] += w[i * N + j, y, x] * x[c, y, x]
```
However, one can easily realize that this can be rewritten as a nested sum (gather) by modifying how the weight tensor is indexed.
```math
out[c,y,x] =  \sum_{i,i^+}^{N} \sum_{j,j^+}^{N} w[(N-i^+-1)\cdot N + (N-j^+-1),y+i,x+j] x[c,y+i,x+j]
```

## Usage example

```python
import torch
from torch_utils import WeightedFilter

in_channels  = 3
kernel_size  = 5

# Create kernel predicting network model
network = UNet(out_channels=kernel_size*kernel_size)

# Create a weighted (kpn) network layer without bias and using gather (no splatting)
kpn_layer = WeightedFilter(in_channels, kernel_size, bias=False, splat=False)

# Load image and guide
color  = loadImage('color.png')
guide  = loadImage('normal.png')
target = loadImage('target.png')

# Run forward pass
kpn_weights = network(color, guide)
out = kpn_layer(color, kpn_weights)

# Compute loss
loss = torch.nn.MSELoss()(out, target)

# Back propagate
loss.backward()
```

## Windows/Anaconda installation

### Requirements
  - PyTorch in Anaconda environment (tested with Python 3.7 and PyTorch 1.6)
  - Visual Studio 2019.
  - Cuda 10.2

### Installing

Open a **"x64 Native Tools Command Prompt for VS 2019"** and start your PyTorch Anaconda environment
from that prompt (it need to be that prompt so the paths to the correct compiler is properly set).

Then type:
```
cd [layerdl installation path]\torch_utils
set DISTUTILS_USE_SDK=1
python setup.py install
```

- List installed packages: `pip list`
- Remove package: `pip uninstall torch-utils`

## Installation in a Docker container

Navigate to the folder where you've cloned `layerdenoise` and build the docker image
`docker build --tag ldenoiser:latest -f docker/Dockerfile .`   

Launch a container   
`docker run --gpus device=0 --shm-size 16G --rm -v /raid:/raid -it ldenoiser:latest bash` 

### Tutorial for building custom modules

https://pytorch.org/tutorials/advanced/cpp_extension.html

