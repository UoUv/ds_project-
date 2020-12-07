# swDNNv1.0
A Deep Learning Library for Sunway TaihuLight

## Features
### Convolution layer
three convolutional layer implementations.  
1. Explicit-GEMM  
2. Implicit-GEMM  
3. Winograd  

### Pooling

### Batch Normalization

### LSTM

### Tensor Transformation

## How to
### Use
mkdir ./build/  
cd build && cmake .. && make

### Add new layers
1. mkdir a directory using your new layer name in ./slave/  
`mkdir ./slave/conv`
2. write your slave code with name sw_slave_XXX, i.e. sw_slave_conv.c  
3. write your host code (interface to call slave code) in ./src/swXXX.c, i.e. swconv.c  
4. write your header file in ./include/swXXX.h