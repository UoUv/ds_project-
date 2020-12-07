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
2. write your slave code wit