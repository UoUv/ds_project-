#include "include/swtensortrans.h"
#include "include/swcommon.h"
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "swblas.h"

// high -> low
// B, N, W, H
inline int image_caffe_offset(int b, int n, int h, int w, int B, int N, int H, int W) {
  return (((b*N + n)*H + h)*W + w);
}

// W, H, N, B
inline int image_swdnn_offset(int b, int n, int h, int w, int B, int N, int H, int W) {
  return (((h*W + w)*N + n)*B + b);
}

void test_tensor_trans_float() {
  int i, j;
  double tt, total_data_size;
  struct timeval t1, t2;

  int B = 128, H = 32, W = 32, N = 64;
  int buff_size = B*H*W*N;
  float* input = _aligned_malloc(sizeof(float)*B*H*W*N, 128);
  float* output = _aligned_malloc(sizeof(float)*B*H*W*N, 128);

  for(i = 0; i < buff_size; ++i)
    input[i] = rand()/(float)RAND_MAX;
  memset(output, 0, sizeof(float)