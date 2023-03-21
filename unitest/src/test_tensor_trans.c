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
  i