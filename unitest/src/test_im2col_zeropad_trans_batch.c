/****
 * Jiarui Fang
 * fang_jiarui@163,com
 * ****/
#include "include/swim2col.h"
#include "include/swcommon.h"
#include "./include/swtensortrans.h"
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "swblas.h"

/******
 * a unitest for batch-pad-im2col
 * Optimizations:
 * 1. batch im2col: transpose input features (B, N, R, C) -> (N, R, C, B), then perform batch-im2col
 * 2. zeropadding, (N, R, C, B) -> (K*K*N + pad, Ro*Co*B +pad), adding pad make GEMM easy
 * 3. batch-GEMM , make sure filters are like (K*K*Ni+pad, No)
 * ***/
void test_im2col_zeropad_batch_trans_swblas_float(int channels, int filters, int height, int width, int kernel_h, int kernel_w, int pad_h, int pad_w, int stride_h, int stride_w, int batch_size) {
  printf("begin test_im2col_zeropad_batch_swblas_float\n");
  int i, 