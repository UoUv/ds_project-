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
  int i, j, k;
  double batch_im2col_tt = 0.;
#define Type float
  struct timeval t1, t2;

  int dilation_h, dilation_w, output_w, output_h;
  dilation_h = 1;
  dilation_w = 1;
  output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  output_w = (width  + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  int im_size = channels*height*width;
  int col_size = output_w*output_h*channels*kernel_h*kernel_w;
  int zeropad_col_rowsize = (output_w * output_h + 127)/128*128;
  int zeropad_col_colsize = (kernel_h * kernel_w * channels + 7)/8*8;
  int pad_col_size = zeropad_col_rowsize * zeropad_col_colsize;
  int group_ = 1;

  printf("forward: channels %d, filters %d, height %d, width %d, kernel_h %d, kernel_w %d, \
      pad_h %d, pad_w %d, output_w %d, output_h %d, stride_h %d, stride_w %d, zeropad_col_rowsize %d (%d)\
      zeropad_col_colsize %d (%d), batch_size %d\n",
          channels, filters, height, width, \
          kernel_h,kernel_w,pad_h,pad_w,output_w,output_h,stride_h,stride_w, zeropad_col_rowsize, output_w*output_h,zeropad_col_colsize,
          kernel_h*kernel_w*channels,
          batch_size);

  if(batch_size*((width + 2*pad_w) + output_w)*sizeof(float) > 60*1024)   {
    printf("batch_size is too large\n");
    return;
  }

  //allocate memory
  float* data_im = (float*)_aligned_malloc(sizeof(float)*im_size*batch_size, 128);
  float* data_col = (float*)_aligned_malloc(sizeof(Type)*col_size*batch_size, 128);
  if(!data_col)
    printf("allocate data_col failed!\n");
  memset(data_col,0.0, sizeof(Type)*col_size*batch_size);
  for(i = 0; i < im_size*batch_size; ++i)
    data_im[i] = rand()/(float)RAND_MAX;


  Type* zero_pad_data_col = (Type*)_aligned_malloc(sizeof(Type)*pad_col_size*batch_size, 128);
  if(!zero_pad_data_col)
    printf("allocate zero_pad_data_col failed!\n");
  memset(zero_pad_data_col, 0.0, sizeof(Type)*pad_col_size*batch_size);

  //params for GEMM
  int N = filters;
  int M = 