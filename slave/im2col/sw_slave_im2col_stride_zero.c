#include "slave.h"
#include "simd.h"
#include "dma.h"
#include "include/swim2col.h"

/*******
 * col (Ni*K*K, Co*Ro)
 * make Ni*K*K is 8x, fortunately we do not align Co*Ro*Batch
 * make Co*Ro is 128x
 * read batch_size input image rows to increase avaiable Bandwidth
 * put a batch of data
 ******/

void sw_im2col_large_stride_zeropad_f(Im2colPara *para) {
  int i, j;
  dma_desc dma_get_im, dma_put_col;
#define Type float
#define SIMDType floatv4
#define SIMDSIZE 4
  int pad_h = para->pad_h;
  int pad_w = para->pad_w;
  int height= para->height;
  int width = para->width;
  int kernel_h = para->kernel_h;
  int kernel_w = para->kernel_w;
  int stride_h = para->stride_h;
  int stride_w = para->stride_w;
  int output_h = (height + 2 * pad_h - kernel_h)/stride_h + 1; // output height with stride
  int output_w = (width + 2 * pad_w - kernel_w)/stride_w + 1;  // output width with stride
  int batch_size = 1;
  int channel_size = height*width;
  int channels = para->channels;
  //int out_channel_size = output_w*output_h*kernel_w*kernel_h;
  //fjrbatch
  int zeropad_col_rowsize = para->zeropad_col_rowsize;
  int zeropad_col_colsize = para->zeropad_col_colsize;
  int zeropad_col_size = zeropad_col_rowsize * zeropad_col_colsize;
  int im_size = channel_size*channels;

  int id = athread_get_id(-1);
  // number of rows of <i