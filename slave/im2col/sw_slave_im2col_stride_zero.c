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
  // number of rows of <id> slave core.
  int local_row_size = (2*para->pad_h + para->height) * para->channels / 64
               + (id< ((2*para->pad_h + para->height) * para->channels % 64));
  // start row index of <id> slave core.
  int row_start= id*((2*para->pad_h+para->height)*para->channels/64)
               + (id<((2*para->pad_h+para->height)*para->channels%64)?
                  id:((2*para->pad_h+para->height)*para->channels%64));
  int row_end = row_start+local_row_size; // row_start<= ir < row_end)
  // buffer size
  int local_buff_size = (para->width + 2*para->pad_w)*batch_size;
  SIMDType* local_vbuffer = (SIMDType*)ldm_malloc(sizeof(Type)*local_buff_size);
  Type* local_buffer = (Type*)local_vbuffer;
  Type* local_outbuff = (Type*)ldm_malloc(sizeof(Type)*output_w*batch_size);
  // begin ptr of dma_get
  Type* input_ptr = (Type*)para->data_im;
  Type* output_ptr= (Type*)para->data_col;

  int input_row, ir, ic, channel, k, ik;
  int output_row, output_col, outoff, inoff;
  volatile int input_replyget=0, replyput=0;
  // dma settings
  dma_set_op(&dma_get_im, DMA_GET);
  dma_set_mode(&dma_get_im, PE_MODE);
  dma_set_reply(&dma_get_im, &input_replyget);

  dma_set_op(&dma_put_col, DMA_PUT);
  dma_set_mode(&dma_put_col, PE_MODE);
  dma_set_reply(&dma_put_col, &replyput);

  //fjr batch
  dma_set_size(&dma_get_im,width*batch_size*sizeof(Type));
  dma_set_size(&dma_put_col,output_w*batch_size*sizeof(Type));
  //dma_set_bsize(&dma_get_im, width*sizeof(Type));
  //dma_set_stepsize(&dma_get_im, (im_size - width)*sizeof(Type));
  //dma_set_bsize(&dma_put_col, output_w*sizeof(Type));
  //dma_set_stepsize(&dma_put_col, (zeropad_col_size - output_w)*sizeof(Type));

  //fjrbatch
//  for(ic=0;ic<pad_w;++ic) local_buffer[ic] = 0.0;
//  for(ic=pad_w+width;ic<local_buff_size;++ic) local_buffer[ic] = 0.0;
  for(ic = 0; ic < local_buff_size; ++ic) local_buffer[ic] = 0.;

  // begin im2col
  for(ir=row_start;ir<row_end;++ir) {
    input_row = (ir)%(height+2*pad_h)-pad_h;
    channel = (ir)/(height+2*pad_h);
    inoff = channel*width*height;
    // the row is pad
    if(!((unsigned)input_row<(unsigned)height)) {
      for(ic = 0; ic < local_buff_size; ++ic) local_buffer[ic] = 0.;
      //for(ic=0;ic<local_buff_size/SIMDSIZE;++ic){
      //  local_vbuffer[ic] = 0.0;
      //}
      //ic = ic*SIMDSIZE;
      //// rest of the unalign