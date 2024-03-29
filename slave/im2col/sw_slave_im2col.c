
#include "slave.h"
#include "simd.h"
#include "dma.h"
#include "include/swim2col.h"

//#define PRINT_DEBUGINFO

__thread_local dma_desc dma_get_im, dma_put_col;
// Precondition: no stride and dilations. float data type
void sw_im2col_large_f(Im2colPara *para) {
#define Type float
#define SIMDType floatv4
#define SIMDSIZE 4
  int pad_h = para->pad_h;
  int pad_w = para->pad_w;
  int height= para->height;
  int width = para->width;
  int kernel_h = para->kernel_h;
  int kernel_w = para->kernel_w;
  int output_h = height + 2 * pad_h - kernel_h + 1;
  int output_w = width + 2 * pad_w - kernel_w + 1;
  int channel_size = height*width;
  int out_channel_size = output_h*output_w*kernel_w*kernel_h;
  int id = athread_get_id(-1);
  // number of rows of <id> slave core.
  int local_row_size = (2*para->pad_h + para->height) * para->channels / 64
               + (id< ((2*para->pad_h + para->height) * para->channels % 64));
  // start row index of <id> slave core.
  int row_start= -para->pad_h+
                  id*((2*para->pad_h+para->height)*para->channels/64)
               + (id<((2*para->pad_h+para->height)*para->channels%64)?
                  id:((2*para->pad_h+para->height)*para->channels%64));
  int row_end = row_start+local_row_size; // row_start<= ir < row_end)
  // buffer size
  int local_buff_size= para->width + 2*para->pad_w;
  int dma_buff_size = para->width;
  SIMDType* local_vbuffer = (SIMDType*)ldm_malloc(sizeof(Type)*local_buff_size);
  Type* local_buffer = (Type*)local_vbuffer;
  // begin ptr of dma_get
  Type* local_buffer_begin;
  Type* input_ptr = (Type*)para->data_im;
  Type* output_ptr= (Type*)para->data_col;

  int input_row, ir, ic, channel, k;
  int output_row, output_col, outoff, inoff;
  volatile int input_replyget=0, replyput=0;
  // dma settings
  dma_set_op(&dma_get_im, DMA_GET);
  dma_set_mode(&dma_get_im, PE_MODE);
  dma_set_reply(&dma_get_im, &input_replyget);

  dma_set_op(&dma_put_col, DMA_PUT);
  dma_set_mode(&dma_put_col, PE_MODE);
  dma_set_reply(&dma_put_col, &replyput);

  dma_set_size(&dma_get_im,width*sizeof(Type));
  dma_set_size(&dma_put_col,output_w*sizeof(Type));
//#define DEBUG
#ifdef DEBUG
  if(id!=0) return ;
#endif

  // begin im2col
  for(ir=row_start;ir<row_end;++ir) {
    input_row = (ir+pad_h)%(height+2*pad_h)-pad_h;
    channel = (ir+pad_h)/(height+2*pad_h);
    inoff = channel*width*height;
    // the row is pad
    if(!((unsigned)input_row<(unsigned)height)) {
      for(ic=0;ic<local_buff_size/SIMDSIZE;++ic){
        local_vbuffer[ic] = 0.0;
      }
      ic = ic*SIMDSIZE;
      // rest of the unaligned
      while(ic<local_buff_size) {
        local_buffer[ic] = 0.0;
        ++ic;
      }
#ifdef DEBUG
      for(;;);
#endif

    } else {
      // padding
      for(ic=0;ic<pad_w/SIMDSIZE;++ic) {
        local_vbuffer[ic] = 0.0;
#ifdef DEBUG
        for(;;);
#endif
      }
      ic = ic*SIMDSIZE;
      while(ic<pad_w) {
        local_buffer[ic] = 0.0;
        ++ic;
#ifdef DEBUG
        for(;;);
#endif
      }
      for(ic=(width+2*pad_w)/SIMDSIZE-SIMDSIZE;ic>=(pad_w+width)/SIMDSIZE;--ic) {
        local_vbuffer[ic] = 0.0;
#ifdef DEBUG
        for(;;);
#endif
      }
#ifdef PRINT_DEBUGINFO
      if(id==0) printf("before dma GET %d\n",input_row);
#endif
      // get data by dma
      dma(dma_get_im,(long)(input_ptr+input_row*width+inoff),(long)(local_buffer+pad_w));
      dma_wait(&input_replyget, 1); input_replyget = 0;
#ifdef PRINT_DEBUGINFO
      if(id==0) printf("dma get end.\n");
#endif
    }

    // put data by dma
    outoff = out_channel_size*channel;
    for(ic=0;ic<kernel_w;++ic) {
      local_buffer_begin = local_buffer+ic;
      for(k=0;k<kernel_h;++k) {
        // put output_w size from local_buffer(ic) to 
        // output_ptr(channel,ic+k*kernel_w,(input_row-k+pad_h)*output_w)
        // output the (ic+k*kernel_w)-th data in each kernel
        output_row = ic+k*kernel_w;
        output_col = (input_row-k+pad_h)*output_w;
        if(output_col<0) break; // out of range
        if(output_col>=output_w*output_h) continue; // out of range
#ifdef PRINT_DEBUGINFO
        if(id==0) printf("before dma PUT %d %d\n",output_row,output_col);
#endif
        dma( dma_put_col,
            (long)(output_ptr+output_row*(output_w*output_h)+output_col+outoff),
            (long)(local_buffer_begin));
        dma_wait(&replyput, 1); replyput = 0;
#ifdef PRINT_DEBUGINFO
        if(id==0) printf("dma put end.\n");
#endif
      }
    }

  }

  ldm_free(local_buffer,sizeof(Type)*local_buff_size);
#undef Type
#undef SIMDType
#undef SIMDSIZE
}

// Precondition: no stride and dilations. double data type
void sw_im2col_large_d(Im2colPara *para) {
#define Type double
#define SIMDType doublev4
#define SIMDSIZE 4
  int pad_h = para->pad_h;
  int pad_w = para->pad_w;
  int height= para->height;
  int width = para->width;
  int kernel_h = para->kernel_h;
  int kernel_w = para->kernel_w;
  int output_h = height + 2 * pad_h - kernel_h + 1;
  int output_w = width + 2 * pad_w - kernel_w + 1;
  int channel_size = height*width;
  int out_channel_size = output_h*output_w*kernel_w*kernel_h;
  int id = athread_get_id(-1);
  // number of rows of <id> slave core.
  int local_row_size = (2*para->pad_h + para->height) * para->channels / 64
               + (id< ((2*para->pad_h + para->height) * para->channels % 64));
  // start row index of <id> slave core.
  int row_start= -para->pad_h+
                  id*((2*para->pad_h+para->height)*para->channels/64)
               + (id<((2*para->pad_h+para->height)*para->channels%64)?
                  id:((2*para->pad_h+para->height)*para->channels%64));
  int row_end = row_start+local_row_size; // row_start<= ir < row_end)
  // buffer size
  int local_buff_size= para->width + 2*para->pad_w;
  int dma_buff_size = para->width;
  SIMDType* local_vbuffer = (SIMDType*)ldm_malloc(sizeof(Type)*local_buff_size);
  Type* local_buffer = (Type*)local_vbuffer;
  // begin ptr of dma_get
  Type* local_buffer_begin;
  Type* input_ptr = (Type*)para->data_im;
  Type* output_ptr= (Type*)para->data_col;

  int input_row, ir, ic, channel, k;
  int output_row, output_col, outoff, inoff;
  volatile int input_replyget=0, replyput=0;
  // dma settings
  dma_set_op(&dma_get_im, DMA_GET);
  dma_set_mode(&dma_get_im, PE_MODE);
  dma_set_reply(&dma_get_im, &input_replyget);

  dma_set_op(&dma_put_col, DMA_PUT);
  dma_set_mode(&dma_put_col, PE_MODE);
  dma_set_reply(&dma_put_col, &replyput);

  dma_set_size(&dma_get_im,width*sizeof(Type));
  dma_set_size(&dma_put_col,output_w*sizeof(Type));
//#define DEBUG
#ifdef DEBUG
  if(id!=0) return ;
#endif

  // begin im2col
  for(ir=row_start;ir<row_end;++ir) {
    input_row = (ir+pad_h)%(height+2*pad_h)-pad_h;
    channel = (ir+pad_h)/(height+2*pad_h);
    inoff = channel*width*height;
    // the row is pad
    if(!((unsigned)input_row<(unsigned)height)) {
      for(ic=0;ic<local_buff_size/SIMDSIZE;++ic){
        local_vbuffer[ic] = 0.0;
      }
      ic = ic*SIMDSIZE;
      // rest of the unaligned
      while(ic<local_buff_size) {
        local_buffer[ic] = 0.0;
        ++ic;
      }
#ifdef DEBUG
      for(;;);
#endif

    } else {
      // padding
      for(ic=0;ic<pad_w/SIMDSIZE;++ic) {
        local_vbuffer[ic] = 0.0;
#ifdef DEBUG
        for(;;);
#endif
      }
      ic = ic*SIMDSIZE;
      while(ic<pad_w) {
        local_buffer[ic] = 0.0;
        ++ic;
#ifdef DEBUG
        for(;;);
#endif
      }
      for(ic=(width+2*pad_w)/SIMDSIZE-SIMDSIZE;ic>=(pad_w+width)/SIMDSIZE;--ic) {
        local_vbuffer[ic] = 0.0;
#ifdef DEBUG
        for(;;);
#endif
      }
      // get data by dma
      dma(dma_get_im,(long)(input_ptr+input_row*width+inoff),(long)(local_buffer+pad_w));
      dma_wait(&input_replyget, 1); input_replyget = 0;
    }

    // put data by dma
    outoff = out_channel_size*channel;
    for(ic=0;ic<kernel_w;++ic) {
      local_buffer_begin = local_buffer+ic;
      for(k=0;k<kernel_h;++k) {
        // put output_w size from local_buffer(ic) to 
        // output_ptr(channel,ic+k*kernel_w,(input_row-k+pad_h)*output_w)
        // output the (ic+k*kernel_w)-th data in each kernel
        output_row = ic+k*kernel_w;
        output_col = (input_row-k+pad_h)*output_w;
        if(output_col<0) break; // out of range
        if(output_col>=output_w*output_h) continue; // out of range
        dma( dma_put_col,
            (long)(output_ptr+output_row*(output_w*output_h)+output_col+outoff),
            (long)(local_buffer_begin));
        dma_wait(&replyput, 1); replyput = 0;
      }
    }

  }

  ldm_free(local_buffer,sizeof(Type)*local_buff_size);
#undef Type
#undef SIMDType
#undef SIMDSIZE
}

