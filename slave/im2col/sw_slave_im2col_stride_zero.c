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
#define Type fl