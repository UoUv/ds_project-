
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <slave.h>
#include <math.h>
#include <dma.h>
#include "./include/swwinogradconv.h"

/***************
 * GEMM PLAN 
 * Jerry Fang 
 * 2018.Sep.19th
 *
 * winograd input transformation
 *
 * ************/
//(3, 3, Ni, No)
//divide among Ni*No
#define SIMDSIZE 4
void FJR_filter_trans(FilterData* param)
{
  int id = athread_get_id(-1);
  int cid = id%8, rid = id/8;
  int Ni = param->Ni;
  int No = param->No;
  //assert(Ni*No%64 == 0);
  int blkNum = 64;
  int blkSize = Ni*No/blkNum;
  while(blkSize > 650) {
    blkNum *= 2;
    blkSize = Ni*No/blkNum;
  }

  if(blkSize > 650) {
    if(0 == id)
      printf("FJR_filter_trans LDM usage overflow!\n");
  }
  //blkNum % 64 == 0

  float* local_input  = (float*) ldm_malloc(sizeof(float)*blkSize*9);
  int local_input_size = blkSize*9;

  float* local_output = (float*) ldm_malloc(sizeof(float)*blkSize*16);
  int local_output_size = blkSize*16;

  volatile int  input_replyget = 0, replyput = 0;
  dma_desc dma_get_input, dma_put_output;

  dma_set_op(&dma_get_input, DMA_GET);
  dma_set_mode(&dma_get_input, PE_MODE);
  dma_set_reply(&dma_get_input, &input_replyget);

  dma_set_op(&dma_put_output, DMA_PUT);