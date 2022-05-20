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
#define SIMDSIZE 4
void FJR_input_trans_Ni512(InputData* param)
{
  int id = athread_get_id(-1);
  int cid = id%8, rid = id/8;
  int Ni = param->Ni;
  int B = param->B;
  int Ri = param->Ri;
  int Ci = param->Ci;
  int NR = (Ri-2)/2;
  int NC = (Ci-2)/2;
  int T = NR*NC;

  int NumNi = 1;
  if(Ni > 500) {
    NumNi = 2;
    Ni = Ni/2;
  }

  float* local_input  = (float*) ldm_malloc(sizeof(float)*Ni*16);
  int local_input_size = Ni*16;

  float* local_output = (float*) ldm_malloc(sizeof(float)*Ni*16);
  int local_output_size = Ni*16;

  if(Ni > 512) {
    if(id == 0)
      printf("input trans LDM is overflow\n");
  }

  volatile int  input_replyget = 0, replyput = 0;
  dma_desc dma_get_input, dma_put_output;

  dma_set_op(&dma_get_input, DMA_GET);
  dma_set_mode(&dma_get_input, PE_MODE);
  dma_set_reply(&dma_get_input, &input_replyget);

  dma_set_op(&dma_put_output, DMA_PUT);
  dma_set_mode(&dma_put_output, PE_MODE);
  dma_set_reply(&dma_put_output, &replyput);

  dma_set_size(&dma_get_input, Ni*4*sizeof(float));
  dma_set_b