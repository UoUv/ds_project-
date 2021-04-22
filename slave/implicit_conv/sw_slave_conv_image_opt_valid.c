
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <slave.h>
#include <math.h>
#include <dma.h>
#include "./include/sw_conv_implicit.h"

/***************
 * GEMM PLAN 
 * Jerry Fang 2016.8.1
 * 写程序也要按照基本法
 * input bCi
 * ************/
//#define DEBUG
#define SIMDSIZE 4
#define Type double
//output image size should be Co*Ro == 4x*4x
//Ni should be 8x
//input should be (4, Ci, Ri, Ni, 8)
//window size is (4,8)

void conv_image_size_aware_opt(ConvData* param)
{
  int cNi, cRi, cCi, cKr, cKc, ccCore, crCore, cNo, cRo, cCo;
  int id = athread_get_id(-1);
  int cid = id%8, rid = id/8;
  int input_calc_index=1, input_load_index=0;
  int weight_calc_index=1, weight_load_index=0;
  int i, j, k;
  int Ni, Ri, Ci, No, K, Ro, Co, bCo;
  Ni = param->_Ni;
  Ri = param->_Ri;
  Ci = param->_Ci;
  No = param->_No;
  K  = param->_K;
  Ro = param->_Ro;
  Co = param->_Co;
  bCo=param->_bCo;
  int bCi=bCo+K-1;
  int RoStart;
  int CoStart;

//4, Ci, Ri, Ni, 8
  Type* local_input  = (Type*) ldm_malloc(sizeof(Type)*Ni/8*bCi*4*2);
  int local_input_size = Ni*bCi/8*4;
//No, Ni, K, K
  Type* local_weight = (Type*) ldm_malloc(sizeof(Type)*Ni/8*No/8*2);
  int local_weight_size = Ni*No/8/8;
//Co, Ro, No, 8
  Type* local_output = (Type*) ldm_malloc(sizeof(Type)*No/8*bCo*4);
  int local_output_size = No/8*bCo*4;

//initilize DMA variables
  volatile int  replyget_weight = 0, replyget_input=0, replyget_output=0, replyput = 0;
  dma_desc dma_get_input, dma_get_weight, dma_get_output, dma_put_output;

  dma_set_op(&dma_get_input, DMA_GET);
  dma_set_mode(&dma_get_input, PE_MODE);
  dma_set_reply(&dma_get_input, &replyget_input);

  dma_set_op(&dma_get_weight, DMA_GET);
  dma_set_mode(&dma_get_weight, PE_MODE);
  dma_set_reply(&dma_get_weight, &replyget_weight);

  dma_set_op(&dma_get_output, DMA_GET);
  dma_set_mode(&dma_get_output, PE_MODE);
  dma_set_reply(&dma_get_output, &replyget_output);

  dma_set_op(&dma_put_output, DMA_PUT);
  dma_set_mode(&dma_put_output, PE_MODE);
  dma_set_reply(&dma_put_output, &replyput);

  //DMA for local_iutput(4, bCi, 1, Ni/8, 1)
  dma_set_size(&dma_get_input, 4*bCi*Ni/8*sizeof(Type));
  dma_set_bsize(&dma_get_input, 4*bCi*sizeof(Type));
  dma_set_stepsize(&dma_get_input, 4*(Ci-bCi)*sizeof(Type));

  //DMA for local_weight(No/8, Ni/8)
  dma_set_size(&dma_get_weight, No*Ni/8/8*sizeof(Type));
  dma_set_bsize(&dma_get_weight, Ni/8*sizeof(Type));
  dma_set_stepsize(&dma_get_weight, Ni/8*7*sizeof(Type));

  //DMA for local_output(4, bCo, 1, No/8, 1)
  dma_set_size(&dma_put_output, 4*bCo*No/8*sizeof(Type));
  dma_set_bsize(&dma_put_output, 4*bCo*sizeof(Type));
  dma_set_stepsize(&dma_put_output, 4*(Co-bCo)*sizeof(Type));
  
 