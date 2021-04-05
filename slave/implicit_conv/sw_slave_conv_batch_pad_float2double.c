/***************
 * GEMM PLAN 
 * Jerry Fang 
 * 2017 June 15 
 *
 * input  is of dim(B, Ni)
 * weight is of dim(Ni, No)
 * ouput  is of dim(B, No)
 *
 * No overlap input DMA and weight DMA
 * ************/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <slave.h>
#include <math.h>
#include "simd.h"
#include "dma.h"
#include "./include/sw_conv_implicit.h"

#define SIMDSIZE  4
#define SIMDType  floatv4
#define Type      float
#define SIMDTypeD doublev4
#define TypeD     double

void conv_pad_float__(ConvData* param)
{
  int cB, cNi, cRi, cCi, cKr, cKc, ccCore, crCore, cNo;
  int ii, jj, cRo, cCo;
  int CoStart;
  int id = athread_get_id(-1);
  int cid = id%8, rid = id/8;
  int input_calc_index=1, input_load_index=0;
  int weight_calc_index=1, weight_load_index=0;
  int i, j;
  int Ni, Ri, Ci, No, K, Ro, Co, B, pad;
  Ni = param->_Ni;
  Ri = param->_Ri;
  Ci = param->_Ci;
  No = param->_No;
  K  = param->_K;
  Ro = param->_Ro;
  Co = param->_Co;
  B  = param->_B;
  pad = param->_pad;
  int CStride=param->_Costride;

//B, Ni, Ci, Ri
//fjr1buf
  SIMDType* local_input  = (SIMDType*) (long)ldm_malloc(sizeof(TypeD)*Ni*B/8/8);
  int local_input_size = Ni*B/8/8/SIMDSIZE;
//No, Ni, K, K
//fjr1buf
  Type* local_weight = (Type*)(long) ldm_malloc(sizeof(TypeD)*Ni*No/8/8);
  int local_weight_size = Ni*No/64;
//B, No, Co, Ro
  SIMDType* local_output = (SIMDType*)(long) ldm_malloc(sizeof(TypeD)*No*B/8/8*CStride);
  int local_output_size = No*B/8/8*CStride;
  SIMDTypeD vdbl;
  SIMDType vflt;
  Type*  fptr = (Type *)local_input;
  TypeD* dptr = (TypeD*)local_input;
  Type*  wfptr = (Type *)local_weight;
  TypeD* wdptr = (TypeD*)local_weight;
  Type*  ofptr = (Type *)local_output;
  TypeD* odptr = (TypeD*)local_output;

//  Type local_weight[K*K*Ni/64*No];
//initilize DMA variables
  volatile int  input_replyget = 0, weight_replyget = 0,  replyput = 0;
  dma_desc dma_get_input, dma_get_weight, dma_get_output, dma_put_output;

  dma_set_op(&dma_get_input, DMA_GET);
  dma_set_mode(&dma_get_input, PE_MODE);
  dma_set_reply(&dma_get_input, &input_replyget);

  dma_set_op(&dma_get_weight, DMA_GET);
  dma_set_mode(&dma_get_weight, PE_MODE);
  dma_set_reply(&dma_get_weight, &weight_replyget);

  dma_set_op(&dma_put_output, DMA_PUT);
  dma_set_mode(&dma_put_output, PE_MODE);
  dma_set_reply(&dma_put_output, &replyput);

  //DMA for local_iutput(B/8, Ni/8)
  dma_set_size(&dma_get_input, B*Ni/8/8/SIMDSIZE*sizeof(SIMDType));
  dma_set_bsize(&dma_get_input, B/SIMDSIZE/8*sizeof(SIMDType));
  dma_set_stepsize(&dma_get_input, B/SIMDSIZE/8*7*sizeof(SIMDType));

  //DMA for local_weight(No/8, Ni/8)
  dma_set_size(&dma_get_weight, No*Ni/8/8*sizeof(Type));
  dma_set_bsize(