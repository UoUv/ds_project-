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
  dma_set_bsize(&dma_get_input, Ni*sizeof(float));
  dma_set_stepsize(&dma_get_input, (Ni*(NumNi-1))*sizeof(float));

  dma_set_size(&dma_put_output, Ni*16*sizeof(float));
  dma_set_bsize(&dma_put_output, Ni*sizeof(float));
  dma_set_stepsize(&dma_put_output, (B*T*Ni*NumNi - Ni)*sizeof(float));


  //(B, NR, NC, Ni)
  int cBlk, blkNi, ii, cNi;
  for(cBlk = id; cBlk < B*T; cBlk += 64) {
    int cB = cBlk/T;
    int cRi = cBlk%T/NC*2;
    int cCi = cBlk%NR*2;
    int cT = cBlk%T;

  //for(int cB = id; cB < B; cB+=64) {
  //  int cT = 0;
  //  for(int cRi = 0; cRi < Ri-2; cRi += 2)
  //    for(int cCi = 0; cCi < Ci-2; cCi += 2) {
    //DMA get a (4,4,Ni) -> put(16,Ni)
    for(blkNi = 0; blkNi < NumNi; ++blkNi) {
      for(ii = 0; ii < 4; ++ii) {
        float* input_offset = (float*)param->input + cB*Ri*Ci*(Ni*NumNi) + ((cRi+ii)*Ci + cCi)*(Ni*NumNi) + blkNi*Ni;
        dma(dma_get_input, (long)(input_offset), (long)(local_input + 4*Ni*ii));
        dma_wait(&input_replyget, 1); input_replyget = 0;
      }
      for(cNi = 0; cNi < Ni; cNi+=4) {
        floatv4 tmp[16];
        floatv4 s[16];
        simd_load(tmp[0], local_input + 0*Ni + cNi);
        simd_load(tmp[1], local_input + 1*Ni + cNi);
        simd_load(tmp[2], local_input + 2*Ni + cNi);
        simd_load(tmp[3], local_input + 3*Ni + cNi);
        simd_load(tmp[4], local_input + 4*Ni + cNi);
        simd_load(tmp[5], local_input + 5*Ni + cNi);
        simd_load(tmp[6], local_input + 6*Ni + cNi);
        simd_load(tmp[7], local_input + 7*Ni + cNi);
        simd_load(tmp[8], local_input + 8*Ni + cNi);
        simd_load(tmp[9], local_input + 9*Ni + cNi);
        simd_load(tmp[10], local_input + 10*Ni + cNi);
        simd_load(tmp[11], local_input + 11*Ni + cNi);
        simd_load(tmp[12], local_input + 12*Ni + cNi);
        simd_load(tmp[13], local_input + 13*Ni + cNi);
        simd_load(tmp[14], local_input + 14*Ni + cNi);
        simd_load(tmp[15], local_inpu