
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
void FJR_input_trans(InputData* param)
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

  float* local_input  = (float*) ldm_malloc(sizeof(float)*Ni*16);
  int local_input_size = Ni*16;

  volatile int  input_replyget = 0, replyput = 0;
  dma_desc dma_get_input, dma_put_output;

  dma_set_op(&dma_get_input, DMA_GET);
  dma_set_mode(&dma_get_input, PE_MODE);
  dma_set_reply(&dma_get_input, &input_replyget);

  dma_set_op(&dma_put_output, DMA_PUT);
  dma_set_mode(&dma_put_output, PE_MODE);
  dma_set_reply(&dma_put_output, &replyput);

  dma_set_size(&dma_get_input, Ni*16*sizeof(float));
  dma_set_bsize(&dma_get_input, Ni*4*sizeof(float));
  dma_set_stepsize(&dma_get_input, (Ci-4)*Ni*sizeof(float));

  dma_set_size(&dma_put_output, Ni*16*sizeof(float));
  dma_set_bsize(&dma_put_output, Ni*sizeof(float));
  dma_set_stepsize(&dma_put_output, (B*T-1)*Ni*sizeof(float));

  //(B, NR, NC, Ni)
  int cBlk, cNi; 
  for(cBlk = id; cBlk < B*T; cBlk += 64) {
    int cB = cBlk/T;
    int cRi = cBlk%T/NC*2;
    int cCi = cBlk%NR*2;
    int cT = cBlk%T;

    float* input_offset = (float*)param->input + cB*Ri*Ci*Ni + (cRi*Ci + cCi)*Ni;
    dma(dma_get_input, (long)(input_offset), (long)(local_input));
    dma_wait(&input_replyget, 1); input_replyget = 0;

    //DMA get a (4,4,Ni) -> put(16,Ni)
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
      simd_load(tmp[15], local_input + 15*Ni + cNi);

      // The tranformation manually simplified
      s[15] =(tmp[5] - tmp[13]) - (tmp[7 ]- tmp[15]);
      simd_store(s[15], local_input + 15*Ni + cNi);
      //
      s[0 ] =(tmp[0] - tmp[8 ]) - (tmp[2 ]- tmp[10]);
      simd_store(s[0],  local_input + 0*Ni + cNi);
      s[1 ] =(tmp[1] - tmp[9 ]) + (tmp[2 ]- tmp[10]);
      simd_store(s[1],  local_input + 1*Ni + cNi);
      s[2 ] =(tmp[2] - tmp[10]) - (tmp[1 ]- tmp[9 ]);
      simd_store(s[2],  local_input + 2*Ni + cNi);
      s[3 ] =(tmp[1] - tmp[9 ]) - (tmp[3 ]- tmp[11]);//2
      simd_store(s[3],  local_input + 3*Ni + cNi);
      s[4 ] =(tmp[4] + tmp[8 ]) - (tmp[6 ]+ tmp[10]);
      simd_store(s[4],  local_input + 4*Ni + cNi);
      s[5 ] =(tmp[5] + tmp[9 ]) + (tmp[6 ]+ tmp[10]);
      simd_store(s[5],  local_input + 5*Ni + cNi);
      s[6 ] =(tmp[6] + tmp[10]) - (tmp[5 ]+ tmp[9 ]);
      simd_store(s[6],  local_input + 6*Ni + cNi);
      s[7 ] =(tmp[5] + tmp[9 ]) - (tmp[7 ]+ tmp[11]);
      simd_store(s[7],  local_input + 7*Ni + cNi);

      s[8 ] =(tmp[8] - tmp[4 ]) - (tmp[10]- tmp[6 ]);
      simd_store(s[8],  local_input + 8*Ni + cNi);
      s[9 ] =(tmp[9] - tmp[5 ]) + (tmp[10]- tmp[6 ]);
      simd_store(s[9],  local_input + 9*Ni + cNi);
      s[10] =(tmp[10]- tmp[6 ]) - (tmp[9 ]- tmp[5 ]);
      simd_store(s[10], local_input + 10*Ni + cNi);
      s[11] =(tmp[9] - tmp[5 ]) - (tmp[11]- tmp[7 ]);
      simd_store(s[11], local_input + 11*Ni + cNi);
      s[12] =(tmp[4] - tmp[12]) - (tmp[6 ]- tmp[14]);
      simd_store(s[12], local_input + 12*Ni + cNi);
      s[13] =(tmp[5] - tmp[13]) + (tmp[6 ]- tmp[14]);
      simd_store(s[13], local_input + 13*Ni + cNi);
      s[14] =(tmp[6] - tmp[14]) - (tmp[5 ]- tmp[13]);
      simd_store(s[14], local_input + 14*Ni + cNi);
    }
    dma(dma_put_output, (long)((float*)param->transInput + cB*Ni*T + cT*Ni), (long)(local_input));
    dma_wait(&replyput, 1); replyput = 0;
  }

  ldm_free(local_input, sizeof(float)*local_input_size);

}//main func
