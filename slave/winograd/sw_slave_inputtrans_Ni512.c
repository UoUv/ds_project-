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

  float* local_output = (float*) ldm_malloc(sizeof(float)*