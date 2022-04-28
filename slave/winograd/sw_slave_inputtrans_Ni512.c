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
  int Ni = 