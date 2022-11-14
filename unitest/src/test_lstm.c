#include "include/swlstm.h"
#include "include/swcommon.h"
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define Dtype float
Dtype sigmoid(Dtype x) {
  return 1. / (1. + exp(-x));
}
int test_lstm()
{
  int a[10],b[10],c[10];
  a[0]=64;a[1]=128;a[2]=256;a[3]=512;
  b[0]=128;b[1]=256;b[2]=512;b[3]=1024;b[4]=1600;
  int num,channels,w,h;
  int spatial_dim;
  int N,H,n,d;
  int i,j,k,z;
  int ii,jj,kk;
  int blob_size;
  float eps_=1e-4;
  struct timeval t1,t2;
  int outer_num_,inner_num_,dim;
  float sum,cont;

  float * clip_t=(float*)malloc(sizeof(float)*512);
  float * pre_gate_t=(float*)malloc(sizeof(float)*512*4*1600);
  float * h_to_gate=(float*)malloc(sizeof(float)*512*4*1