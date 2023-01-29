#include "include/swsoftmax.h"
#include "include/swcommon.h"
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define Dtype float
int test_softmax()
{
  int a[3],b[5],c[4];
  a[0]=32;a[1]=64;a[2]=128;
  b[0]=3;b[1]=64;b[2]=128;b[3]=256;b[4]=512;
  c[0]=7;c[1]=14;c[2]=28;c[3]=56;
  int num,channels,w,h;
  int spatial_dim;
  int i,j,k,z;
  int ii,jj,kk;
  int blob_size;
  int use_global_stats_=0;
  float moving_a