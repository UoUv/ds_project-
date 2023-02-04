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
  float moving_average_fraction_=0.9;
  float eps_=1e-4;
  struct timeval t1,t2;
  int outer_num_,inner_num_,dim;
  float sum;

  float * top_diff=(float*)malloc(sizeof(float)*128*512*56*56);
  float * top_data=(float*)malloc(sizeof(float)*128*512*56*56);
  float * bottom_diff=(float*)malloc(sizeof(float)*128*512*56*56);
  float * scale_data=(float*)malloc(sizeof(float)*56*56);

  float * my_bottom_diff=(float*)malloc(sizeof(float)*128*512*56*56);
  float * my_scale_data=(float*)malloc(sizeof(float)*56*56);

  char out[20]="0 0 0 time";

  printf("start\n");
  for(ii=0;ii<3;++ii){
    for(jj=0;jj<5;++jj){
      for(kk=0;kk<4;++kk)
      {
        //if(k==1) break;
        printf("i=%d j=%d k=%d\n",ii,jj,kk);
        num=a[ii];channe