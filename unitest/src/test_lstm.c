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
  float * h_to_gate=(float*)malloc(sizeof(float)*512*4*1600);
  float * c_t_1=(float*)malloc(sizeof(float)*512*1600);

  float * gate_t=(float*)malloc(sizeof(float)*512*4*1600);
  float * h_t=(float*)malloc(sizeof(float)*512*1600);
  float * c_t=(float*)malloc(sizeof(float)*512*1600);

  float * my_gate_t=(float*)malloc(sizeof(float)*512*4*1600);
  float * my_h_t=(float*)malloc(sizeof(float)*512*1600);
  float * my_c_t=(float*)malloc(sizeof(float)*512*1600);
  float * my_pre_gate_t=(float*)malloc(sizeof(float)*512*4*1600);

  float * backup_c_t_1 = c_t_1;
  float * backup_h_to_gate = h_to_gate;
  float * backup_my_h_t = my_h_t;
  float * backup_my_gate_t = my_gate_t;
  float * backup_my_c_t = my_c_t;



  char out[20]="0 0 time";

  printf("start\n");
  for(ii=0;ii<4;++ii){
    for(jj=0;jj<5;++jj){
        //if(k==1) break;
        printf("i=%d j=%d",ii,jj);
        N=a[ii];H=b[jj];

        blob_size=N*4*H;

        //with origin data
        //init
        for(z = 0; z < blob_size; ++z ) my_pre_gate_t[z]=pre_gate_t[z] = rand()/(float)RAND_MAX;
        for(z = 0; z < blob_size; ++z ) h_