
extern "C"{
#include "./include/sw_conv_implicit.h"
}
#include "./unitest/include/conv_layer_impl_v2.hpp"
//#include "./unitest/include/conv_layer_impl.hpp"
#include "athread.h"
#include <math.h>
#include <stdio.h>
#include "athread.h"
#include <sys/time.h>
//#define CHECKRES

/*
 * float pad 
 */

//pad in
void test_forward_pad_float() {
  int Ni, No, B, Co, Ro, Ci, Ri, K, pad;
  Ni = 512;
  No = 512;
  B  = 128;
  K  = 3;
  pad = 1;
  Ci = 8;
  Ri = 8;
  Co = Ci+2*pad-K+1;
  Ro = Ri+2*pad-K+1;

  int in_size     = Ni*B*Ci*Ri;
  int weight_size = Ni*No*K*K;
  int out_size    = No*B*Co*Ro;

  float* in = (float*)malloc(sizeof(float)*in_size);
  float* weight = (float*)malloc(sizeof(float)*weight_size);
  float* out = (float*)malloc(sizeof(float)*out_size);
  float* out_ref_ori = (float*)malloc(sizeof(float)*out_size);

  double* in_d = (double*)malloc(sizeof(double)*in_size);
  double* weight_d = (double*)malloc(sizeof(double)*weight_size);
  double* out_ref = (double*)malloc(sizeof(double)*out_size);

  for( int i = 0; i < in_size; ++i )
    in[i] = rand()/(float)RAND_MAX;

  for( int i = 0; i < weight_size; ++i )
    weight[i] = rand()/(float)RAND_MAX;

  for( int i = 0; i < in_size; ++i )
    in_d[i] = in[i];

  for( int i = 0; i < weight_size; ++i )
    weight_d[i] = weight[i];

  for( int i = 0; i < out_size; ++i ) {
    out_ref[i] = 0;
    out[i] = 0;
    out_ref_ori[i] = 0;
  }


  for( int st = 0; st < 1; ++st ){

    printf("running sw version pad conv...\n");
    sw_conv_forward_pad_impl_f(
        in,
        weight,
        out,
        Ci,
        Ri,
        K,
        Ni,
        No,
        B,
        pad);
    printf("sw version pad conv OK\n");

    /*
    sw_conv_forward_pad_impl_d(
        in_d,
        weight_d,
        out_ref,
        Ci,
        Ri,
        K,
        Ni,
        No,
        B,
        pad);
    printf("sw version pad conv double OK\n");
    */

    sw_conv_forward_pad_impl_f_ori(
        in,
        weight,
        out_ref_ori,
        Ci,
        Ri,
        K,
        Ni,
        No,
        B,
        pad);
    printf("inner loop OK!\n");
  }

  printf("calculating errors...\n");
  float sum = 0, sum_ref = 0;
  for( int i = 0; i < out_size; ++i ) {
   if( fabs(out_ref[i] - out[i]) > 1e-4 ) {
     printf("ERROR at %d: %f vs %f\n", i, out_ref[i], out[i]);
     printf("*********** pad failed ************\n");
     free(out_ref);
     free(out);
     free(in);
     free(weight);
     return ;
   }
   sum += out[i];
   sum_ref += out_ref[i];
  }
  if( fabs(sum_ref - sum) > 1e-4 ) {
     printf("ERROR at SUM: %f vs %f\n", sum_ref, sum);
     printf("*********** pad failed ************\n");
     free(out_ref);
     free(out);
     free(in);
     free(weight);
     return ;
  }
  printf("sum %f vs sum_ref %f athread forward OK!\n", sum, sum_ref);


  free(out_ref);
  free(out);
  free(in);
  free(weight);

  free(weight_d);
  free(in_d);
}

// double pad in
void test_forward_pad() {
  int Ni, No, B, Co, Ro, Ci, Ri, K, pad;
  Ni = 512;
  No = 512;
  B  = 128;
  K  = 3;
  pad = 1;
  Ci = 8;
  Ri = 8;
  Co = Ci+2*pad-K+1;
  Ro = Ri+2*pad-K+1;
