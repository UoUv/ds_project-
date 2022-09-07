
/*************************************************************************
	> File Name: conv_layer_impl.h
	> Author: Jiarui Fang 
	> mail: fang_jiarui@163.com
  > Created Time: Fri 30 Dec 2016 10:24:37 AM CST
  > This file provide a MPE version for correctness check for SPE version on Sunway
 ************************************************************************/
#ifndef CONVLAYER_IMPL
#define CONVLAYER_IMPL
#include <stdio.h>

//TODO B, Ni, Ci, Ri
int inGetIdx(int cB, int cNi, int cRi, int cCi, int B, int Ni, int Ri,int Ci){
//  return cB + cNi*B + cCi*B*Ni + cRi*Ci*Ni*B;
  return (((cB * Ni + cNi)*Ri + cRi)*Ci + cCi);
}

//TODO B, No, Co, Ro
int outGetIdx(int cB, int cNo, int cRo, int cCo, int B, int No, int Ro, int Co){
//  return cB + cNo*B + cCo*B*No + cRo*Co*No*B;
  return (((cB * No + cNo)*Ro + cRo)*Co + cCo);
}

//Ni, No, K, K
//TODO FJR diff from SW version
//Kc, Kr, Ni, No
int weightGetIdx(int cNo, int cNi, int cKr, int cKc,  int No, int Ni, int K){
  //return cNo + cNo*Ni + (cKr*K + cKc)*No*Ni;
  return (((cNo*Ni + cNi)*K + cKr)*K + cKc);
}

inline int offset(const int n, const int c, const int h,
    const int w,
    const int batchs,
    const int channels,
    const int height,
    const int width) {
  return ((n * channels + c) * height + h) * width + w;
}

template<typename Type>
void conv_forward_pad_impl(Type* input,
    Type* weight,
    Type* output,
        int Ci,
        int Ri,
        int K,
        int Ni,
        int No,
        int B,
        int pad)
{
  int cB,cNo,cNi,cRo,cCo,cKr,cKc;
  int Co = Ci+2*pad-K+1;
  int Ro = Ri+2*pad-K+1;


  for(cB = 0; cB<B; cB++)
    for(cNo=0; cNo<No; cNo++)
      for(cNi = 0; cNi<Ni; cNi++)
        for(cRo=0; cRo<Ro; cRo++)
          for(cCo=0; cCo<Co; cCo++)
            for(cKr = 0 ;cKr<K; cKr++)
              for(cKc = 0; cKc<K; cKc++)
              {
                  int cRi = cRo+cKr-pad;
                  int cCi = cCo+cKc-pad;
                  if(cRi >= 0 && cRi < Ri && cCi >= 0 && cCi < Ci) {
                    *(output + outGetIdx(cB, cNo, cRo, cCo, B, No, Ro, Co)) +=
                      *(input + inGetIdx(cB, cNi, cRi, cCi, B, Ni, Ri, Ci)) *
                      *(weight + weightGetIdx(cNo, cNi, cKr, cKc, No, Ni, K));
                  }
              }
  printf("conv output forward is OK\n");
}

template<typename Type>
void conv_forward_impl(
    const Type* input,
    const Type* weight,
    Type* output,
    //const Type* bias,
        int Ci,
        int Ri,
        int K,
        int Ni,
        int No,
        int B)
{
  int cB,cNo,cNi,cRo,cCo,cKr,cKc;