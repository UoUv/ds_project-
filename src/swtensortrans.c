
#include <stdio.h>
#include <stdlib.h>
#include <athread.h>
#include <simd.h>
#include "./include/swtensortrans.h"

extern void SLAVE_FUN(swapBN)();
extern void SLAVE_FUN(swapNBHW)();
extern void SLAVE_FUN(swapNBHW_ROLL)();
extern void SLAVE_FUN(swapBN_f)();
extern void SLAVE_FUN(swapNBHW_f)();
extern void SLAVE_FUN(swapNBHW_ROLL_f)();

// high -> low
// B, N, W, H
inline int image_caffe_offset(int b, int n, int h, int w, int B, int N, int H, int W) {
  return (((b*N + n)*H + h)*W + w);
}

// W, H, N, B
inline int image_swdnn_offset(int b, int n, int h, int w, int B, int N, int H, int W) {
  return (((h*W + w)*N + n)*B + b);
}
inline int get_split_size(int nSize,int nMaxSize)
{
	int nVal = nSize/nMaxSize,nSplitSize = 0;
	if(nVal<1) 
	{
      nSplitSize = nSize - nSize%4;
	}
	else if(nVal>=nMaxSize) nSplitSize = nMaxSize;