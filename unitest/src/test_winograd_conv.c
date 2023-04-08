
/***
 * Jerry Fang
 * 2018.9.18
 * Do not forget the shame of our nation
 * **/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cblas.h>
#include <string.h>
#include <malloc.h>
#include <math.h>

#include <assert.h>
#include "include/swcommon.h"
#include "include/swwinogradconv.h"

//R = filter_row
//S = filter_col
//P = Co
//Q = Ro
//K = No
//direct_conv with no input pad
void direct_conv(float * D0, float * F, float * O, const int N, const int K, const int P, const int Q, const int C, const int R, const int S) {
    const int P_pad = P + 2; 
    const int Q_pad = Q + 2; 
    int n, k, p, q, c, r, s; 
    float sum; 
    for (n = 0; n < N; n++) {
#pragma omp parallel for
        for (k = 0; k < K; k++) {
            for (p = 1; p < P_pad-1; p++) {
                for (q = 1; q < P_pad-1; q++) {
                    sum = 0; 
#pragma unroll
                    for (c = 0; c < C; c++) {
#pragma unroll
                        for (r = 0; r < R; r++) {
#pragma unroll
                            for (s = 0; s < S; s++) {
                                sum += F[k*C*R*S + c*R*S + r*S + s]*D0[n*C*P_pad*Q_pad + c*P_pad*Q_pad + (p+r-1)*Q_pad + (q+s-1)]; 
                            }
                        }
                    }
                    O[n*K*P*Q+ k*P*Q+ (p-1)*Q+ (q-1)] = sum; 
                }
            }
        }
    }
}


void fjr_direct_conv(float * D0, float * F, float * O, const int N, const int K, const int P, const int Q, const int C, const int R, const int S) {
    const int P_pad = P + 2; 
    const int Q_pad = Q + 2; 
    int n, k, p, q, c, r, s; 
    float sum; 
    for (n = 0; n < N; n++) {