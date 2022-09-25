/****
 * Jiarui Fang
 * fang_jiarui@163,com
 * ****/
#include "include/swim2col.h"
#include "include/swcommon.h"
#include "./include/swtensortrans.h"
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "swblas.h"

/******
 * a unitest for batch-pad-im2col
 * Optimizations:
 * 1. batch im2col: transpose input features (B, N, R, C) -> (N, R, C, B), then perform batch-im2col
 * 2. zeropadding, (N, R, C, B) -> (K*K*N + pad, Ro*Co*B +pad),