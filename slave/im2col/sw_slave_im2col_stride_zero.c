#include "slave.h"
#include "simd.h"
#include "dma.h"
#include "include/swim2col.h"

/*******
 * col (Ni*K*K, Co*Ro)
 * make Ni*K*K is 8x, fortunately we do not align Co*Ro*Batch
 * make Co*Ro is 128x
 * read batch_size input image rows to increase avai