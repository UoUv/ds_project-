#include <stdio.h>
#include <assert.h>
#include "athread.h"
#include <math.h>

#include "../include/swlstm.h"

extern SLAVE_FUN(lstm_slave_clip_forward_f)();
extern SLAVE_FUN(lstm_slave_noclip_forward_f)