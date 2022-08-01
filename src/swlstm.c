#include <stdio.h>
#include <assert.h>
#include "athread.h"
#include <math.h>

#include "../include/swlstm.h"

extern SLAVE_FUN(lstm_slave_clip_forward_f)();
extern SLAVE_FUN(lstm_slave_noclip_forward_f)();
extern SLAVE_FUN(lstm_std_slave_forward_f)();
void sw_lstm_clip_forward_impl_f(
        float * clip_t,
        float * pre_gate_t,
        float * h_to_gate,
        float * ga