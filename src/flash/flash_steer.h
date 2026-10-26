#pragma once
// Activation-steering vectors (QE_STEER / QE_STEER_DIR): the loader shared with the 27B engine (src/steer.h: the
// llama.cpp control-vector GGUF + JSON sidecar, validations, per-request mode / layers, the vector set and its info).
//
// Where the vector acts on Flash-Next (see Worker::steer / k_steer): at the output of layer L (after the FFN
// hyper-connection combine, llama.cpp's "l_last"), on each of the 4 HC streams: add R_i += alpha v, project
// R_i -= alpha (R_i . v) v.
#include "../steer.h"

namespace flash {
using namespace qsteer;
}  // namespace flash
