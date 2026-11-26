#!/bin/bash
# Build ink_run (Inkling 2 Mini one-card engine) and libqinkling.so (the server's C API, server/qe_capi.h).
# No --use_fast_math (IEEE exp / tanh for the reference numerics). LIB=0: ink_run only.
set -e
cd "$(dirname "$0")"
OUT=${OUT:-../../build_inkling}
mkdir -p $OUT
CXX=${CXX:-$(command -v g++-15 || command -v g++)}
NV="nvcc -O3 -std=c++20 -arch=sm_86 -ccbin $CXX ${EXTRA:-}"
$NV -lineinfo -Xcompiler -O3 -c ink.cu -o $OUT/ink.o
$NV -c ink_main.cpp -o $OUT/ink_main.o
$NV $OUT/ink.o $OUT/ink_main.o -lcublas -o $OUT/ink_run
echo built $OUT/ink_run
# [f52_review L3] k_q8tc runs at __launch_bounds__(256, 4) = at most 64 registers: a spill (compiler update, INK_Q8PF = 3)
# would cost speed silently, so the build says so. [ink-1c8] review L2: the same for every k_q8w form (256, 2: <= 128
# registers; <2, 4> sits at 106) and k_dec8; the INK_Q8W=0 off-switch forms (k_g12tc I8 at S = 2) are only reported
if command -v cuobjdump > /dev/null; then
  ru=$(cuobjdump -res-usage $OUT/ink.o 2>/dev/null | grep -A1 -E "Function _ZN3ink(6k_q8tc|5k_q8w|6k_dec8|7k_g12tcILi3072E[^ ]*Lb1EEE)" \
       | grep -oE "Function [^:]*|REG:[0-9]* STACK:[0-9]* .*LOCAL:[0-9]*" | paste - - | sed 's/^Function //; s/ SHARED:[0-9]*//')
  while read -r fn res; do
    [ -z "$fn" ] && continue
    case $fn in *k_g12tc*) tag="note (INK_Q8W=0 off-switch form, speed only)";; *) tag=WARNING;; esac
    echo "$res" | grep -q "STACK:0 .*LOCAL:0" || echo "$tag: $fn spills: $res"
  done <<< "$ru"
  n=$(echo "$ru" | grep -cE "k_q8tc|k_q8w|k_dec8")
  [ "$n" -ge 6 ] && echo "k_q8tc / k_q8w / k_dec8: $n kernels checked for spills" || echo "WARNING: spill check found $n of the 6 k_q8tc / k_q8w / k_dec8 kernels"
fi
if [ "${LIB:-1}" != 0 ]; then
  # [ink-srv-capi] the shared library: the engine again with -fPIC (same source and flags) + the C API
  $NV -lineinfo -Xcompiler -O3,-fPIC -c ink.cu -o $OUT/ink_pic.o
  $NV -Xcompiler -O3,-fPIC -c ink_capi.cpp -o $OUT/ink_capi.o
  $NV -Xcompiler -O3,-fPIC -c ink_capi_cache.cpp -o $OUT/ink_capi_cache.o
  $NV -shared $OUT/ink_pic.o $OUT/ink_capi.o $OUT/ink_capi_cache.o -lcublas -o $OUT/libqinkling.so
  echo built $OUT/libqinkling.so
fi
