#!/bin/bash
# Build gptoss_run and libqgptoss.so. No --use_fast_math (IEEE exp / division for the reference numerics), no -march.
set -e
cd "$(dirname "$0")"
OUT=${OUT:-../../build_gptoss}
mkdir -p $OUT/pic
CXX=${CXX:-$(command -v g++-15 || command -v g++)}
NV="nvcc -O3 -std=c++20 -arch=sm_86 -ccbin $CXX ${EXTRA:-}"   # EXTRA: experiment defines (e.g. -DGPTOSS_EXP_NOSCL)
$NV -lineinfo -Xcompiler -O3 -c gptoss.cu -o $OUT/gptoss.o
$NV -c gptoss_main.cpp -o $OUT/gptoss_main.o
$NV $OUT/gptoss.o $OUT/gptoss_main.o -lcublas -lpthread -o $OUT/gptoss_run
echo built $OUT/gptoss_run
$NV -lineinfo -Xcompiler -O3,-fPIC -c gptoss.cu -o $OUT/pic/gptoss.o
$CXX -O3 -std=c++20 -fPIC -c gptoss_capi.cpp -o $OUT/pic/gptoss_capi.o
$CXX -O3 -std=c++20 -fPIC -c capi_srv.cpp -o $OUT/pic/capi_srv.o   # [srv-gptoss-capi]
$CXX -O3 -std=c++20 -fPIC -c capi_cache.cpp -o $OUT/pic/capi_cache.o   # [srv-gptoss-cache]
$NV -shared $OUT/pic/gptoss.o $OUT/pic/gptoss_capi.o $OUT/pic/capi_srv.o $OUT/pic/capi_cache.o -lcublas -lpthread -o $OUT/libqgptoss.so
echo built $OUT/libqgptoss.so
