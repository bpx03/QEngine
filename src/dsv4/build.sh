#!/bin/bash
# Build dsv4_run (M1). Own flags: no --use_fast_math (the reference numerics need IEEE exp / division).
set -e
cd "$(dirname "$0")"
OUT=${OUT:-../../build_dsv4}
mkdir -p $OUT
CU=/usr/local/cuda
# [dsv4-rel3-isa] portable: no -march anywhere (plain x86-64); the CPU lane kernels (dsv4_cpu_isa.cpp) are built twice,
# AVX-512 (Zen 4 set) and AVX2 (x86-64-v3), each by a target pragma inside the file, and picked at run time (cpuid;
# DSV4_CPU_ISA=avx2 forces AVX2). CXX: the host compiler (default g++-15, else g++; GCC >= 13 for znver4)
CXX=${CXX:-$(command -v g++-15 || command -v g++)}
GV=$($CXX -dumpversion | cut -d. -f1)
[ "${GV:-0}" -ge 13 ] || { echo "build.sh: $CXX is GCC $GV; the DSV4 CPU kernels need GCC >= 13"; exit 1; }
cpu_objs() {   # cpu_objs DIR [extra flags]: the lane's three objects (kernels tuned for Zen 4: -mtune never changes the ISA)
    $CXX -O3 -std=c++20 $2 -I$CU/include -c dsv4_cpu.cpp -o $1/dsv4_cpu.o
    $CXX -O3 -mtune=znver4 -std=c++20 $2 -DDSV4_ISA_AVX512=1 -I$CU/include -c dsv4_cpu_isa.cpp -o $1/dsv4_cpu_avx512.o
    $CXX -O3 -mtune=znver4 -std=c++20 $2 -DDSV4_ISA_AVX512=0 -I$CU/include -c dsv4_cpu_isa.cpp -o $1/dsv4_cpu_avx2.o
}
cpu_objs $OUT
$CXX -O3 -std=c++20 -I$CU/include -c ../safetensors.cpp -o $OUT/safetensors.o
$CXX -O3 -std=c++20 -I$CU/include -c gguf_reader.cpp -o $OUT/gguf_reader.o   # R16 [dsv4-gguf-iq] (review H1)
nvcc -O3 -std=c++20 -arch=sm_86 -lineinfo -ccbin $CXX -Xcompiler -O3 -c dsv4.cu -o $OUT/dsv4.o
nvcc -O3 -std=c++20 -arch=sm_86 -ccbin $CXX -c dsv4_main.cpp -o $OUT/dsv4_main.o
nvcc -O3 -std=c++20 -arch=sm_86 -ccbin $CXX -c dsv4_sample.cu -o $OUT/dsv4_sample.o   # release prep [dsv4-rel-sample]
nvcc -O3 -std=c++20 -arch=sm_86 -ccbin $CXX -c ../kernels/sample.cu -o $OUT/sample.o
nvcc -arch=sm_86 -ccbin $CXX $OUT/dsv4.o $OUT/dsv4_main.o $OUT/dsv4_cpu.o $OUT/dsv4_cpu_avx512.o $OUT/dsv4_cpu_avx2.o $OUT/safetensors.o $OUT/gguf_reader.o $OUT/dsv4_sample.o $OUT/sample.o -lcublas -lcublasLt -lpthread -o $OUT/dsv4_run
echo built $OUT/dsv4_run
# release prep [dsv4-rel-capi]: the server library (position independent copies of the same objects + the C API)
if [ -n "${LIB:-1}" ]; then
    mkdir -p $OUT/pic
    cpu_objs $OUT/pic -fPIC
    $CXX -O3 -std=c++20 -fPIC -I$CU/include -c ../safetensors.cpp -o $OUT/pic/safetensors.o
    $CXX -O3 -std=c++20 -fPIC -I$CU/include -c gguf_reader.cpp -o $OUT/pic/gguf_reader.o
    $CXX -O3 -std=c++20 -fPIC -I$CU/include -c dsv4_capi.cpp -o $OUT/pic/dsv4_capi.o
    nvcc -O3 -std=c++20 -arch=sm_86 -lineinfo -ccbin $CXX -Xcompiler -O3,-fPIC -c dsv4.cu -o $OUT/pic/dsv4.o
    nvcc -O3 -std=c++20 -arch=sm_86 -ccbin $CXX -Xcompiler -fPIC -c dsv4_sample.cu -o $OUT/pic/dsv4_sample.o
    nvcc -O3 -std=c++20 -arch=sm_86 -ccbin $CXX -Xcompiler -fPIC -c ../kernels/sample.cu -o $OUT/pic/sample.o
    nvcc -shared -arch=sm_86 -ccbin $CXX $OUT/pic/*.o -lcublas -lcublasLt -lpthread -o $OUT/libqdsv4.so
    echo built $OUT/libqdsv4.so
fi
