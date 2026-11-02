cmake_minimum_required(VERSION 3.24)
# nvcc 13.1 rejects the system GCC 16; GCC 15 works
if(NOT CMAKE_CXX_COMPILER AND EXISTS /usr/bin/g++-15)
  set(CMAKE_CXX_COMPILER /usr/bin/g++-15)
  set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-15)
endif()
# GPU targets: default sm_86 (RTX 3090, tuned) for fast dev / local builds. Set before project() and only when the
# user gave none, so -DCMAKE_CUDA_ARCHITECTURES="86;120" (or CUDAARCHS) wins: 120 = RTX 5090 (built in the Docker
# images, community-tested, not tuned).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
  set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA SASS targets (86 = RTX 3090, 120 = RTX 5090)")
endif()
project(qengine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -lineinfo --expt-relaxed-constexpr")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# build identity (remote diagnosis): the [box] start-up line of Flash-Next and the server's /api/stats `build` /
# start-up log (build/qe_build.txt next to the libraries, server/app.py version_info). Set at BUILD time, so a
# `cmake --build` after a commit or an edit names the code it compiled: git hash of the source tree (+ "-dirty" with
# uncommitted changes to tracked files). -DQE_GIT_HASH=... or the QE_GIT_HASH environment variable at configure time
# override it (trees without .git: release exports, the Docker build stage).
if(NOT DEFINED QE_GIT_HASH AND DEFINED ENV{QE_GIT_HASH})
  set(QE_GIT_HASH "$ENV{QE_GIT_HASH}")
endif()
file(WRITE ${CMAKE_BINARY_DIR}/qe_build_id.cmake [=[
set(id "${FORCED}")
if(id STREQUAL "" OR id STREQUAL "unknown")
  set(id "unknown")
  execute_process(COMMAND git -C "${SRC}" rev-parse --show-toplevel OUTPUT_VARIABLE top
                  OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET RESULT_VARIABLE rc)
  if(rc EQUAL 0 AND top)
    file(REAL_PATH "${top}" top)
    file(REAL_PATH "${SRC}" src)
    if(top STREQUAL src)  # the tree itself, not a repository it was unpacked into
      execute_process(COMMAND git -C "${SRC}" rev-parse --short=7 HEAD OUTPUT_VARIABLE h
                      OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
      execute_process(COMMAND git -C "${SRC}" status --porcelain -uno OUTPUT_VARIABLE d
                      OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
      if(h)
        set(id "${h}")
        if(d)
          string(APPEND id "-dirty")
        endif()
      endif()
    endif()
  endif()
endif()
file(MAKE_DIRECTORY "${OUT}/gen")
file(WRITE "${OUT}/qe_build.txt.tmp" "${id}\n")
file(WRITE "${OUT}/qe_build_id.h.tmp" "#define QE_BUILD_ID \"${id}\"\n")
execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different "${OUT}/qe_build.txt.tmp" "${OUT}/qe_build.txt")
execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different "${OUT}/qe_build_id.h.tmp" "${OUT}/gen/qe_build_id.h")
]=])
add_custom_target(qe_build_id ALL
  COMMAND ${CMAKE_COMMAND} -DSRC=${CMAKE_SOURCE_DIR} -DOUT=${CMAKE_BINARY_DIR} "-DFORCED=${QE_GIT_HASH}"
          -P ${CMAKE_BINARY_DIR}/qe_build_id.cmake
  BYPRODUCTS ${CMAKE_BINARY_DIR}/qe_build.txt ${CMAKE_BINARY_DIR}/gen/qe_build_id.h
  COMMENT "build id" VERBATIM)
string(REPLACE ";" "," QE_ARCH_STR "${CMAKE_CUDA_ARCHITECTURES}")
add_compile_definitions(QE_CUDA_ARCHS="${QE_ARCH_STR}")
find_package(CUDAToolkit REQUIRED)

file(GLOB KERNELS CONFIGURE_DEPENDS src/kernels/*.cu)
add_library(qengine SHARED ${KERNELS} src/capi.cpp src/config.cpp src/safetensors.cpp src/model.cpp src/engine.cpp)
target_include_directories(qengine PUBLIC src)
target_link_libraries(qengine PUBLIC CUDA::cudart CUDA::cuda_driver)
add_dependencies(qengine qe_build_id)  # qe_build.txt next to the library is current when it is linked

# Qwen3.8-Flash-Next engine (preview)
# kernels + safetensors without the 27B engine / C API (libqflash must not export libqengine's qe_* symbols)
add_library(qkernels OBJECT ${KERNELS} src/safetensors.cpp)
target_include_directories(qkernels PUBLIC src)
add_executable(qe_w8q src/tools/qe_w8q.cu $<TARGET_OBJECTS:qkernels>)  # INT8 g128 of a bf16 matrix with the engine kernel (convert_flash_next.py --int8)
target_include_directories(qe_w8q PRIVATE src)
target_link_libraries(qe_w8q PRIVATE CUDA::cudart CUDA::cuda_driver)
add_executable(flash_run src/flash/flash.cu src/flash/flash_seq.cu src/flash/flash_pp.cpp src/flash/card_probe.cu src/flash/flash_main.cpp src/build_id.cpp $<TARGET_OBJECTS:qkernels>)
target_include_directories(flash_run PRIVATE src)
target_link_libraries(flash_run PRIVATE CUDA::cudart CUDA::cuda_driver)
target_include_directories(flash_run PRIVATE ${CMAKE_BINARY_DIR}/gen)
add_dependencies(flash_run qe_build_id)
# C API of the Flash-Next engine for the server (server/qe_capi.h tier 0)
add_library(qflash SHARED src/flash/flash.cu src/flash/flash_seq.cu src/flash/flash_pp.cpp src/flash/card_probe.cu src/flash/flash_capi.cpp src/build_id.cpp $<TARGET_OBJECTS:qkernels>)
target_include_directories(qflash PRIVATE src)
target_link_libraries(qflash PRIVATE CUDA::cudart CUDA::cuda_driver)
target_include_directories(qflash PRIVATE ${CMAKE_BINARY_DIR}/gen)
add_dependencies(qflash qe_build_id)
