# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License").

include(GNUInstallDirs)

set(XLLM_MLU_ARCHITECTURES "mtp_592" CACHE STRING
    "BangC device architecture; the initial PLE implementation supports mtp_592")
if(NOT XLLM_MLU_ARCHITECTURES STREQUAL "mtp_592")
  message(FATAL_ERROR
    "xLLM's current BangC PLE kernels support only mtp_592; "
    "requested XLLM_MLU_ARCHITECTURES=${XLLM_MLU_ARCHITECTURES}")
endif()

if(NOT NEUWARE_HOME)
  if(DEFINED ENV{NEUWARE_HOME} AND NOT "$ENV{NEUWARE_HOME}" STREQUAL "")
    set(NEUWARE_HOME "$ENV{NEUWARE_HOME}")
  else()
    set(NEUWARE_HOME "/usr/local/neuware")
  endif()
endif()
list(APPEND CMAKE_MODULE_PATH
  "${NEUWARE_HOME}/cmake" "${NEUWARE_HOME}/cmake/modules")
set(NEUWARE_ROOT_DIR "${NEUWARE_HOME}")
find_package(BANG REQUIRED)
if(NOT BANG_CNCC_EXECUTABLE)
  message(FATAL_ERROR "cncc is required to build xLLM BangC sources")
endif()
find_library(XLLM_BANGC_CNRT_LIBRARY NAMES cnrt
  HINTS "${NEUWARE_HOME}/lib64" "${NEUWARE_HOME}/lib" REQUIRED)

# Keep kernel compilation separate from Torch headers and host registration.
# FindBANG tracks the .mlu sources and their includes for incremental builds.
file(GLOB_RECURSE XLLM_BANGC_KERNEL_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.mlu")
file(GLOB_RECURSE XLLM_BANGC_API_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/torch_api/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/register/*.cpp")
if(NOT XLLM_BANGC_KERNEL_SOURCES OR NOT XLLM_BANGC_API_SOURCES)
  message(FATAL_ERROR "xLLM BangC requires both kernel and Tensor API sources")
endif()

string(APPEND BANG_CNCC_FLAGS
  " -std=c++17 -fPIC --bang-mlu-arch=${XLLM_MLU_ARCHITECTURES}"
  " -D_GLIBCXX_USE_CXX11_ABI=${USE_CXX11_ABI}")
# FindBANG expects this list even with a single-configuration Ninja generator;
# without it, Release would silently omit its device optimization flags.
set(BANG_configuration_types Debug Release RelWithDebInfo MinSizeRel)
list(APPEND BANG_CNCC_FLAGS_DEBUG -O0 -g)
list(APPEND BANG_CNCC_FLAGS_RELEASE -O3 -DNDEBUG)
list(APPEND BANG_CNCC_FLAGS_RELWITHDEBINFO -O3 -g -DNDEBUG)
list(APPEND BANG_CNCC_FLAGS_MINSIZEREL -Os -DNDEBUG)
# DFlash2 requires BF16 multiply and add to round separately. CNAS's O3
# instruction-combine pass changes intermediate rounding despite -ffp-contract=off.
# Keep O3 scheduling, but disable that backend pass only for the two convolution
# translation units. Sampling and the existing PLE kernels retain their flags.
set(XLLM_DFLASH2_CONV_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/src/dflash2_grouped_conv/kernel.mlu"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/dflash2_conv_norm/conv_norm.mlu")
list(REMOVE_ITEM XLLM_BANGC_KERNEL_SOURCES ${XLLM_DFLASH2_CONV_SOURCES})
bang_add_library(xllm_mlu_bangc_kernels STATIC ${XLLM_BANGC_KERNEL_SOURCES})
bang_add_library(xllm_mlu_dflash2_conv_kernels STATIC ${XLLM_DFLASH2_CONV_SOURCES}
  OPTIONS -Xbang-cnas -fno-instruction-combine)
foreach(bangc_target xllm_mlu_bangc_kernels xllm_mlu_dflash2_conv_kernels)
  target_include_directories(${bangc_target} PRIVATE
    "${PROJECT_SOURCE_DIR}/xllm/core" "${NEUWARE_HOME}/include")
  set_target_properties(${bangc_target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endforeach()

add_library(xllm_mlu_ops SHARED ${XLLM_BANGC_API_SOURCES})
add_library(xllm::mlu_ops ALIAS xllm_mlu_ops)
target_include_directories(xllm_mlu_ops
  PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/xllm/core>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/xllm/core>"
  PRIVATE "${NEUWARE_HOME}/include")
target_compile_definitions(xllm_mlu_ops PUBLIC
  "_GLIBCXX_USE_CXX11_ABI=${USE_CXX11_ABI}")
target_link_libraries(xllm_mlu_ops
  PUBLIC torch
  PRIVATE xllm_mlu_bangc_kernels xllm_mlu_dflash2_conv_kernels torch_mlu glog::glog
    ${XLLM_BANGC_CNRT_LIBRARY})
# Keep private archive symbols local, including glog/gflags registries that
# may also be linked statically into the executable loading this library.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_options(xllm_mlu_ops PRIVATE "LINKER:--exclude-libs,ALL")
endif()
# A registration-only translation unit is compiled directly into this shared
# object, so static-archive dead stripping cannot remove its initializers.
set_target_properties(xllm_mlu_ops PROPERTIES
  OUTPUT_NAME xllm_mlu_ops
  POSITION_INDEPENDENT_CODE ON
  BUILD_WITH_INSTALL_RPATH ON
  INSTALL_RPATH
    "$ORIGIN;$ORIGIN/../torch/lib;$ORIGIN/../torch_mlu/csrc/lib;$ORIGIN/../torch_mlu;${NEUWARE_HOME}/lib64;${NEUWARE_HOME}/lib"
  INSTALL_RPATH_USE_LINK_PATH FALSE)

install(TARGETS xllm_mlu_ops LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/xllm/core/kernels/mlu/bangc/include"
  FILES_MATCHING PATTERN "*.h")
