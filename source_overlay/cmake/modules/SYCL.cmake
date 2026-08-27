# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0

if(USE_SYCL)
  tvm_file_glob(GLOB RUNTIME_SYCL_SRCS src/runtime/sycl/*.cc)
  list(APPEND RUNTIME_SRCS ${RUNTIME_SYCL_SRCS})

  if(${USE_SYCL} MATCHES ${IS_TRUE_PATTERN})
    find_program(SYCL_CXX_COMPILER NAMES icpx clang++ REQUIRED)
  else()
    set(SYCL_CXX_COMPILER ${USE_SYCL})
  endif()

  get_filename_component(SYCL_BIN_DIR ${SYCL_CXX_COMPILER} DIRECTORY)
  get_filename_component(SYCL_ROOT ${SYCL_BIN_DIR} DIRECTORY)
  find_library(SYCL_RUNTIME_LIBRARY sycl HINTS "${SYCL_ROOT}/lib" REQUIRED)

  if(NOT DEFINED SYCL_FLAGS)
    set(SYCL_FLAGS "-fsycl -fPIC -shared")
  endif()
  if(NOT DEFINED SYCL_TEMP_FOLDER)
    set(SYCL_TEMP_FOLDER "/tmp/tvm_sycl")
  endif()

  file(MAKE_DIRECTORY ${SYCL_TEMP_FOLDER})
  set_source_files_properties(
    ${RUNTIME_SYCL_SRCS}
    src/target/source/codegen_sycl.cc
    PROPERTIES COMPILE_OPTIONS "-fsycl"
  )
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${SYCL_RUNTIME_LIBRARY})
  add_compile_definitions(
    SYCL_CXX_COMPILER="${SYCL_CXX_COMPILER}"
    SYCL_FLAGS="${SYCL_FLAGS}"
    SYCL_TEMP_FOLDER="${SYCL_TEMP_FOLDER}"
  )
  message(STATUS "Build with SYCL support: ${SYCL_CXX_COMPILER}")
else()
  list(REMOVE_ITEM COMPILER_SRCS src/target/source/codegen_sycl.cc)
endif()
