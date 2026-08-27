/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file sycl_module.cc
 */
#include "sycl_module.h"

#include <tvm/ffi/function.h>

#include <cstdint>
#include <dlfcn.h>
#include <fstream>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "../file_utils.h"
#include "../source_utils.h"
#include "sycl_common.h"

namespace tvm {
namespace runtime {

class SYCLWrappedFunc {
 public:
  // initialize the SYCL function.
  void Init(SYCLModuleNode* m, std::string func_name, size_t num_void_args,
            const std::vector<std::string>& launch_param_tags, void* so_handler,
            const std::string& module_key) {
    VLOG(0) << "Init SYCLWrappedFunc";
    w_ = m->GetGlobalWorkspace();
    m_ = m;
    func_name_ = func_name;
    uint32_t kernel_hash = static_cast<uint32_t>(std::hash<std::string>{}(func_name_));
    private_func_name_ = "w_h" + std::to_string(kernel_hash) + "_m" + module_key;
    so_handler_ = so_handler;
    launch_param_config_.Init(num_void_args, launch_param_tags);
  }
  void operator()(ffi::PackedArgs args, ffi::Any* rv, void** void_args) const {
    VLOG(0) << "enter sycl wrapped func operator()";
    ICHECK(w_->devices.size() != 0) << "No SYCL device";

  using KernelFuncT = void (*)(sycl::queue&, sycl::range<3>, sycl::range<3>, void**);
  auto* sym = dlsym(so_handler_, private_func_name_.c_str());
  ICHECK(sym != nullptr) << "ERROR:" << dlerror() << ":dlsym public=" << func_name_
                         << " private=" << private_func_name_ << "\n";
  auto kernel_func = reinterpret_cast<KernelFuncT>(sym);

  ThreadWorkLoad wl = launch_param_config_.Extract(args);
  sycl::range<3> k0_dimGrid(wl.grid_dim(0), wl.grid_dim(1), wl.grid_dim(2));
  // Swap block dimensions 0 and 2 to match the SYCL dimension limits.
  sycl::range<3> k0_dimBlock(wl.block_dim(2), wl.block_dim(1), wl.block_dim(0));

  syclT::SYCLThreadEntry* t = w_->GetThreadEntry();
  sycl::queue Queue = w_->GetQueue(t->device);
  auto launch_kernel = [&]() {
    try {
      kernel_func(Queue, k0_dimGrid, k0_dimBlock, void_args);
    } catch (const sycl::exception& e) {
      ICHECK(e.code() == sycl::errc::success)
          << "SYCL kernel launch failed: func=" << func_name_
          << " grid=(" << k0_dimGrid[0] << "," << k0_dimGrid[1] << "," << k0_dimGrid[2] << ")"
          << " block=(" << k0_dimBlock[0] << "," << k0_dimBlock[1] << "," << k0_dimBlock[2] << ")"
          << " error=" << e.what();
    }
  };

  if (w_->IsProfiling(t->device)) {
    // Bracket the launch with barrier events and read their timestamps.
    sycl::event ev_begin = Queue.ext_oneapi_submit_barrier();
    launch_kernel();
    sycl::event ev_end = Queue.ext_oneapi_submit_barrier();

    try {
      // Query each profiling timestamp with its concrete enumeration.
      uint64_t beg =
          ev_begin.get_profiling_info<sycl::info::event_profiling::command_submit>();
      uint64_t end =
          ev_end.get_profiling_info<sycl::info::event_profiling::command_end>();
      double us = static_cast<double>(end - beg) / 1000.0;
      VLOG(0) << "[SYCL PROF] " << func_name_ << " ~ " << us << " us";
    } catch (const sycl::exception& e) {
      VLOG(0) << "[SYCL PROF] profiling info unavailable: " << e.what();
    }
  } else {
    launch_kernel();
  }

  try {
    Queue.wait_and_throw();
  } catch (const sycl::exception& e) {
    ICHECK(e.code() == sycl::errc::success)
        << "SYCL kernel wait failed: func=" << func_name_
        << " grid=(" << k0_dimGrid[0] << "," << k0_dimGrid[1] << "," << k0_dimGrid[2] << ")"
        << " block=(" << k0_dimBlock[0] << "," << k0_dimBlock[1] << "," << k0_dimBlock[2] << ")"
        << " error=" << e.what();
  }
  }

 private:
  // global workspace.
  syclT::SYCLWorkspace* w_;
  // The module
  SYCLModuleNode* m_;
  // resource handle
  ObjectPtr<Object> sptr_;
  // The name of the function.
  std::string func_name_;
  // The private exported C wrapper symbol in the generated shared object.
  std::string private_func_name_;
  // convert code for void argument
  std::vector<size_t> arg_size_;
  // launch parameters config
  LaunchParamConfig launch_param_config_;
  // share library handler
  void *so_handler_;
};

SYCLModuleNode::~SYCLModuleNode() {
  // Keep JIT-compiled shared objects loaded until process exit. The DPC++ CUDA
  // plugin stores device-image registrations in process-global state; unloading
  // an image can invalidate kernels that are still referenced by later modules.
}

syclT::SYCLWorkspace* SYCLModuleNode::GetGlobalWorkspace() {
  return syclT::SYCLWorkspace::Global();
}

ffi::Function SYCLModuleNode::GetFunction(const String& name,
                                          const ObjectPtr<Object>& sptr_to_self) {
  VLOG(0) << "SYCLModuleNode::GetFunction: " << name;
  ICHECK_EQ(sptr_to_self.get(), this);
  ICHECK_NE(name, symbol::tvm_module_main) << "Device function do not have main";
  VLOG(0) << "SYCLModuleNode::find fmap_";

  auto it = fmap_.find(name);
  if (it == fmap_.end()) return ffi::Function();
  const FunctionInfo& info = it->second;
  VLOG(0) << "SYCLModuleNode::begin SYCLWrappedFunc";

  SYCLWrappedFunc f;
  std::vector<size_t> arg_size(info.arg_types.size());
  for (size_t i = 0; i < info.arg_types.size(); ++i) {
    DLDataType t = info.arg_types[i];
    ICHECK_EQ(t.lanes, 1U);
    if (t.code == kDLOpaqueHandle) {
      // specially store pointer type size in SYCL driver
      arg_size[i] = sizeof(void*);
    } else {
      uint32_t bits = t.bits;
      ICHECK_EQ(bits % 8, 0U);
      arg_size[i] = bits / 8;
    }
  }
  VLOG(0) << "SYCLModuleNode::begin initialize the wrapped func:";

  // initialize the wrapped func.
  f.Init(this, name, info.arg_types.size(), info.launch_param_tags, so_handler_,
         this->lib_compiler.file_path_key);
  VLOG(0) << "SYCLModuleNode::finish initialize the wrapped func:";

  return PackFuncVoidAddr(f, info.arg_types);
}

String SYCLModuleNode::GetSource(const String& format) {
  if (format == fmt_) return data_;
  if (fmt_ == "sycl") {
    return data_;
  } else {
    return source_;
  }
}

void SYCLModuleNode::Init() {
  workspace_ = GetGlobalWorkspace();
  workspace_->Init();
  // split into source artifacts for each kernel
  parsed_kernels_ = SplitKernels(GetSource("sycl"));
  ICHECK(!parsed_kernels_.empty()) << "The SYCL module expects a kernel delimited "
                                   << "source from code generation, but no kernel "
                                   << "delimiter was found.";
  ICHECK(fmap_.size() == parsed_kernels_.size())
      << "The number of parsed kernel sources does not match the number of kernel functions";
  if (this->lib_compiler.load_from_file) {
    so_handler_ = dlopen(this->lib_compiler.shared_lib_path.c_str(), RTLD_LAZY);
  } else {
    // create the folder to store sycl temporary files
    if (access(this->lib_compiler.prefix.c_str(), F_OK) == -1) {
      std::string cmd = "mkdir " + this->lib_compiler.prefix;
      system(cmd.c_str());
    }
    // sycl kernel source code
    std::ofstream kernels_file;
    kernels_file.open(this->lib_compiler.source_file_path);
    kernels_file << GetSource("sycl");
    kernels_file.close();
    // compile kernel source code to shared library
    VLOG(0) << this->lib_compiler.command;
    system(this->lib_compiler.command.c_str());

    // dlopen sycl share libary
    so_handler_ = dlopen(this->lib_compiler.shared_lib_path.c_str(), RTLD_LAZY);
  }
  ICHECK(so_handler_ != NULL) << "ERROR:" << dlerror() << ":dlopen\n";
}



Module SYCLModuleCreate(std::string data, std::string fmt,
                          std::unordered_map<std::string, FunctionInfo> fmap, std::string source) {
  auto n = make_object<SYCLModuleNode>(data, fmt, fmap, source);
  n->Init();
  return Module(n);
}
Module SYCLModuleCreate(std::string data, std::string fmt,
                          std::unordered_map<std::string, FunctionInfo> fmap, std::string source, std::string path_key) {
  auto n = make_object<SYCLModuleNode>(data, fmt, fmap, source, path_key);
  n->Init();
  return Module(n);
}

void SYCLModuleNode::SaveToFile(const String& file_name, const String& format) {
  std::string fmt = GetFileFormat(file_name, format);
  ICHECK_EQ(fmt, fmt_) << "Can only save to format=" << fmt_;
  std::string meta_file = GetMetaFilePath(file_name);
  SaveMetaDataToFile(meta_file, fmap_);
  SaveBinaryToFile(file_name, data_);
  SaveBinaryToFile(file_name+"_path_key", this->lib_compiler.file_path_key);
}

void SYCLModuleNode::SaveToBinary(dmlc::Stream* stream) {
  stream->Write(fmt_);
  stream->Write(fmap_);
  stream->Write(data_);
  stream->Write(this->lib_compiler.file_path_key);
}

// Load module from module.
Module SYCLModuleLoadFile(const std::string& file_name, const std::string& format) {
  std::string data;
  std::unordered_map<std::string, FunctionInfo> fmap;
  std::string fmt = GetFileFormat(file_name, format);
  std::string meta_file = GetMetaFilePath(file_name);
  LoadBinaryFromFile(file_name, &data);
  LoadMetaDataFromFile(meta_file, &fmap);
  std::string path_key;
  LoadBinaryFromFile(file_name+"_path_key", &path_key);
  return SYCLModuleCreate(data, fmt, fmap, std::string(), path_key);
}

Module SYCLModuleLoadBinary(void* strm) {
  dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
  std::string data;
  std::unordered_map<std::string, FunctionInfo> fmap;
  std::string fmt;
  stream->Read(&fmt);
  stream->Read(&fmap);
  stream->Read(&data);
  std::string path_key;
  stream->Read(&path_key);
  return SYCLModuleCreate(data, fmt, fmap, std::string(), path_key);
}

TVM_FFI_REGISTER_GLOBAL("runtime.module.loadfile_sycl").set_body_typed(SYCLModuleLoadFile);

TVM_FFI_REGISTER_GLOBAL("runtime.module.loadfile_syclbin").set_body_typed(SYCLModuleLoadFile);

TVM_FFI_REGISTER_GLOBAL("runtime.module.loadbinary_sycl").set_body_typed(SYCLModuleLoadBinary);
}  // namespace runtime
}  // namespace tvm
