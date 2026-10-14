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

#include <dlfcn.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

extern char** environ;

#include "../file_utils.h"
#include "../source_utils.h"
#include "sycl_common.h"

namespace tvm {
namespace runtime {

namespace {

// ---------------------------------------------------------------------------------------------
// JIT compilation of generated SYCL source.
//   TVM_SYCL_JIT_CACHE=0  disable reuse of libraries compiled from identical source
//   TVM_SYCL_JIT_JOBS=N   compile a module as up to N translation units in parallel
//                         (1 = one translation unit, the original behaviour)
// ---------------------------------------------------------------------------------------------
constexpr const char* kTUBoundary = "\n// TVM_SYCL_TU_BOUNDARY\n";

bool JitCacheEnabled() {
  const char* v = std::getenv("TVM_SYCL_JIT_CACHE");
  return v == nullptr || std::string(v) != "0";
}

int JitJobs() {
  if (const char* v = std::getenv("TVM_SYCL_JIT_JOBS")) {
    return std::max(1, std::atoi(v));
  }
  // Default 1: with DPC++ (intel/llvm 646ab08, CUDA) the -fno-sycl-rdc link of split units costs
  // more than the parallel compile saves (ResNet-18: 35 s as one unit vs. 10 s + 54 s link).
  return 1;
}

std::string ReadTextFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return std::string();
  return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

void WriteTextFile(const std::string& path, const std::string& text) {
  std::ofstream out(path, std::ios::binary);
  out << text;
  ICHECK(out.good()) << "Failed to write " << path;
}

std::string UniqueTempSuffix() {
  static std::atomic<uint64_t> counter{0};
  return ".tmp." + std::to_string(getpid()) + "." + std::to_string(counter++);
}

// Run shell commands concurrently (posix_spawn + waitpid; no threads, no system()).
void RunCommands(const std::vector<std::string>& cmds) {
  std::vector<pid_t> pids;
  for (const std::string& cmd : cmds) {
    VLOG(0) << cmd;
    pid_t pid;
    const char* argv[] = {"sh", "-c", cmd.c_str(), nullptr};
    int rc = posix_spawn(&pid, "/bin/sh", nullptr, nullptr, const_cast<char* const*>(argv), environ);
    ICHECK_EQ(rc, 0) << "posix_spawn failed for: " << cmd;
    pids.push_back(pid);
  }
  std::vector<std::string> failed;
  for (size_t i = 0; i < pids.size(); ++i) {
    int status = 0;
    waitpid(pids[i], &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) failed.push_back(cmds[i]);
  }
  ICHECK(failed.empty()) << "SYCL kernel compilation failed: " << failed.front();
}

// Split generated source into the common prelude and per-function chunks (codegen marks the
// boundaries). Returns an empty vector when the source carries no boundaries.
std::vector<std::string> SplitTranslationUnits(const std::string& source, std::string* prelude) {
  std::vector<std::string> chunks;
  size_t pos = source.find(kTUBoundary);
  if (pos == std::string::npos) return chunks;
  *prelude = source.substr(0, pos);
  const size_t len = std::string(kTUBoundary).size();
  while (pos != std::string::npos) {
    size_t next = source.find(kTUBoundary, pos + len);
    chunks.push_back(source.substr(pos + len, next == std::string::npos ? std::string::npos
                                                                         : next - pos - len));
    pos = next;
  }
  return chunks;
}

std::string WithoutSharedFlag(const std::string& flags) {
  std::istringstream in(flags);
  std::string tok, out;
  while (in >> tok) {
    if (tok == "-shared") continue;
    out += (out.empty() ? "" : " ") + tok;
  }
  return out;
}

// Compile the module source into lib.shared_lib_path. The library is first written under a
// unique temporary name and then renamed, so concurrent builders (e.g. MetaSchedule's
// LocalBuilder processes) never observe a partially written file. The source is published
// next to it only after the library is in place; it is the cache-validation record.
void CompileSharedLibrary(const syclT::SYCL_LIB_COMPILER& lib, const std::string& source) {
  const std::string tmp = lib.shared_lib_path + UniqueTempSuffix();
  const std::string module_def = " -DTVM_SYCL_MODULE_ID=m" + lib.file_path_key;
  std::string prelude;
  std::vector<std::string> chunks = SplitTranslationUnits(source, &prelude);
  const int jobs = std::min<int>(JitJobs(), static_cast<int>(chunks.size()));
  std::vector<std::string> temps;
  if (jobs <= 1) {
    WriteTextFile(tmp + ".cc", source);
    temps.push_back(tmp + ".cc");
    RunCommands({lib.sycl_compiler + " " + lib.sycl_flags + module_def + " " + tmp + ".cc -o " +
                 tmp + ".so"});
  } else {
    // Balance translation units by source length (greedy: longest chunk to the lightest unit).
    std::vector<size_t> order(chunks.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](size_t a, size_t b) { return chunks[a].size() > chunks[b].size(); });
    std::vector<std::string> units(jobs, prelude);
    std::vector<size_t> load(jobs, 0);
    for (size_t i : order) {
      size_t j = std::min_element(load.begin(), load.end()) - load.begin();
      units[j] += std::string(kTUBoundary) + chunks[i];
      load[j] += chunks[i].size();
    }
    const std::string cflags = WithoutSharedFlag(lib.sycl_flags) + " -fno-sycl-rdc -c";
    std::vector<std::string> cmds;
    std::string objs;
    for (int j = 0; j < jobs; ++j) {
      const std::string tu = tmp + "." + std::to_string(j) + ".cc";
      const std::string obj = tmp + "." + std::to_string(j) + ".o";
      WriteTextFile(tu, units[j]);
      temps.push_back(tu);
      temps.push_back(obj);
      cmds.push_back(lib.sycl_compiler + " " + cflags + module_def + " " + tu + " -o " + obj);
      objs += " " + obj;
    }
    RunCommands(cmds);
    RunCommands({lib.sycl_compiler + " " + lib.sycl_flags + " -fno-sycl-rdc" + objs + " -o " +
                 tmp + ".so"});
  }
  std::filesystem::rename(tmp + ".so", lib.shared_lib_path);
  WriteTextFile(tmp + ".src", source);
  std::filesystem::rename(tmp + ".src", lib.source_file_path);
  for (const std::string& f : temps) std::filesystem::remove(f);
}

}  // namespace

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
    // Resolve the generated wrapper once. (The original backend called dlsym on every launch.)
    void* sym = dlsym(so_handler_, private_func_name_.c_str());
    ICHECK(sym != nullptr) << "ERROR:" << dlerror() << ":dlsym public=" << func_name_
                           << " private=" << private_func_name_ << "\n";
    kernel_func_ = reinterpret_cast<KernelFuncT>(sym);
  }
  void operator()(ffi::PackedArgs args, ffi::Any* rv, void** void_args) const {
  auto kernel_func = kernel_func_;
  ThreadWorkLoad wl = launch_param_config_.Extract(args);
  // Both ranges are reversed: SYCL dimension 2 is the fastest-varying one and maps to the
  // backend's x axis, which has the largest work-group and grid limits. The code generator
  // reads threadIdx.d and blockIdx.d from SYCL dimension 2 - d.
  sycl::range<3> k0_dimGrid(wl.grid_dim(2), wl.grid_dim(1), wl.grid_dim(0));
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

  // The queue is in-order, so later kernels, copies and frees already observe this launch.
  // The per-launch host wait inherited from the original backend is therefore not needed for
  // correctness; it is kept as an option (TVM_SYCL_SYNC_EACH_LAUNCH=1) to reproduce earlier
  // measurements. Without it, device faults surface at the next StreamSync/copy/free.
  if (!SyncEachLaunch()) return;
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
  using KernelFuncT = void (*)(sycl::queue&, sycl::range<3>, sycl::range<3>, void**);
  KernelFuncT kernel_func_{nullptr};
  static bool SyncEachLaunch() {
    static const bool enabled = [] {
      const char* v = std::getenv("TVM_SYCL_SYNC_EACH_LAUNCH");
      return v != nullptr && std::string(v) == "1";
    }();
    return enabled;
  }
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
    std::filesystem::create_directories(this->lib_compiler.prefix);
    const std::string source = GetSource("sycl");
    // Reuse a library compiled earlier from the identical source (content-addressed key).
    const bool hit = JitCacheEnabled() &&
                     std::filesystem::exists(this->lib_compiler.shared_lib_path) &&
                     ReadTextFile(this->lib_compiler.source_file_path) == source;
    if (!hit) {
      CompileSharedLibrary(this->lib_compiler, source);
    }
    VLOG(0) << "SYCL JIT " << (hit ? "cache hit: " : "compiled: ")
            << this->lib_compiler.shared_lib_path;
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
