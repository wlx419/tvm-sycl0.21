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
 * \file sycl_common.h
 * \brief SYCL common header
 */
#ifndef TVM_RUNTIME_SYCL_SYCL_COMMON_H_
#define TVM_RUNTIME_SYCL_SYCL_COMMON_H_


// #include <tvm/runtime/logging.h>
// #include <tvm/runtime/device_api.h>
// #include <tvm/runtime/packed_func.h>
// #include <tvm/runtime/profiling.h>

#include <tvm/ffi/function.h>
#include <tvm/runtime/base.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/memory/memory_manager.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/profiling.h>
//#include <dlpack/dlpack.h>

#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <fcntl.h>  //open("/dev/urandom", O_RDONLY);
#ifdef _WIN32
    #include <process.h>
#else
    #include <unistd.h>
#endif

#include "../file_utils.h"
#include "../meta_data.h"
#include "../pack_args.h"
#include "../texture.h"
#include "../thread_storage_scope.h"
#include "../workspace_pool.h"

namespace tvm {
namespace runtime {
namespace syclT {

inline const char* SYCLGetErrorString(std::error_code error_code) {
  sycl::errc error_code_value = static_cast<sycl::errc>(error_code.value());
  switch(error_code_value){
    case sycl::errc::success:
      return "SUCCESS";
    case sycl::errc::runtime:
      return "RUNTIME ERROR";
    case sycl::errc::kernel:
      return "KERNEL ERROR";
    case sycl::errc::accessor:
      return "ACCESSOR ERROR";
    case sycl::errc::nd_range:
      return "NDRANGE ERROR";
    case sycl::errc::event:
      return "EVENT ERROR";
    case sycl::errc::kernel_argument:
      return "KERNEL ARGUMENT ERROR";
    case sycl::errc::build:
      return "BUILD ERROR";
    case sycl::errc::invalid:
      return "INVALID ERROR";
    case sycl::errc::memory_allocation:
      return "MEMORY ALLOCATION";
    case sycl::errc::platform:
      return "PLATFORM ERROR";
    case sycl::errc::profiling:
      return "PROFILING ERROR";
    case sycl::errc::feature_not_supported:
      return "FEATURE NOT SUPPORTED";
    case sycl::errc::kernel_not_supported:
      return "KERNEL NOT SUPPORTED";
    case sycl::errc::backend_mismatch:
      return "BACKEND MISMATCH";
  }
  return "UNKNOWN ERROR";
}

/*!
 * \brief Protected SYCL call
 * \param func Expression to call.
 */
#define SYCL_CALL(func)                                                               \
  do {                                                                                \
    try {                                                                             \
      func;                                                                           \
    } catch (const sycl::exception& e) {                                               \
      LOG(FATAL) << "SYCL call failed, code=" << e.code().value() << ": " << e.what(); \
    }                                                                                 \
  } while (0)

/*!
 * \brief Whether the device queues are created with discard_events.
 * The runtime never consumes the event of a kernel launch, so the queue need not create one per
 * submission; with this property every event returned by the queue is invalid, and the runtime
 * synchronizes on the (in-order) queue instead. It cuts the host cost of a launch (RTX 4090:
 * 5.1 -> 4.1 us; gfx906 with the 2022 HIP plugin: about 20 -> 7 us) but slows command-graph
 * replay in the DPC++ versions we use, so it is the default only for builds without
 * sycl_ext_oneapi_graph. TVM_SYCL_DISCARD_EVENTS=1/0 overrides the default.
 */
inline bool DiscardEvents() {
  static const bool enabled = [] {
    const char* v = std::getenv("TVM_SYCL_DISCARD_EVENTS");
    if (v != nullptr) return std::string(v) == "1";
#ifdef SYCL_EXT_ONEAPI_GRAPH
    return false;
#else
    return true;
#endif
  }();
  return enabled;
}

/*!
 * \brief DPC++ flags for the JIT compile of generated device code.
 * For the CUDA plugin (nvptx64), DPC++'s implicit global offset is disabled: TVM never launches a
 * kernel with an offset, and the implicit offset otherwise gives every kernel a zero-initialized
 * 16-byte stack frame and a duplicate "_with_offset" kernel. TVM_SYCL_GLOBAL_OFFSET=1 keeps it.
 * Floating-point contraction is also enabled across statements (-ffp-contract=fast), the
 * default of nvcc (--fmad=true) that TVM's CUDA path compiles with: DPC++'s default contracts
 * only within one expression, and sycl::vec arithmetic is a chain of operator calls, so DLight's
 * `acc = acc + a * b` stayed a separate multiply and add. TVM_SYCL_FP_CONTRACT=0 keeps DPC++'s
 * default. TVM_SYCL_EXTRA_FLAGS appends flags. The flags are part of the JIT cache key.
 */
inline std::string DefaultJitFlags() {
  std::string flags = SYCL_FLAGS;
  const bool nvptx = flags.find("nvptx64") != std::string::npos;
  const char* keep_offset = std::getenv("TVM_SYCL_GLOBAL_OFFSET");
  if (!(keep_offset != nullptr && std::string(keep_offset) == "1") && nvptx) {
    flags += " -Xclang -mllvm -Xclang -enable-global-offset=false";
  }
  const char* fp_contract = std::getenv("TVM_SYCL_FP_CONTRACT");
  if (!(fp_contract != nullptr && std::string(fp_contract) == "0") && nvptx) {
    flags += " -ffp-contract=fast";
  }
  if (const char* extra = std::getenv("TVM_SYCL_EXTRA_FLAGS")) {
    flags += std::string(" ") + extra;
  }
  return flags;
}

class SYCL_LIB_COMPILER {
  public:
    SYCL_LIB_COMPILER(){}
    SYCL_LIB_COMPILER(int module_id){
      int pid = (int)getpid();
      file_path_key = std::to_string(pid) + "_" +std::to_string(module_id);
      std::string filename = prefix + "/sycl_" + file_path_key;
      source_file_path = filename + ".cc";
      shared_lib_path = filename + ".so";
      command = sycl_compiler +" "+sycl_flags +" -DTVM_SYCL_MODULE_ID=m"+file_path_key+
                " "+ source_file_path +" -o "+shared_lib_path;
    }
    SYCL_LIB_COMPILER(std::string path_key){
      load_from_file = true;
      file_path_key = path_key;
      std::string filename = prefix + "/sycl_" + file_path_key;
      source_file_path = filename + ".cc";
      shared_lib_path = filename + ".so";
      command = sycl_compiler +" "+sycl_flags +" -DTVM_SYCL_MODULE_ID=m"+file_path_key+
                " "+ source_file_path +" -o "+shared_lib_path;
    }
    // Content-addressed library: the key hashes the kernel source together with the compiler and
    // its flags, so an identical module (another process, a re-applied database, a rebuilt
    // deployment) maps to the same file name and the same exported symbols, and can reuse the
    // shared library that is already on disk.
    static SYCL_LIB_COMPILER FromContent(const std::string& source) {
      SYCL_LIB_COMPILER c;
      uint64_t h = 1469598103934665603ULL;  // FNV-1a, 64-bit
      auto mix = [&h](const std::string& s) {
        for (unsigned char ch : s) {
          h ^= ch;
          h *= 1099511628211ULL;
        }
        h ^= 0xff;
        h *= 1099511628211ULL;
      };
      mix(source);
      mix(c.sycl_flags);
      mix(c.sycl_compiler);
      char buf[17];
      snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(h));
      c.file_path_key = std::string("c") + buf;
      c.SetPaths();
      return c;
    }
    void SetPaths() {
      std::string filename = prefix + "/sycl_" + file_path_key;
      source_file_path = filename + ".cc";
      shared_lib_path = filename + ".so";
      command = sycl_compiler + " " + sycl_flags + " -DTVM_SYCL_MODULE_ID=m" + file_path_key + " " +
                source_file_path + " -o " + shared_lib_path;
    }
    std::string sycl_compiler = SYCL_CXX_COMPILER;
    std::string sycl_flags = DefaultJitFlags();
    std::string prefix = SYCL_TEMP_FOLDER;
    std::string file_path_key;
    std::string source_file_path;
    std::string shared_lib_path;
    std::string command;
    bool load_from_file = false;
  private:
    std::string getUUID(unsigned int len = 5){
      std::string str;
      unsigned char* buffer = new unsigned char[len+1];
      char* uuid = new char [2*len+1];
      int fd = open("/dev/urandom", O_RDONLY);
      if (read(fd, buffer, len) == len) {
        for(int i = 0;i < len;i++)
            sprintf(uuid + i*2, "%02X",buffer[i]);
        uuid[2*len] = '\0';
        str = uuid;
        
      } else {
        printf("Error: GetUnique %d\n", __LINE__);
      }
      delete [] buffer;
      delete [] uuid;
      return str;
    }
};

class SYCLThreadEntry;

/*!
 * \brief Process global SYCL workspace.
 */
class SYCLWorkspace : public DeviceAPI {
 public:
  // global platform
  std::vector<sycl::platform> platforms;
  // global platform name
  std::vector<std::string> platform_names;
  // global context of this process
  std::vector<sycl::context> contexts;
  // whether the workspace it initialized.
  bool initialized_{false};
  // the device type
  std::string device_type;
  // the devices
  std::vector<sycl::device> devices;
  // the queues
  std::vector<sycl::queue> queues;
  // the events
  std::vector<std::vector<sycl::event>> events;
  // the mutex for initialization
  std::mutex mu;
  // the number of sycl_modules
  int module_num = 0;
  // destructor
  ~SYCLWorkspace() {
    for(auto queue : queues){
      SYCL_CALL(queue.wait_and_throw());
    }
  }
  // Initialzie the device.
  void Init(const std::string& device_type, const std::string& platform_name = "");
  virtual void Init() { Init("gpu"); }
  // Check whether the context is SYCL or not.
  virtual bool IsSYCLDevice(Device dev) {
    return dev.device_type == static_cast<DLDeviceType>(kDLOneAPI);
  }
  // get the queue of the device
  sycl::queue GetQueue(Device dev);
  // get the event queue of the context
  std::vector<sycl::event>& GetEventQueue(Device dev);
  // is current syclQueue in profiling mode
  bool IsProfiling(Device dev);

  // override device API
  void SetDevice(Device dev) final;
  void GetAttr(Device dev, DeviceAttrKind kind, ffi::Any* rv) final;
  void* AllocDataSpace(Device dev, size_t size, size_t alignment, DLDataType type_hint) final;
  void FreeDataSpace(Device dev, void* ptr) final;
  void* AllocWorkspace(Device dev, size_t size, DLDataType type_hint) final;
  void FreeWorkspace(Device dev, void* data) final;
  void StreamSync(Device dev, TVMStreamHandle stream) final;

  /*!
   * \brief Get the thread local ThreadEntry
   */
  virtual SYCLThreadEntry* GetThreadEntry();

  // get the global workspace
  static SYCLWorkspace* Global();

  void CopyDataFromTo(DLTensor* from, DLTensor* to, TVMStreamHandle stream) final;  
                           
};

/*! \brief Thread local workspace */
class SYCLThreadEntry {
 public:
  /*! \brief The current device */
  Device device;
  /*! \brief workspace pool */
  WorkspacePool pool;
  // constructor
  SYCLThreadEntry(DLDeviceType device_type, DeviceAPI* device_api)
      : pool(device_type, device_api) {
    device.device_id = 0;
    device.device_type = device_type;
  }
  SYCLThreadEntry() : SYCLThreadEntry(DLDeviceType(kDLOneAPI), SYCLWorkspace::Global()) {}

  // get the global workspace
  static SYCLThreadEntry* ThreadLocal();
};

}  // namespace syclT

// Module to support thread-safe multi-device execution.
// SYCL runtime is a bit tricky because clSetKernelArg is not thread-safe
// To make the call thread-safe, we create a thread-local kernel table
// and lazily install new kernels into the kernel table when the kernel is called.
// The kernels are recycled when the module get destructed.
class SYCLModuleNode : public ModuleNode {
 public:
  explicit SYCLModuleNode(std::string data, std::string fmt,
                            std::unordered_map<std::string, FunctionInfo> fmap, std::string source)
      : data_(data), fmt_(fmt), fmap_(fmap), source_(source) {
        workspace_ = GetGlobalWorkspace();
        workspace_->module_num++;           //the number of sycl_modules add 1
        // TVM_SYCL_JIT_CACHE=0 restores the original per-process, per-module file names
        // (no reuse across modules or processes).
        const char* cache = std::getenv("TVM_SYCL_JIT_CACHE");
        if (cache != nullptr && std::string(cache) == "0") {
          lib_compiler = syclT::SYCL_LIB_COMPILER(workspace_->module_num);
        } else {
          lib_compiler = syclT::SYCL_LIB_COMPILER::FromContent(data);
        }
      }
  // load from file
  explicit SYCLModuleNode(std::string data, std::string fmt,
                            std::unordered_map<std::string, FunctionInfo> fmap, std::string source, std::string path_key)
      : data_(data), fmt_(fmt), fmap_(fmap), source_(source) {
        workspace_ = GetGlobalWorkspace();
        lib_compiler = syclT::SYCL_LIB_COMPILER(path_key);
      }
  // destructor
  ~SYCLModuleNode();

  /*!
   * \brief Get the global workspace
   */
  virtual syclT::SYCLWorkspace* GetGlobalWorkspace();

  const char* type_key() const final { return "sycl"; }
  int GetPropertyMask() const final {
    return ModulePropertyMask::kBinarySerializable | ModulePropertyMask::kRunnable;
  }
  ffi::Function GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) final;

  void SaveToFile(const String& file_name, const String& format) final;

  void SaveToBinary(dmlc::Stream* stream) final;
  String GetSource(const String& format) final;

  // Initialize the programs
  void Init();

 private:
  // The workspace, need to keep reference to use it in destructor.
  // In case of static destruction order problem.
  syclT::SYCLWorkspace* workspace_;
  // the binary data
  std::string data_;
  // The format
  std::string fmt_;
  // function information table. Mapping from primitive name to .
  std::unordered_map<std::string, FunctionInfo> fmap_;
  // The SYCL source.
  std::string source_;
  // parsed kernel data, Mapping from primitive name to kernel code.
  std::unordered_map<std::string, std::string> parsed_kernels_;
  //id of SYCLModuleNode, start from 1, to detemine unique filename of source code and share library
  syclT::SYCL_LIB_COMPILER lib_compiler;
  // share library handler
  void * so_handler_ = nullptr;
  // share library name
  std::string dynamic_library_name;
  // share library path
  static const std::string library_path;
};

/*!
 * \brief SYCL timer node (used by time_evaluator, e.g. in MetaSchedule measurement).
 *
 * Waits for the device queue at Start and Stop, like TVM's DefaultTimerNode. The inherited timer
 * read the host clock only; that matched device time only while every launch waited for its
 * kernel, and with asynchronous dispatch it measured enqueue time, so time_evaluator kept
 * raising its repeat count until the queue filled (thousands of launches per candidate).
 */
class SYCLTimerNode : public TimerNode {
 public:
  explicit SYCLTimerNode(Device dev) : device_(dev) {}
  void Start() final {
    DeviceAPI::Get(device_)->StreamSync(device_, nullptr);
    this->start = std::chrono::steady_clock::now();
  }
  void Stop() final {
    DeviceAPI::Get(device_)->StreamSync(device_, nullptr);
    this->end = std::chrono::steady_clock::now();
  }
  int64_t SyncAndGetElapsedNanos() final {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(this->end - this->start).count();
  }
  static constexpr const char* _type_key = "SYCLTimerNode";
  TVM_DECLARE_FINAL_OBJECT_INFO(SYCLTimerNode, TimerNode);

 private:
  Device device_;
  std::chrono::steady_clock::time_point start, end;
};

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_SYCL_SYCL_COMMON_H_
