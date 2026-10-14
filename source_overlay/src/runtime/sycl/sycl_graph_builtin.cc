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
 * \file sycl_graph_builtin.cc
 * \brief SYCL command-graph builtins for the Relax virtual machine.
 *
 * Counterpart of src/runtime/vm/cuda/cuda_graph_builtin.cc. RewriteCUDAGraph lifts the static
 * parts of a Relax function into capture functions; the SYCL pipeline points those call sites
 * at the builtins below. The first call runs the capture function eagerly (warm-up), records
 * the same launches into a sycl_ext_oneapi_graph command graph and finalizes it; later calls
 * submit the executable graph, replacing one host submission per kernel by one per region.
 * Recording requires asynchronous dispatch (TVM_SYCL_SYNC_EACH_LAUNCH unset).
 */
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/function.h>
#include <tvm/runtime/vm/vm.h>

#include <cstdlib>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../support/utils.h"
#include "sycl_common.h"

namespace tvm {
namespace runtime {
namespace vm {

#ifdef SYCL_EXT_ONEAPI_GRAPH

namespace {

namespace sycl_exp = sycl::ext::oneapi::experimental;

struct SYCLGraphCaptureKey {
  int64_t index;
  ffi::Shape shape_expr;
  SYCLGraphCaptureKey(int64_t index, const Optional<ffi::Shape>& shape_expr) : index(index) {
    if (shape_expr) this->shape_expr = shape_expr.value();
  }
};

struct SYCLGraphCaptureKeyHash {
  size_t operator()(const SYCLGraphCaptureKey& key) const {
    std::hash<int64_t> hash_fn;
    size_t hash = hash_fn(key.index);
    for (const auto& shape : key.shape_expr) support::HashCombine(hash, hash_fn(shape));
    return hash;
  }
};

struct SYCLGraphCaptureKeyEqual {
  bool operator()(const SYCLGraphCaptureKey& lhs, const SYCLGraphCaptureKey& rhs) const {
    return lhs.index == rhs.index && std::equal(lhs.shape_expr.begin(), lhs.shape_expr.end(),
                                                rhs.shape_expr.begin(), rhs.shape_expr.end());
  }
};

struct SYCLGraphCapturedState {
  ObjectRef states;
  // The recorded graph is kept so that the executable graph can be finalized again.
  std::optional<sycl_exp::command_graph<sycl_exp::graph_state::modifiable>> graph;
  std::optional<sycl_exp::command_graph<sycl_exp::graph_state::executable>> exec;
  std::optional<sycl::event> last_run;
  bool replayed = false;  // used instead of last_run when the queue discards events
  int64_t replays_since_finalize = 0;
};

sycl::queue CurrentQueue() {
  auto* w = syclT::SYCLWorkspace::Global();
  return w->GetQueue(w->GetThreadEntry()->device);
}

/*!
 * \brief Replays after which an executable graph is finalized again (0: never).
 * DPC++ (646ab08) keeps the event of every replay of an executable graph and checks all of them
 * before the next replay, so the host cost of a replay grows by about 0.25 us per earlier replay
 * on the RTX 4090 (17 us over the first 100 replays, 500 us after 1,000). A fresh executable graph
 * starts with an empty list; re-finalizing every 32 replays (finalize: about 140 us for 26
 * kernels) bounds the cost at 10-13 us. The SYCL runtime of DPC++ v7.1.1 (libsycl 9) replays in a
 * constant 3.4-3.7 us, where re-finalizing only adds cost (5.4 us), so the default period is 0
 * from libsycl 9 on. TVM_SYCL_GRAPH_REFINALIZE sets the period.
 */
int64_t RefinalizePeriod() {
  static const int64_t period = [] {
    const char* v = std::getenv("TVM_SYCL_GRAPH_REFINALIZE");
    if (v != nullptr) return static_cast<int64_t>(std::atoll(v));
#if defined(__LIBSYCL_MAJOR_VERSION) && __LIBSYCL_MAJOR_VERSION >= 9
    return int64_t{0};
#else
    return int64_t{32};
#endif
  }();
  return period;
}

}  // namespace

/*! \brief The VM extension of SYCL command graphs. */
class SYCLGraphExtensionNode : public VMExtensionNode {
 public:
  TVM_DECLARE_FINAL_OBJECT_INFO(SYCLGraphExtensionNode, VMExtensionNode);

  ObjectRef RunOrCapture(VirtualMachine* vm, const ObjectRef& capture_func, ObjectRef args,
                         int64_t entry_index, Optional<ffi::Shape> shape_expr) {
    sycl::queue queue = CurrentQueue();
    // TVM_SYCL_GRAPH_MODE: record (default) | eager (never record; run the region every call)
    static const std::string mode = [] {
      const char* v = std::getenv("TVM_SYCL_GRAPH_MODE");
      return std::string(v ? v : "record");
    }();
    if (mode == "eager") {
      Array<ObjectRef> eager_args = Downcast<Array<ObjectRef>>(args);
      std::vector<AnyView> eager_packed(eager_args.size());
      for (size_t i = 0; i < eager_args.size(); ++i) eager_packed[i] = eager_args[i];
      ffi::Any rv;
      vm->InvokeClosurePacked(capture_func,
                              ffi::PackedArgs(eager_packed.data(), static_cast<int>(eager_packed.size())),
                              &rv);
      return rv.cast<ObjectRef>();
    }
    SYCLGraphCaptureKey entry_key{entry_index, shape_expr};
    if (auto it = capture_cache_.find(entry_key); it != capture_cache_.end()) {
      SYCLGraphCapturedState& entry = it->second;
      // This DPC++ version rejects re-submitting an executable graph while its previous run is
      // still in flight; callers normally synchronize between inferences, so this wait is free.
      if (syclT::DiscardEvents()) {
        // The queue returns no usable event; wait on the in-order queue instead.
        if (entry.replayed) SYCL_CALL(queue.wait_and_throw());
      } else if (entry.last_run) {
        SYCL_CALL(entry.last_run->wait_and_throw());
      }
      const int64_t period = RefinalizePeriod();
      if (period > 0 && entry.replays_since_finalize >= period) {
        entry.exec.reset();  // earlier replays have completed (waited above)
        SYCL_CALL(entry.exec.emplace(entry.graph->finalize()));
        entry.replays_since_finalize = 0;
      }
      if (syclT::DiscardEvents()) {
        SYCL_CALL(queue.ext_oneapi_graph(*entry.exec));
        entry.replayed = true;
      } else {
        SYCL_CALL(entry.last_run = queue.ext_oneapi_graph(*entry.exec));
      }
      ++entry.replays_since_finalize;
      return entry.states;
    }

    Array<ObjectRef> tuple_args = Downcast<Array<ObjectRef>>(args);
    int nargs = static_cast<int>(tuple_args.size());
    std::vector<AnyView> packed_args(nargs);
    for (int i = 0; i < nargs; ++i) packed_args[i] = tuple_args[i];

    ffi::Any capture_func_rv;
    // Warm-up run: executes the region eagerly (its results are the results of this call) and
    // performs one-time work such as JIT library loading before recording starts.
    vm->InvokeClosurePacked(capture_func, ffi::PackedArgs(packed_args.data(), nargs),
                            &capture_func_rv);

    // Recording must not overlap work that is still in flight on the queue (the warm-up run).
    SYCL_CALL(queue.wait_and_throw());
    sycl_exp::command_graph<sycl_exp::graph_state::modifiable> graph{queue.get_context(),
                                                                     queue.get_device()};
    graph.begin_recording(queue);
    try {
      vm->InvokeClosurePacked(capture_func, ffi::PackedArgs(packed_args.data(), nargs),
                              &capture_func_rv);
    } catch (...) {
      graph.end_recording(queue);
      throw;
    }
    graph.end_recording(queue);

    SYCLGraphCapturedState entry;
    entry.states = capture_func_rv.cast<ObjectRef>();
    SYCL_CALL(entry.exec.emplace(graph.finalize()));
    entry.graph.emplace(std::move(graph));
    ObjectRef states = entry.states;
    capture_cache_.emplace(entry_key, std::move(entry));
    return states;
  }

  ObjectRef GetCachedAllocation(VirtualMachine* vm, const ObjectRef& alloc_func,
                                int64_t entry_index) {
    if (auto it = alloc_cache_.find(entry_index); it != alloc_cache_.end()) return it->second;
    ffi::Any alloc_func_rv;
    vm->InvokeClosurePacked(alloc_func, ffi::PackedArgs(nullptr, 0), &alloc_func_rv);
    ObjectRef alloc_result = alloc_func_rv.cast<ObjectRef>();
    alloc_cache_[entry_index] = alloc_result;
    return alloc_result;
  }

  static constexpr const char* _type_key = "vm.SYCLGraphExtension";

 private:
  std::unordered_map<SYCLGraphCaptureKey, SYCLGraphCapturedState, SYCLGraphCaptureKeyHash,
                     SYCLGraphCaptureKeyEqual>
      capture_cache_;
  std::unordered_map<int64_t, ObjectRef> alloc_cache_;
};

class SYCLGraphExtension : public VMExtension {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(SYCLGraphExtension, VMExtension, SYCLGraphExtensionNode);
  static SYCLGraphExtension Create() {
    auto data_ = make_object<SYCLGraphExtensionNode>();
    return SYCLGraphExtension(std::move(data_));
  }
};

TVM_FFI_REGISTER_GLOBAL("vm.builtin.sycl_graph.run_or_capture")
    .set_body_packed([](ffi::PackedArgs args, ffi::Any* rv) {
      ICHECK(args.size() == 5 || args.size() == 4);
      VirtualMachine* vm = VirtualMachine::GetContextPtr(args[0]);
      auto extension = vm->GetOrCreateExtension<SYCLGraphExtension>();
      auto capture_func = args[1].cast<ObjectRef>();
      auto func_args = args[2].cast<ObjectRef>();
      int64_t entry_index = args[3].cast<int64_t>();
      Optional<ffi::Shape> shape_expr = std::nullopt;
      if (args.size() == 5) shape_expr = args[4].cast<ffi::Shape>();
      *rv = extension->RunOrCapture(vm, capture_func, func_args, entry_index, shape_expr);
    });

TVM_FFI_REGISTER_GLOBAL("vm.builtin.sycl_graph.get_cached_alloc")
    .set_body_packed([](ffi::PackedArgs args, ffi::Any* rv) {
      ICHECK_EQ(args.size(), 3);
      VirtualMachine* vm = VirtualMachine::GetContextPtr(args[0]);
      auto extension = vm->GetOrCreateExtension<SYCLGraphExtension>();
      auto alloc_func = args[1].cast<ObjectRef>();
      int64_t entry_index = args[2].cast<int64_t>();
      *rv = extension->GetCachedAllocation(vm, alloc_func, entry_index);
    });

#else  // SYCL_EXT_ONEAPI_GRAPH

// DPC++ builds without sycl_ext_oneapi_graph (e.g. the 2022 HIP build on the DCU): graph capture
// is unavailable; compile without relax.backend.use_cuda_graph.
TVM_FFI_REGISTER_GLOBAL("vm.builtin.sycl_graph.run_or_capture")
    .set_body_packed([](ffi::PackedArgs, ffi::Any*) {
      LOG(FATAL) << "This DPC++ build does not provide sycl_ext_oneapi_graph";
    });
TVM_FFI_REGISTER_GLOBAL("vm.builtin.sycl_graph.get_cached_alloc")
    .set_body_packed([](ffi::PackedArgs, ffi::Any*) {
      LOG(FATAL) << "This DPC++ build does not provide sycl_ext_oneapi_graph";
    });

#endif  // SYCL_EXT_ONEAPI_GRAPH

}  // namespace vm
}  // namespace runtime
}  // namespace tvm
