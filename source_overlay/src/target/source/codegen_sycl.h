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

/*! \file codegen_sycl.h
 *  \brief Generate SYCL device code.
 */
#ifndef TVM_TARGET_SOURCE_CODEGEN_SYCL_H_
#define TVM_TARGET_SOURCE_CODEGEN_SYCL_H_

#include <tvm/target/codegen.h>

#include <sstream>
#include <string>
#include <unordered_map>

#include "codegen_c.h"

namespace tvm {
namespace codegen {

class CodeGenSYCL final : public CodeGenC {
 public:
  CodeGenSYCL();
  std::string Finish() final;

  using CodeGenC::AddFunction;
  void AddFunction(const PrimFunc& f);
  void InitFuncState(const PrimFunc& f) final;
  // Kernel attributes: " [[sycl::reqd_sub_group_size(W)]]" when the kernel uses sub-group
  // shuffles, " [[intel::max_work_group_size(z, y, x)]]" for static thread extents.
  std::string WorkGroupSizeAttr(const PrimFunc& f);
  void SetSharedMemoryBudget(size_t bytes) { shared_memory_budget_ = bytes; }
  void SetWarpSize(int64_t warp_size) { warp_size_ = warp_size; }
  void SetMathWorkarounds(bool enabled) { math_workarounds_ = enabled; }
  void PrintFuncPrefix(std::ostream& os) final;                              // NOLINT(*)
  void BindThreadIndex(const IterVar& iv) final;                             // NOLINT(*)
  void PrintStorageScope(const std::string& scope, std::ostream& os) final;  // NOLINT(*)
  void PrintStorageSpace(const std::string& scope, std::ostream& os);        // NOLINT(*)
  void PrintStorageSync(const CallNode* op) final;                           // NOLINT(*)
  void PrintType(DataType t, std::ostream& os) final;                        // NOLINT(*)
  void PrintType(const Type& type, std::ostream& os) final;                  // NOLINT(*)
  std::string GetVecLoad(DataType t, const BufferNode* buffer, PrimExpr base) final;
  void PrintVecStore(const BufferNode* buffer, DataType t, PrimExpr base,
                     const std::string& value) final;  // NOLINT(*)
  void PrintVecElemLoad(const std::string& vec, DataType t, int i,
                        std::ostream& os) final;  // NOLINT(*)
  void PrintVecElemStore(const std::string& vec, DataType t, int i,
                         const std::string& value) final;
  void PrintVecElemLoadExpr(DataType t, int i, const std::string& value,
                            std::ostream& os) final;  // NOLINT(*)
  std::string CastFromTo(std::string value, DataType from, DataType target) final;
  std::string CastTo(std::string value, DataType target);
  void PrintVecAddr(const BufferNode* buffer, DataType t, PrimExpr base,
                    std::ostream& os);  // NOLINT(*)

  void VisitStmt_(const AllocateNode* op) final;                     // NOLINT(*)
  void VisitStmt_(const AttrStmtNode* op) final;                     // NOLINT(*)
  void VisitExpr_(const DivNode* op, std::ostream& os) final;        // NOLINT(*)
  void VisitExpr_(const CallNode* op, std::ostream& os) final;       // NOLINT(*)
  void VisitExpr_(const BroadcastNode* op, std::ostream& os) final;  // NOLINT(*)
  void VisitExpr_(const FloatImmNode* op, std::ostream& os) final;   // NOLINT(*)
  void VisitExpr_(const MinNode* op, std::ostream& os) final;        // NOLINT(*)
  void VisitExpr_(const MaxNode* op, std::ostream& os) final;        // NOLINT(*)
  void VisitExpr_(const AndNode* op, std::ostream& os) final;        // NOLINT(*)
  void VisitExpr_(const OrNode* op, std::ostream& os) final;         // NOLINT(*)
  void VisitExpr_(const SelectNode* op, std::ostream& os) final;     // NOLINT(*)
  void VisitExpr_(const RampNode* op, std::ostream& os) final;       // NOLINT(*)

 private:
  bool enable_fp16_{false};
  bool enable_fp64_{false};
  bool enable_atomics_{false};
  size_t current_shared_memory_usage_{0};
  size_t shared_memory_budget_{49152};
  // Target thread_warp_size; the sub-group size required by kernels that use shuffles.
  int64_t warp_size_{32};
  // float32 erf through float64 and a sign-stable tanh (HIP plugin defects); off behind CUDA.
  bool math_workarounds_{true};
  std::ostringstream mem_allocate_code;
  std::unordered_map<const Object*, size_t> allocation_size_;
};

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_TARGET_SOURCE_CODEGEN_SYCL_H_
