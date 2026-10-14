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
 * \file codegen_sycl.cc
 */
#include "codegen_sycl.h"

#include <tvm/tir/stmt_functor.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

#include "../../runtime/sycl/sycl_module.h"
#include "../../runtime/thread_storage_scope.h"
#include "../build_common.h"

namespace tvm {
namespace codegen {
// Shared-memory usage is tracked while emitting one kernel.
constexpr size_t MAX_SHARED_MEMORY_SIZE = 49152;

CodeGenSYCL::CodeGenSYCL() {
  // Set SYCL specific restrict keyword
  restrict_keyword_ = "";
}


void CodeGenSYCL::InitFuncState(const PrimFunc& f) {
  CodeGenC::InitFuncState(f);
  for (Var arg : f->params) {
    if (arg.dtype().is_handle()) {
      alloc_storage_scope_[arg.get()] = "global";
    }
  }
}

namespace {
// Collects the threadIdx.{x,y,z} extents of a kernel (1 when an axis is unbound) and whether it
// uses the sub-group shuffle builtins (intrin_rule_sycl.cc).
class ThreadExtentCollector : public tir::StmtExprVisitor {
 public:
  PrimExpr extent[3] = {1, 1, 1};
  bool uses_sub_group_shuffle = false;

 private:
  void VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == tir::attr::thread_extent) {
      if (const auto* iv = op->node.as<IterVarNode>()) {
        const std::string& tag = iv->thread_tag;
        if (tag.size() == 11 && tag.compare(0, 10, "threadIdx.") == 0) {
          extent[tag[10] - 'x'] = op->value;
        }
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitExpr_(const CallNode* op) final {
    if (op->op.same_as(builtin::call_pure_extern()) && !op->args.empty()) {
      if (const auto* name = op->args[0].as<StringImmNode>()) {
        if (std::string(name->value).rfind("tvm_sycl_shuffle", 0) == 0) uses_sub_group_shuffle = true;
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }
};
}  // namespace

std::string CodeGenSYCL::WorkGroupSizeAttr(const PrimFunc& f) {
  // The SYCL counterpart of the __launch_bounds__ that CodeGenCUDA prints. DPC++ lowers
  // intel::max_work_group_size to PTX .maxntid (the directive __launch_bounds__ produces), so
  // ptxas sizes the register budget for the real work-group instead of assuming no bound;
  // sycl::reqd_work_group_size is not lowered to .reqntid by the DPC++ versions we use.
  // TVM launches every kernel with exactly these thread extents.
  // TVM_SYCL_LAUNCH_BOUNDS=0 omits the attribute.
  ThreadExtentCollector collector;
  collector(f->body);
  std::ostringstream os;
  // Sub-group shuffles assume that the sub-group is exactly one TVM warp; require it, so a
  // target whose thread_warp_size differs from the device's sub-group size fails to build or
  // launch instead of reducing across the wrong lanes.
  if (collector.uses_sub_group_shuffle) {
    os << " [[sycl::reqd_sub_group_size(" << warp_size_ << ")]]";
  }
  const char* env = std::getenv("TVM_SYCL_LAUNCH_BOUNDS");
  if (env != nullptr && std::string(env) == "0") return os.str();
  int64_t ext[3];
  for (int d = 0; d < 3; ++d) {
    const auto* imm = collector.extent[d].as<IntImmNode>();
    if (imm == nullptr) return os.str();  // dynamic extent
    ext[d] = imm->value;
  }
  if (ext[0] * ext[1] * ext[2] <= 1) return os.str();
  // nd_range dimension 0 is threadIdx.z (the block range is reversed at launch).
  os << " [[intel::max_work_group_size(" << ext[2] << ", " << ext[1] << ", " << ext[0] << ")]]";
  return os.str();
}

void CodeGenSYCL::AddFunction(const PrimFunc& f) {
  // from CodeGenC -------------------
  // clear previous generated state.
  this->InitFuncState(f);
  // clear pre mem_allocate_code
  this->mem_allocate_code = std::ostringstream("");
  // reserve keywords
  ReserveKeywordsAsUnique();
  this->PrintExtraAttrs(f, stream);

  auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
  ICHECK(global_symbol.defined())
      << "CodeGenSYCL: Expect PrimFunc to have the global_symbol attribute";
  std::string global_symbol_str = static_cast<std::string>(global_symbol.value());
  uint32_t kernel_hash = static_cast<uint32_t>(std::hash<std::string>{}(global_symbol_str));
  std::string kernel_symbol = "h" + std::to_string(kernel_hash);
  bool no_alias = f->HasNonzeroAttr(tir::attr::kNoAlias);
  // Print the packed function
  stream << "// CodeGenSYCL: NOTE: Auto-generated packed function\n";
  stream << "// Function: " << global_symbol_str << "\n";
  stream << "class TVM_SYCL_KERNEL_NAME(" << kernel_symbol << ", TVM_SYCL_MODULE_ID);\n";
  PrintFuncPrefix(stream);
  stream << " TVM_SYCL_WRAPPER_NAME(" << kernel_symbol << ", TVM_SYCL_MODULE_ID)";
  stream << "(queue &Qh, sycl::range<3> dimGrid, sycl::range<3> dimBlock, void** void_args) {\n";
  int func_scope = this->BeginScope();
  // Print the packed function's void_args
  for (size_t i = 0; i < f->params.size(); ++i) {
    tir::Var v = f->params[i];
    std::string vid = AllocVarID(v.get());
    if (v.dtype().is_handle()) {
      // Register handle data type
      // TODO(tvm-team): consider simply keep type info in the
      // type annotation(via a normalizing rewriting).
      if (auto* ptr = v->type_annotation.as<PointerTypeNode>()) {
        if (auto* prim = ptr->element_type.as<PrimTypeNode>()) {
          RegisterHandleType(v.get(), prim->dtype);
        }
      }
    }
    this->PrintIndent();
    PrintType(GetType(v), stream);
    if (no_alias) {
      PrintRestrict(v, stream);
    }
    stream << ' ' << vid << " = (";
    PrintType(GetType(v), stream);
    stream << ")(*(int64_t *)(void_args[" << i << "]));\n";
  }

  // Submit the kernel through TVM_SYCL_LAUNCH (defined in the prelude of BuildSYCL): a
  // handler-free, event-free submission where the DPC++ headers provide one, else
  // queue::submit with handler::parallel_for.
  this->PrintIndent();
  stream << "TVM_SYCL_LAUNCH(Qh, TVM_SYCL_KERNEL_NAME(" << kernel_symbol
         << ", TVM_SYCL_MODULE_ID), dimGrid, dimBlock)\n";
  int func_scope2 = this->BeginScope();
  this->PrintIndent();
  stream << "[=](sycl::nd_item<3> item) [[intel::kernel_args_restrict]]" << WorkGroupSizeAttr(f)
         << " {\n";
  int func_scope3 = this->BeginScope();
  // Function code
  this->PrintStmt(f->body);
  this->EndScope(func_scope3);
  this->PrintIndent();
  stream << "}\n";
  this->EndScope(func_scope2);
  this->PrintIndent();
  stream << "TVM_SYCL_LAUNCH_END\n";
  this->EndScope(func_scope);
  stream << "}\n";
}

void CodeGenSYCL::PrintFuncPrefix(std::ostream& os) {
  os << "#ifdef __cplusplus\n"
         << "extern \"C\"\n"
         << "#endif\n"
         << "void";
}

std::string CodeGenSYCL::Finish() {
  // fp16 and fp64
  if (enable_fp16_) {
  }

  if (enable_fp64_) {
  }

  // atomic_add
  if (enable_atomics_) {

  }
  return CodeGenC::Finish();
}

void CodeGenSYCL::BindThreadIndex(const IterVar& iv) {
  ICHECK(!var_idmap_.count(iv->var.get()));
  runtime::ThreadScope ts = runtime::ThreadScope::Create(iv->thread_tag);
  std::ostringstream os;
  // SYCL's rightmost nd_range dimension varies fastest and maps to the backend's x axis, so
  // both threadIdx.d and blockIdx.d use SYCL dimension 2 - d. (The original mapping kept
  // blockIdx.x on SYCL dimension 0, i.e. CUDA grid z, which limits it to 65,535 work-groups
  // on NVIDIA; the runtime builds the grid range in the same reversed order.)
  // The indices are read as 32-bit values (TVM_SYCL_LOCAL_ID/GROUP_ID, see BuildSYCL).
  if (ts.rank == 1) {
    os << "TVM_SYCL_LOCAL_ID(" << 2 - ts.dim_index << ")";
  } else {
    os << "TVM_SYCL_GROUP_ID(" << 2 - ts.dim_index << ")";
  }
  var_idmap_[iv->var.get()] = CastFromTo(os.str(), DataType::Int(32), iv->var.dtype());
}

// This path targets Linux/LP64, where long and unsigned long are 64-bit.
static_assert(sizeof(long) == 8, "This code assumes LP64 (Linux), where long is 64-bit.");
static_assert(sizeof(unsigned long) == 8, "This code assumes LP64 (Linux), where unsigned long is 64-bit.");

void CodeGenSYCL::PrintType(DataType t, std::ostream& os) {  // NOLINT(*)
  const int lanes = t.lanes();

  // Emit a scalar or sycl::vec<T, N> by lane count; unusual widths use a pointer fallback.
  auto print_vec = [&](const char* base) {
    if (lanes == 1) {
      os << base;
    } else if ((lanes >= 2 && lanes <= 4) || lanes == 8 || lanes == 16) {
      os << "sycl::vec<" << base << ", " << lanes << ">";
    } else {
      // Preserve the existing pointer fallback for widths other than 2/3/4/8/16.
      os << base << "*";
    }
  };

  // Handle, void, and bool.
  if (t.is_handle()) {
    ICHECK_EQ(lanes, 1) << "do not yet support vector types for handle";
    os << "void*";
    return;
  }
  if (t.is_void()) {
    os << "void";
    return;
  }
  if (t == DataType::Bool()) {
    os << "bool";
    return;
  }

  // Floating-point types.
  if (t.is_float()) {
    switch (t.bits()) {
      case 16:
        enable_fp16_ = true;
        // Use the SYCL 2020 sycl::half type.
        print_vec("sycl::half");
        return;
      case 32:
        print_vec("float");
        return;
      case 64:
        enable_fp64_ = true;
        print_vec("double");
        return;
      default:
        LOG(FATAL) << "Cannot convert float type " << t << " to SYCL type";
    }
  }

  // Signed integer types.
  if (t.is_int()) {
    switch (t.bits()) {
      case 8:
        print_vec("std::int8_t");
        return;
      case 16:
        print_vec("std::int16_t");
        return;
      case 32:
        print_vec("int");
        return;
      case 64:
        // Linux/LP64: long is 64-bit.
        print_vec("long");
        return;
      default:
        LOG(FATAL) << "Cannot convert int type " << t << " to SYCL type";
    }
  }

  // Unsigned integer types.
  if (t.is_uint()) {
    switch (t.bits()) {
      case 8:
        print_vec("std::uint8_t");
        return;
      case 16:
        print_vec("std::uint16_t");
        return;
      case 32:
        print_vec("unsigned int");
        return;
      case 64:
        // Linux/LP64: unsigned long is 64-bit.
        print_vec("unsigned long");
        return;
      default:
        LOG(FATAL) << "Cannot convert uint type " << t << " to SYCL type";
    }
  }

  LOG(FATAL) << "Cannot convert type " << t << " to SYCL type";
}


void CodeGenSYCL::PrintType(const Type& type, std::ostream& os) {  // NOLINT(*)
  if (auto* ptr = type.as<PrimTypeNode>()) {
    return PrintType(ptr->dtype, os);
  } else if (auto* ptr = type.as<PointerTypeNode>()) {
    PrintType(ptr->element_type, os);
    os << '*';
  } else if (IsVoidType(type)) {
    os << "void";
  } else {
    LOG(FATAL) << "Type " << type << " does not have a corresponding C Type";
  }
}

void CodeGenSYCL::PrintVecAddr(const BufferNode* buffer, DataType t, PrimExpr base,
                                 std::ostream& os) {  // NOLINT(*)
  const VarNode* buffer_var = buffer->data.get();
  if (!HandleTypeMatch(buffer_var, t.element_of())) {
    os << '(';
    PrintType(t.element_of(), os);
    os << "*)";
  }
  os << GetVarID(buffer_var) << " + ";
  PrintExpr(base, os);
}

void CodeGenSYCL::PrintStorageSpace(const std::string& scope, std::ostream& os) {  // NOLINT(*)
  os << "sycl::access::address_space::generic_space";
  /*
  if (scope == "global") {
    os << "sycl::access::address_space::global_space";
  } else if (scope == "shared") {
    os << "sycl::access::address_space::local_space";
  } else {
    os << "sycl::access::address_space::private_space";
  }
  */
}

// ---------------- GetVecLoad ----------------
std::string CodeGenSYCL::GetVecLoad(DataType t,const BufferNode* buf,PrimExpr base){
  if(!(t.lanes()==2||t.lanes()==4||t.lanes()==8||t.lanes()==16)) {
    // sycl::vec<T, 3> has the size and alignment of sycl::vec<T, 4>, unlike CUDA's float3, so a
    // cast access at a stride-3 address is misaligned and touches a fourth element. Load the
    // lanes one by one instead.
    std::ostringstream os;
    os << "({ ";
    PrintType(t.element_of(), os);
    os << "* _p_ = ";
    PrintVecAddr(buf, t, base, os);
    os << "; ";
    PrintType(t, os);
    os << "(";
    for (int i = 0; i < t.lanes(); ++i) os << (i ? ", " : "") << "_p_[" << i << "]";
    os << "); })";
    return os.str();
  }

  std::ostringstream os;
  os<<"({\n"; this->PrintIndent();
  os<<"using vec_t = sycl::vec<"; PrintType(t.element_of(),os);
  os<<","<<t.lanes()<<">;\n";

  this->PrintIndent(); os<<"auto _raw_ = "; PrintVecAddr(buf,t,base,os); os<<";\n";
  this->PrintIndent(); os<<"auto _raw_vec_ = reinterpret_cast<vec_t*>(_raw_);\n";

  auto it = alloc_storage_scope_.find(buf->data.get());
  std::string scope = (it!=alloc_storage_scope_.end())? it->second : "global";
  this->PrintIndent();
  if (scope == "local") {
    // Private array: keep the generic pointer (casting it to the global space is invalid).
    os << "vec_t* _vptr_ = _raw_vec_;\n";
  } else {
    const char* space = (scope == "shared") ? "local_space" : "global_space";
    os << "auto _vptr_ = TVM_SYCL_AS_CAST(" << space << ", _raw_vec_);\n";
  }

  this->PrintIndent(); os<<"vec_t _v_ = *_vptr_;\n";
  this->PrintIndent(); os<<"_v_; })";
  return os.str();
}

// ---------------- PrintVecStore ----------------
void CodeGenSYCL::PrintVecStore(const BufferNode* buf,DataType t,PrimExpr base,
                                const std::string& value){
  if(!(t.lanes()==2||t.lanes()==4||t.lanes()==8||t.lanes()==16)){
    // Element-wise store for widths whose sycl::vec layout differs from the buffer's (see
    // GetVecLoad).
    this->PrintIndent();
    stream << "{ ";
    PrintType(t.element_of(), stream);
    stream << "* _p_ = ";
    PrintVecAddr(buf, t, base, stream);
    stream << "; ";
    PrintType(t, stream);
    stream << " _v_ = " << value << ";";
    for (int i = 0; i < t.lanes(); ++i) stream << " _p_[" << i << "] = _v_[" << i << "];";
    stream << " }\n";
    return;
  }
  this->PrintIndent();
  stream<<"{ using vec_t = sycl::vec<"; PrintType(t.element_of(),stream);
  stream<<","<<t.lanes()<<">;\n";

  this->PrintIndent(); stream<<"auto _raw_ = "; PrintVecAddr(buf,t,base,stream);
  stream<<";\n";
  this->PrintIndent(); stream<<"auto _raw_vec_ = reinterpret_cast<vec_t*>(_raw_);\n";

  auto it = alloc_storage_scope_.find(buf->data.get());
  std::string scope = (it!=alloc_storage_scope_.end())? it->second : "global";
  this->PrintIndent();
  if (scope == "local") {
    // Private array: keep the generic pointer (casting it to the global space is invalid).
    stream << "vec_t* _vptr_ = _raw_vec_;\n";
  } else {
    const char* space = (scope == "shared") ? "local_space" : "global_space";
    stream << "auto _vptr_ = TVM_SYCL_AS_CAST(" << space << ", _raw_vec_);\n";
  }

  this->PrintIndent();
  stream<<"*_vptr_ = static_cast<vec_t>("<<value<<"); }\n";
}


void CodeGenSYCL::PrintStorageSync(const CallNode* op) {
  const std::string& sync = op->args[0].as<StringImmNode>()->value;
  if (sync == "warp") {
    // LowerThreadAllreduce no longer emits warp-level syncs for SYCL. Any other producer would
    // need a sub-group barrier at a position that may be divergent, which SYCL does not allow,
    // so reject the schedule at compile time instead of emitting a silent no-op. The legacy
    // no-op is kept only behind TVM_SYCL_LEGACY_WARP_SYNC=1 to reproduce earlier results.
    const char* legacy = std::getenv("TVM_SYCL_LEGACY_WARP_SYNC");
    ICHECK(legacy != nullptr && std::string(legacy) == "1")
        << "CodeGenSYCL: storage_sync(\"warp\") is not supported by the SYCL backend";
    this->PrintIndent();
    this->stream << "/* warp storage_sync: no-op on SYCL to avoid divergence deadlock */\n";
  } else if (sync == "shared") {
    this->PrintIndent();
    this->stream << "group_barrier(item.get_group());\n";
  } else if (sync == "global") {
    LOG(FATAL) << "global barrier is not supported";
  }
}


void CodeGenSYCL::PrintStorageScope(const std::string& scope, std::ostream& os) {  // NOLINT(*)
  // SYCL storage scopes are materialized at allocation/use sites.  Keep this
  // hook as a no-op; inspecting the current stream with substr() is unsafe when
  // the stream is shorter than the requested suffix and can crash codegen for
  // schedules that use shared/local cache stages.
}

std::string CodeGenSYCL::CastFromTo(std::string value, DataType from, DataType target) {
  if (from == target) return value;
  return CastTo(value, target);
}

std::string CodeGenSYCL::CastTo(std::string value, DataType target) {
  std::ostringstream os;
  if (target.lanes() == 1) {
    os << "((";
    this->PrintType(target, os);
    os << ")" << value << ")";
  } else {  // convert vector type
    os << "(";
    os << value << ".convert<";
    PrintType(target.element_of(), os);
    os << ">()";
    os << ")";
  }
  return os.str();
}

void CodeGenSYCL::VisitStmt_(const AttrStmtNode* op) {
  // Software pipelines from InjectSoftwarePipeline. Without tir.use_async_copy the copies inside
  // an async stage are ordinary loads and stores, so the commit/wait markers have nothing to
  // wait for (on CUDA they become cp.async.commit_group / wait_group with no cp.async in
  // flight). The stage bodies are emitted unchanged; the pipeline's work-group barriers come
  // from storage_sync as usual.
  if (op->attr_key == tir::attr::async_commit_queue_scope ||
      op->attr_key == tir::attr::async_scope) {
    this->VisitStmt(op->body);
    return;
  }
  if (op->attr_key == tir::attr::async_wait_queue_scope) {
    const auto* inner = op->body.as<AttrStmtNode>();
    ICHECK(inner && inner->attr_key == tir::attr::async_wait_inflight_count);
    this->VisitStmt(inner->body);
    return;
  }
  CodeGenC::VisitStmt_(op);
}

void CodeGenSYCL::VisitStmt_(const AllocateNode* op) {
  allocation_size_.insert({op->buffer_var.get(), op->ConstantAllocationSize() * op->dtype.lanes()});
  ICHECK(!is_zero(op->condition));
  std::string vid = AllocVarID(op->buffer_var.get());

  size_t constant_size = op->ConstantAllocationSize();
  ICHECK_GT(constant_size, 0) << "Can only handle constant size stack allocation for now";

  auto scope = GetPtrStorageScope(op->buffer_var);
  alloc_storage_scope_[op->buffer_var.get()] = scope;
  ICHECK_NE(scope, "global") << "Cannot allocate global memory when targeting SYCL.";
  
  if (scope == "shared") {
    size_t bytes_per_elem = (op->dtype.bits() / 8) * op->dtype.lanes();
    size_t required_bytes = constant_size * bytes_per_elem;
  
    // Reject kernels whose group-local allocations exceed the budget. (An earlier version
    // shrank the allocation and only warned, which turns an invalid schedule into
    // out-of-bounds group-local accesses.)
    ICHECK_LE(current_shared_memory_usage_ + required_bytes, shared_memory_budget_)
        << "CodeGenSYCL: group-local allocation of " << required_bytes << " bytes exceeds the "
        << shared_memory_budget_ << "-byte budget (" << current_shared_memory_usage_
        << " bytes already allocated in this kernel)";
  
    current_shared_memory_usage_ += required_bytes;
  
    // group_local_memory aligns to alignof(T); a plain float[N] is only 4-byte aligned, so vector
    // accesses (sycl::vec, up to 64 bytes) into a later buffer could be misaligned. Allocate an
    // over-aligned wrapper and use its array.
    // group_local_memory<T>(g) value-initialises T: work-item 0 zero-fills the whole buffer and
    // the work-group then waits at a barrier, at the start of every work-group. TIR never reads
    // shared memory before writing it, so use the _for_overwrite form, which for a trivial T
    // neither initialises nor synchronises. TVM_SYCL_LOCAL_ZERO_INIT=1 restores the old form.
    const char* zero_init = std::getenv("TVM_SYCL_LOCAL_ZERO_INIT");
    const bool for_overwrite = !(zero_init != nullptr && std::string(zero_init) == "1");
    this->PrintIndent();
    stream << "auto " << vid << " = (*sycl::ext::oneapi::group_local_memory"
           << (for_overwrite ? "_for_overwrite" : "") << "<tvm_sycl_aligned_array<";
    PrintType(op->dtype, stream);
    stream << ", " << constant_size << ">>(item.get_group())).data;\n";
  } else {
    this->PrintIndent();
    stream << "alignas(64) ";
    PrintType(op->dtype, stream);
    stream << ' ' << vid << '[' << constant_size << "];\n";
  }
  RegisterHandleType(op->buffer_var.get(), op->dtype);
  this->PrintStmt(op->body);
}

void CodeGenSYCL::VisitExpr_(const DivNode* op, std::ostream& os) {
  const auto* denominator = op->b.as<CallNode>();
  if (op->dtype.is_float() && op->dtype.bits() == 32 && denominator != nullptr &&
      denominator->op.same_as(builtin_call_pure_extern_) &&
      denominator->args.size() == 2) {
    const auto func = Downcast<StringImm>(denominator->args[0]);
    if (func->value == "sqrt" || func->value == "sqrtf") {
      os << '(';
      PrintExpr(op->a, os);
      os << " * sycl::rsqrt(";
      PrintExpr(denominator->args[1], os);
      os << "))";
      return;
    }
  }
  CodeGenC::VisitExpr_(op, os);
}

void CodeGenSYCL::VisitExpr_(const CallNode* op, std::ostream& os) {
  if (op->op.same_as(builtin::address_of())) {
    // Overload tvm_address_of to add storage scope (e.g. __global).
    const BufferLoadNode* load = op->args[0].as<BufferLoadNode>();
    ICHECK(op->args.size() == 1 && load);
    ICHECK_EQ(load->indices.size(), 1) << "CodeGenSYCL only supports flat memory allocations.";
    os << "((";
    this->PrintType(load->dtype.element_of(), os);
    os << " *)" << this->GetVarID(load->buffer->data.get()) << " + ";
    this->PrintExpr(load->indices[0], os);
    os << ')';
  } else if (op->op.same_as(builtin_call_extern_)) {
    auto func = Downcast<StringImm>(op->args[0]);
    // Enable atomics extension if used.
    if (func->value == "atomic_add") {
      enable_atomics_ = true;
    }
    CodeGenC::VisitExpr_(op, os);
  } else if (op->op.same_as(builtin_call_pure_extern_)) {
    ICHECK_GE(op->args.size(), 1U);
    auto func = Downcast<StringImm>(op->args[0]);
    // Workarounds for the HIP plugin's float32 erf/tanh (see the overlay README, M1). Behind the CUDA
    // plugin libdevice's float32 erf/tanh are accurate, and float64 runs at a small fraction of
    // float32 throughput on consumer GPUs (GELU kernels were 7.5x slower), so they are skipped there.
    if (math_workarounds_ && (func->value == "erf" || func->value == "erff") &&
        op->dtype.is_float() && op->dtype.bits() == 32 && op->args.size() == 2) {
      const std::string x = PrintExpr(op->args[1]);
      os << "((float)sycl::erf((double)(" << x << ")))";
      return;
    }
    if (math_workarounds_ && (func->value == "tanh" || func->value == "tanhf") &&
        op->dtype.is_float() && op->dtype.bits() == 32 && op->args.size() == 2) {
      const std::string x = PrintExpr(op->args[1]);
      const std::string positive_exp = "exp(-2.000000e+00f * (" + x + "))";
      const std::string negative_exp = "exp(2.000000e+00f * (" + x + "))";
      os << "select(((" << negative_exp << " - 1.000000e+00f) / (" << negative_exp
         << " + 1.000000e+00f)), ((1.000000e+00f - " << positive_exp
         << ") / (1.000000e+00f + " << positive_exp << ")), (0.000000e+00f <= " << x
         << "))";
      return;
    }
    if ((func->value == "erf" || func->value == "erff" || func->value == "tanh" ||
         func->value == "tanhf") &&
        op->dtype.is_float() && op->args.size() == 2) {
      // Explicit SYCL overloads (avoids ::erf(double) / C erff resolution).
      const bool is_erf = func->value == "erf" || func->value == "erff";
      os << (is_erf ? "sycl::erf(" : "sycl::tanh(") << PrintExpr(op->args[1]) << ')';
      return;
    }
    // default generated math function name is non-exist in SYCL. Not know where the function names are generated ?
    std::unordered_map<std::string, std::string> func_names = {{"expf", "exp"}, {"powf", "pow"}, {"tanhf", "tanh"}};
    if(func_names.find(func->value) != func_names.end()){
      func = StringImm(func_names[func->value]);
    }
    this->PrintCallExtern(GetType(GetRef<PrimExpr>(op)), func->value, op->args, true, os);
  } else {
    CodeGenC::VisitExpr_(op, os);
  }
}

void CodeGenSYCL::VisitExpr_(const BroadcastNode* op, std::ostream& os) {  // NOLINT(*)
  std::string v = PrintExpr(op->value);
  os << "(";
  os << "vec<";
  PrintType(op->dtype.element_of(), os);
  os << ", " << op->lanes << ">";
  os << "{";
  const auto* lanes_imm = op->lanes.as<IntImmNode>();
  ICHECK(lanes_imm) << "Ramp lanes must be IntImm";
  int lanes = static_cast<int>(lanes_imm->value);
  for (int i = 0; i < lanes; i++) {
    if (i != 0) os << ", ";
    os << v;
  }
  os << "})";
}

void CodeGenSYCL::VisitExpr_(const FloatImmNode* op, std::ostream& os) {  // NOLINT(*)
  if (std::isinf(op->value)) {
    if (op->value < 0) {
      os << "-";
    }
    os << "INFINITY";
  } else if (std::isnan(op->value)) {
    os << "NAN";
  } else {
    CodeGenC::VisitExpr_(op, os);
  }
}

template <typename T>
inline void PrintBinaryExpr(const T* op, const char* opstr, std::ostream& os, CodeGenSYCL* p) {
  if (op->dtype.lanes() == 1) {
    os << opstr << "((";
    p->PrintType(op->a->dtype, os);
    os << ")";
    p->PrintExpr(op->a, os);
    os << ", (";
    p->PrintType(op->b->dtype, os);
    os << ")";
    p->PrintExpr(op->b, os);
    os << ')';
  } else {
    p->PrintVecBinaryOp(opstr, op->dtype, op->a, op->b, os);
  }
}

// Floating-point min/max as sycl::fmin/fmax, the counterparts of CUDA's fminf/fmaxf (one
// min/max instruction, NaN operands ignored). sycl::min/max are the "common" functions, which DPC++
// lowers to a compare and a select; integers keep them. TVM_SYCL_FMINMAX=0 restores min/max.
static bool UseFMinMax(DataType t) {
  static const bool enabled = [] {
    const char* v = std::getenv("TVM_SYCL_FMINMAX");
    return !(v != nullptr && std::string(v) == "0");
  }();
  return enabled && t.is_float() && (t.bits() == 16 || t.bits() == 32 || t.bits() == 64);
}

void CodeGenSYCL::VisitExpr_(const MinNode* op, std::ostream& os) {
  PrintBinaryExpr(op, UseFMinMax(op->dtype) ? "sycl::fmin" : "min", os, this);
}

void CodeGenSYCL::VisitExpr_(const MaxNode* op, std::ostream& os) {
  PrintBinaryExpr(op, UseFMinMax(op->dtype) ? "sycl::fmax" : "max", os, this);
}

void CodeGenSYCL::VisitExpr_(const AndNode* op, std::ostream& os) {
  std::ostringstream oss;
  os << "(";
  this->PrintExpr(op->a, oss);
  os << CastTo(oss.str(), op->dtype);
  oss.str("");
  os << " && ";
  this->PrintExpr(op->b, oss);
  os << CastTo(oss.str(), op->dtype);
  os << ")";
}

void CodeGenSYCL::VisitExpr_(const OrNode* op, std::ostream& os) {
  std::ostringstream oss;
  os << "(";
  this->PrintExpr(op->a, oss);
  os << CastTo(oss.str(), op->dtype);
  oss.str("");
  os << " || ";
  this->PrintExpr(op->b, oss);
  os << CastTo(oss.str(), op->dtype);
  os << ")";
}

void CodeGenSYCL::VisitExpr_(const SelectNode* op, std::ostream& os) {
  std::ostringstream oss;
  os << "select(";
  PrintExpr(op->false_value, oss);
  os << CastFromTo(oss.str(), op->false_value.dtype(), op->dtype);
  oss.str("");
  os << ", ";
  PrintExpr(op->true_value, oss);
  os << CastFromTo(oss.str(), op->true_value.dtype(), op->dtype);
  oss.str("");
  os << ", ";
  PrintExpr(op->condition, oss);
  if (op->dtype.is_float()) {
    if (op->condition.dtype().is_uint() || op->condition.dtype().is_int()) {
      os << oss.str();
    } else {
      os << CastTo(oss.str(), DataType::Int(op->dtype.bits(), op->dtype.lanes()));
    }
  } else {
    os << CastFromTo(oss.str(), op->condition.dtype(), op->dtype);
  }
  os << ")";
}

void CodeGenSYCL::PrintVecElemLoad(const std::string& vec, DataType t, int i,
                                std::ostream& os) {  // NOLINT(*)
  os << vec << "[" << i << "]" << std::dec;

}

void CodeGenSYCL::PrintVecElemStore(const std::string& vec, DataType t, int i,
                                 const std::string& value) {
  this->PrintIndent();
  stream << vec << "[" << i << "]" << " = " << value << ";\n" << std::dec;

}

void CodeGenSYCL::PrintVecElemLoadExpr(DataType t, int i, const std::string& value, std::ostream& os) {
  ICHECK_GT(t.lanes(), 1);
  /*
  if (t.bits() == 8 && (t.is_int() || t.is_uint())) {
    if (i != 0) {
      os << "|";
    }
    os << "((0x000000ff << " << i * 8 << ") & (" << value << " << " << i * 8 << "))";
    return;
  }
  */

  if (i == 0) {
    os << "((";
    PrintType(t, os);
    os << "){";
  }
  os << value;
  if (i != t.lanes() - 1) {
    os << ",";
  } else {
    os << "})";
  }
  return;
}


void CodeGenSYCL::VisitExpr_(const RampNode* op, std::ostream& os) {  // NOLINT(*)
  // constraint of current logic
  ICHECK_EQ(op->base.dtype(), DataType::Int(32));

  const auto* lanes_imm = op->lanes.as<IntImmNode>();
  ICHECK(lanes_imm) << "Ramp lanes must be IntImm";
  int lanes = static_cast<int>(lanes_imm->value);

  os << "((int" << lanes << "){";
  for (int i = 0; i < lanes; ++i) {
    os << "(" << PrintExpr(op->base) << ")"
       << "+(" << PrintExpr(op->stride) << "*" << i << ")";
    if (i != lanes - 1) os << ", ";
  }
  os << "})";
}


runtime::Module BuildSYCL(IRModule mod, Target target) {
  bool output_ssa = false;
  std::stringstream code;
  code << "// tvm target: " << target->str() << "\n";
  // Kernel submission. DPC++ v7 submits sycl_khr_free_function_commands launches (and queue
  // shortcuts) without a handler, and launch_grouped without an event: 2.5 us of host time per
  // launch on the RTX 4090 against 4.0 us through queue::submit (CUDA: 2.2 us). DPC++ v7 still
  // marks the extension unfinished, so it is enabled before the SYCL header is included. Older
  // DPC++ (646ab08, the 2022 HIP build) has no such path and keeps queue::submit;
  // TVM_SYCL_DIRECT_SUBMIT=0 keeps it everywhere.
  const char* direct_env = std::getenv("TVM_SYCL_DIRECT_SUBMIT");
  if (direct_env != nullptr && std::string(direct_env) == "0") {
    code << "#define TVM_SYCL_HANDLER_SUBMIT\n";
  }
  code << "#if !defined(TVM_SYCL_HANDLER_SUBMIT) && !defined(__DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS)\n"
          "#define __DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS\n"
          "#endif\n";
  code << "#include <sycl/sycl.hpp>\n";
  code << "using namespace sycl;\n";
  code << "#if !defined(TVM_SYCL_HANDLER_SUBMIT) && defined(SYCL_KHR_FREE_FUNCTION_COMMANDS)\n"
          "#define TVM_SYCL_LAUNCH(q, name, grid, block) "
          "sycl::khr::launch_grouped(q, (grid) * (block), (block),\n"
          "#define TVM_SYCL_LAUNCH_END );\n"
          "#else\n"
          "#define TVM_SYCL_LAUNCH(q, name, grid, block) q.submit([=](sycl::handler& h) { "
          "h.parallel_for<name>(sycl::nd_range<3>((grid) * (block), (block)),\n"
          "#define TVM_SYCL_LAUNCH_END ); });\n"
          "#endif\n";
  code << "#ifndef TVM_SYCL_MODULE_ID\n";
  code << "#define TVM_SYCL_MODULE_ID m0\n";
  code << "#endif\n";
  code << "#define TVM_SYCL_KERNEL_NAME_IMPL(func, module_id) func##_##module_id\n";
  code << "#define TVM_SYCL_KERNEL_NAME(func, module_id) "
          "TVM_SYCL_KERNEL_NAME_IMPL(func, module_id)\n";
  code << "#define TVM_SYCL_WRAPPER_NAME_IMPL(func, module_id) w_##func##_##module_id\n";
  code << "#define TVM_SYCL_WRAPPER_NAME(func, module_id) "
          "TVM_SYCL_WRAPPER_NAME_IMPL(func, module_id)\n";
  code << "template <typename T, std::size_t N> struct alignas(64) tvm_sycl_aligned_array "
          "{ T data[N]; };\n";
  // Address-space qualification of vector accesses. DPC++ builds that predate SYCL 2020's
  // address_space_cast (e.g. the 2022 HIP build on the DCU) define
  // TVM_SYCL_NO_ADDRESS_SPACE_CAST in SYCL_FLAGS and use the generic pointer.
  code << "#ifndef TVM_SYCL_NO_ADDRESS_SPACE_CAST\n"
          "#define TVM_SYCL_AS_CAST(space, p) sycl::address_space_cast<"
          "sycl::access::address_space::space, sycl::access::decorated::no>(p)\n"
          "#else\n"
          "#define TVM_SYCL_AS_CAST(space, p) (p)\n"
          "#endif\n";
  // Work-item and work-group indices as int. nd_item returns size_t; behind the CUDA plugin
  // that is the sign-extended PTX register, and LLVM then evaluates `(int)id * c + k` index
  // expressions in 64 bits (shl/ashr pairs, one live 64-bit address per shared-memory load,
  // twice CUDA's registers in DLight matmuls). Opt-in (TVM_SYCL_NATIVE_IDS=1): NVPTX device code
  // reads the 32-bit special registers directly, as CUDA's threadIdx/blockIdx do (SYCL dimension
  // 2 is the x axis), masked to 31 bits so that `/` and `%` by constants stay unsigned (DPC++
  // optimizes before the NVPTX backend attaches the registers' value ranges); "hw" masks to the
  // hardware ranges, which lets LLVM narrow index arithmetic to 16 bits. On the RTX 4090 this
  // brings DLight's 4096^3 matmul and ResNet stage-1 convolution to TVM-CUDA's time (1.00x and
  // 1.04x, from 1.07x and 1.21x) and DLight ResNet-18 3% closer, but makes ResNet-18 schedules
  // tuned with nd_item indices 5-24% slower (residual-add convolutions up to 1.9x), and a retune
  // with it does not recover that (0.571 vs 0.530 ms), so the default keeps nd_item's indices.
  const char* ids_env = std::getenv("TVM_SYCL_NATIVE_IDS");
  const std::string ids_mode = ids_env != nullptr ? ids_env : "0";
  if (ids_mode != "1" && ids_mode != "hw") {
    code << "#define TVM_SYCL_PORTABLE_IDS\n";
  }
  const bool hw_ranges = ids_mode == "hw";
  code << "#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__) && !defined(TVM_SYCL_PORTABLE_IDS)\n"
       << "#define TVM_SYCL_LOCAL_ID_2 ((int)(__nvvm_read_ptx_sreg_tid_x() & "
       << (hw_ranges ? "0x3ff" : "0x7fffffff") << "))\n"
       << "#define TVM_SYCL_LOCAL_ID_1 ((int)(__nvvm_read_ptx_sreg_tid_y() & "
       << (hw_ranges ? "0x3ff" : "0x7fffffff") << "))\n"
       << "#define TVM_SYCL_LOCAL_ID_0 ((int)(__nvvm_read_ptx_sreg_tid_z() & "
       << (hw_ranges ? "0x3f" : "0x7fffffff") << "))\n"
       << "#define TVM_SYCL_GROUP_ID_2 ((int)(__nvvm_read_ptx_sreg_ctaid_x() & 0x7fffffff))\n"
       << "#define TVM_SYCL_GROUP_ID_1 ((int)(__nvvm_read_ptx_sreg_ctaid_y() & "
       << (hw_ranges ? "0xffff" : "0x7fffffff") << "))\n"
       << "#define TVM_SYCL_GROUP_ID_0 ((int)(__nvvm_read_ptx_sreg_ctaid_z() & "
       << (hw_ranges ? "0xffff" : "0x7fffffff") << "))\n"
          "#define TVM_SYCL_LOCAL_ID(d) TVM_SYCL_LOCAL_ID_##d\n"
          "#define TVM_SYCL_GROUP_ID(d) TVM_SYCL_GROUP_ID_##d\n"
          "#else\n"
          "#define TVM_SYCL_LOCAL_ID(d) ((int)item.get_local_id(d))\n"
          "#define TVM_SYCL_GROUP_ID(d) ((int)item.get_group(d))\n"
          "#endif\n";
  // Warp-shuffle builtins (intrin_rule_sycl.cc) on the kernel's sub-group; `item` is the
  // nd_item in scope inside every generated kernel.
  code << "#define tvm_sycl_shuffle(v, lane) "
          "sycl::select_from_group(item.get_sub_group(), (v), (lane))\n";
  code << "#define tvm_sycl_shuffle_down(v, delta) "
          "sycl::shift_group_left(item.get_sub_group(), (v), (delta))\n";
  code << "#define tvm_sycl_shuffle_up(v, delta) "
          "sycl::shift_group_right(item.get_sub_group(), (v), (delta))\n";
  // Whole-work-group reductions (lower_thread_allreduce.cc, TVM_SYCL_GROUP_REDUCE=1).
  code << "#define tvm_sycl_group_reduce_add(v) "
          "sycl::reduce_over_group(item.get_group(), (v), sycl::plus<>())\n";
  code << "#define tvm_sycl_group_reduce_max(v) "
          "sycl::reduce_over_group(item.get_group(), (v), sycl::maximum<>())\n";
  code << "#define tvm_sycl_group_reduce_min(v) "
          "sycl::reduce_over_group(item.get_group(), (v), sycl::minimum<>())\n";
  const auto fpostproc = tvm::ffi::Function::GetGlobal("tvm_callback_opencl_postproc");

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>()) << "CodeGenSYCL: Can only take PrimFunc";
    auto f = Downcast<PrimFunc>(kv.second);
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch)
        << "CodeGenSYCL: expect calling_conv equals CallingConv::kDeviceKernelLaunch";
    CodeGenSYCL cg;
    cg.Init(output_ssa);
    // Group-local memory is allocated statically; budget = min(target limit, 48 KB static limit).
    int64_t target_limit =
        target->GetAttr<Integer>("max_shared_memory_per_block").value_or(Integer(MAX_SHARED_MEMORY_SIZE))->value;
    cg.SetSharedMemoryBudget(static_cast<size_t>(
        std::min<int64_t>(target_limit, static_cast<int64_t>(MAX_SHARED_MEMORY_SIZE))));
    cg.SetWarpSize(target->GetAttr<Integer>("thread_warp_size").value_or(Integer(32))->value);
    // erf/tanh workarounds unless the target names a CUDA architecture (behind the CUDA plugin);
    // TVM_SYCL_MATH_WORKAROUNDS=0/1 overrides.
    {
      const std::string arch = target->GetAttr<String>("arch").value_or(String(""));
      bool workarounds = arch.rfind("sm_", 0) != 0;
      if (const char* env = std::getenv("TVM_SYCL_MATH_WORKAROUNDS")) {
        workarounds = std::string(env) == "1";
      }
      cg.SetMathWorkarounds(workarounds);
    }
    cg.AddFunction(f);
    std::string fsource = cg.Finish();
    // Debug for SYCL
    VLOG(0) << "BuildSYCL: code:\n" << fsource;
    if (fpostproc) {
      fsource = (*fpostproc)(fsource, target).cast<std::string>();
    }
    // Each function is self-contained after the common prelude; the marker lets the runtime
    // compile groups of functions as separate translation units in parallel.
    code << "\n// TVM_SYCL_TU_BOUNDARY\n" << fsource;
  }

  return SYCLModuleCreate(code.str(), "sycl", ExtractFuncInfo(mod), code.str());
}

TVM_FFI_REGISTER_GLOBAL("target.build.sycl").set_body_typed(BuildSYCL);
}  // namespace codegen
}  // namespace tvm
