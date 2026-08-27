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

#include <cmath>
#include <string>
#include <vector>

#include "../../runtime/sycl/sycl_module.h"
#include "../../runtime/thread_storage_scope.h"
#include "../build_common.h"

namespace tvm {
namespace codegen {
  // 在文件开头合适的位置添加定义
constexpr size_t MAX_SHARED_MEMORY_SIZE = 49152; // Further reduced from 49152

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
// void CodeGenSYCL::AddFunction(const GlobalVar& gvar, const PrimFunc& f) {
//   // If the function has already been forward-declared, this is a
//   // no-op.
//   DeclareFunction(gvar, f);
//   auto function_name = GetFunctionName(gvar);
//   this->InitFuncState(f);
//   // clear pre mem_allocate_code
//   this->mem_allocate_code = std::ostringstream("");
//   // reserve keywords
//   ReserveKeywordsAsUnique();
//   this->PrintExtraAttrs(f, stream);

//   auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
//   ICHECK(global_symbol.defined())
//       << "CodeGenSYCL: Expect PrimFunc to have the global_symbol attribute";
//   bool no_alias = f->HasNonzeroAttr(tir::attr::kNoAlias);
//   // Print the packed function
//   //stream << "// CodeGenSYCL: NOTE: Auto-generated packed function\n";
//   stream << "// Function: " << static_cast<std::string>(global_symbol.value()) << "\n";
//   PrintFuncPrefix(stream);
//   stream << " " << static_cast<std::string>(global_symbol.value());
//   stream << "(queue &Q, sycl::range<3> dimGrid, sycl::range<3> dimBlock, void** void_args) {\n";
//   int func_scope = this->BeginScope();
//   // Print the packed function's void_args
//   for (size_t i = 0; i < f->params.size(); ++i) {
//     tir::Var v = f->params[i];
//     std::string vid = AllocVarID(v.get());
//     if (v.dtype().is_handle()) {
//       // Register handle data type
//       // TODO(tvm-team): consider simply keep type info in the
//       // type annotation(via a normalizing rewriting).
//       if (auto* ptr = v->type_annotation.as<PointerTypeNode>()) {
//         if (auto* prim = ptr->element_type.as<PrimTypeNode>()) {
//           RegisterHandleType(v.get(), prim->dtype);
//         }
//       }
//     }
//     this->PrintIndent();
//     PrintType(GetType(v), stream);
//     if (no_alias) {
//       PrintRestrict(v, stream);
//     }
//     stream << ' ' << vid << " = (";
//     PrintType(GetType(v), stream);
//     stream << ")(*(int64_t *)(void_args[" << i << "]));\n";
//   }

//   // Print submit  
//   this->PrintIndent();
//   stream << "Q.submit([&](handler &h) {\n";
//   int func_scope2 = this->BeginScope();
  
//   this->PrintIndent();
//   stream << "h.parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](sycl::nd_item<3> item) [[intel::kernel_args_restrict]] {\n";
//   int func_scope3 = this->BeginScope();
//   // Function code
//   this->PrintStmt(f->body);
//   this->EndScope(func_scope3);
//   this->PrintIndent();
//   stream << "});\n";
//   this->EndScope(func_scope2);
//   this->PrintIndent();
//   stream << "});\n";
//   this->PrintIndent();
//   this->EndScope(func_scope);
//   stream << "}\n";


//   // // clear previous generated state.
//   // InitFuncState(f);

//   // PrintFunctionSignature(function_name, f, stream);
//   // stream << " {\n";
//   // this->PreFunctionBody(f);
//   // int func_scope = this->BeginScope();
//   // this->PrintStmt(f->body);
//   // this->EndScope(func_scope);
//   // this->PrintIndent();
//   // this->stream << "}\n\n";
// }

// void CodeGenSYCL::AddFunction(const PrimFunc& f) {
//   // from CodeGenC -------------------
//   // clear previous generated state.
//   this->InitFuncState(f);
//   // clear pre mem_allocate_code
//   this->mem_allocate_code = std::ostringstream("");
//   // reserve keywords
//   ReserveKeywordsAsUnique();
//   this->PrintExtraAttrs(f, stream);

//   auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
//   ICHECK(global_symbol.defined())
//       << "CodeGenSYCL: Expect PrimFunc to have the global_symbol attribute";
//   bool no_alias = f->HasNonzeroAttr(tir::attr::kNoAlias);
//   // Print the packed function
//   stream << "// CodeGenSYCL: NOTE: Auto-generated packed function\n";
//   stream << "// Function: " << static_cast<std::string>(global_symbol.value()) << "\n";
//   //PrintFuncPrefix(stream);
//   stream << " " << static_cast<std::string>(global_symbol.value());
//   stream << "(queue &Q, sycl::range<3> dimGrid, sycl::range<3> dimBlock, void** void_args) {\n";
//   int func_scope = this->BeginScope();
//   // Print the packed function's void_args
//   for (size_t i = 0; i < f->params.size(); ++i) {
//     tir::Var v = f->params[i];
//     std::string vid = AllocVarID(v.get());
//     if (v.dtype().is_handle()) {
//       // Register handle data type
//       // TODO(tvm-team): consider simply keep type info in the
//       // type annotation(via a normalizing rewriting).
//       if (auto* ptr = v->type_annotation.as<PointerTypeNode>()) {
//         if (auto* prim = ptr->element_type.as<PrimTypeNode>()) {
//           RegisterHandleType(v.get(), prim->dtype);
//         }
//       }
//     }
//     this->PrintIndent();
//     PrintType(GetType(v), stream);
//     if (no_alias) {
//       PrintRestrict(v, stream);
//     }
//     stream << ' ' << vid << " = (";
//     PrintType(GetType(v), stream);
//     stream << ")(*(int64_t *)(void_args[" << i << "]));\n";
//   }

//   // Print submit  
//   this->PrintIndent();
//   stream << "Q.submit([&](handler &h) {\n";
//   int func_scope2 = this->BeginScope();
  
//   this->PrintIndent();
//   stream << "h.parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](sycl::nd_item<3> item) [[intel::kernel_args_restrict]] {\n";
//   int func_scope3 = this->BeginScope();
//   // Function code
//   this->PrintStmt(f->body);
//   this->EndScope(func_scope3);
//   this->PrintIndent();
//   stream << "});\n";
//   this->EndScope(func_scope2);
//   this->PrintIndent();
//   stream << "});\n";
//   this->PrintIndent();
//   this->EndScope(func_scope);
//   stream << "}\n";
//   /*
//   // insert mem_allocate_code
//   std::string preCode = stream.str();
//   size_t start = preCode.find("Q.submit([&](handler &h) {\n");
//   start += strlen("Q.submit([&](handler &h) {\n");
//   preCode.insert(start, mem_allocate_code.str());
//   stream = std::ostringstream(preCode);
//   */
// }

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
  bool no_alias = f->HasNonzeroAttr(tir::attr::kNoAlias);
  // Print the packed function
  stream << "// CodeGenSYCL: NOTE: Auto-generated packed function\n";
  stream << "// Function: " << static_cast<std::string>(global_symbol.value()) << "\n";
  PrintFuncPrefix(stream);
  stream << " " << static_cast<std::string>(global_symbol.value());
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

  // Print submit  
  this->PrintIndent();
  stream << "Qh.submit([=](handler &h) {\n";
  int func_scope2 = this->BeginScope();
  
  this->PrintIndent();
  stream << "h.parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](sycl::nd_item<3> item) [[intel::kernel_args_restrict]] {\n";
  int func_scope3 = this->BeginScope();
  // Function code
  this->PrintStmt(f->body);
  this->EndScope(func_scope3);
  this->PrintIndent();
  stream << "});\n";
  this->EndScope(func_scope2);
  this->PrintIndent();
  stream << "});\n";
  this->PrintIndent();
  this->EndScope(func_scope);
  stream << "}\n";
  /*
  // insert mem_allocate_code
  std::string preCode = stream.str();
  size_t start = preCode.find("Q.submit([&](handler &h) {\n");
  start += strlen("Q.submit([&](handler &h) {\n");
  preCode.insert(start, mem_allocate_code.str());
  stream = std::ostringstream(preCode);
  */
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
  if (ts.rank == 1) {
    // swap the first(0) and third(2) dimension size in a SYCL workGroup.
    os << "item.get_local_id(" << 2-ts.dim_index << ")";
  } else {
    os << "item.get_group(" << ts.dim_index << ")";
  }
  var_idmap_[iv->var.get()] = CastFromTo(os.str(), DataType::UInt(64), iv->var.dtype());
}

#include <sycl/sycl.hpp>
#include <cstdint>

// 只考虑 Linux/LP64：确保 long/unsigned long 是 64 位
static_assert(sizeof(long) == 8, "This code assumes LP64 (Linux), where long is 64-bit.");
static_assert(sizeof(unsigned long) == 8, "This code assumes LP64 (Linux), where unsigned long is 64-bit.");

void CodeGenSYCL::PrintType(DataType t, std::ostream& os) {  // NOLINT(*)
  const int lanes = t.lanes();

  // 辅助：按 lanes 输出标量或 sycl::vec<T, N>；超常规宽度退化为指针
  auto print_vec = [&](const char* base) {
    if (lanes == 1) {
      os << base;
    } else if ((lanes >= 2 && lanes <= 4) || lanes == 8 || lanes == 16) {
      os << "sycl::vec<" << base << ", " << lanes << ">";
    } else {
      // 非 2/3/4/8/16 的向量宽度，使用指针退化策略（与原逻辑一致）
      os << base << "*";
    }
  };

  // 句柄/void/bool
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

  // 浮点
  if (t.is_float()) {
    switch (t.bits()) {
      case 16:
        enable_fp16_ = true;
        // 使用 SYCL 2020 类型 sycl::half
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

  // 有符号整型
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
        // Linux/LP64: long 即 64 位
        print_vec("long");
        return;
      default:
        LOG(FATAL) << "Cannot convert int type " << t << " to SYCL type";
    }
  }

  // 无符号整型
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
        // Linux/LP64: unsigned long 即 64 位
        print_vec("unsigned long");
        return;
      default:
        LOG(FATAL) << "Cannot convert uint type " << t << " to SYCL type";
    }
  }

  LOG(FATAL) << "Cannot convert type " << t << " to SYCL type";
}


// void CodeGenSYCL::PrintType(DataType t, std::ostream& os) {  // NOLINT(*)
//   int lanes = t.lanes();
//   if (t.is_handle()) {
//     ICHECK_EQ(lanes, 1) << "do not yet support vector types";
//     os << "void*";
//     return;
//   }
//   if (t.is_void()) {
//     os << "void";
//     return;
//   }
//   if (t == DataType::Bool()) {
//     os << "bool";
//     return;
//   }
//   bool fail = false;
//   if (t.is_float()) {
//     switch (t.bits()) {
//       case 16:
//         os << "half";
//         enable_fp16_ = true;
//         break;
//       case 32:
//         os << "float";
//         break;
//       case 64:
//         os << "double";
//         enable_fp64_ = true;
//         break;
//       default:
//         fail = true;
//         break;
//     }
//     if (!fail && lanes == 1) return;
//     // Only use SYCL vector types for supported sizes
//     if (!fail && ((lanes >= 2 && lanes <= 4) || lanes == 8 || lanes == 16)) {
//       os << lanes;
//       return;
//     }
//     // For larger sizes, we'll handle manually via arrays
//     if (!fail) {
//       os << "*";
//       return;
//     }
//   } else if (t.is_int() && t.bits() == 32) {
//     // Add support for int32
//     os << "int";
//     if (lanes > 1) {
//       // Handle vector types
//       if ((lanes >= 2 && lanes <= 4) || lanes == 8 || lanes == 16) {
//         os << lanes;
//       } else {
//         os << "*";
//       }
//     }
//     return;
//   }
//   LOG(FATAL) << "Cannot convert type " << t << " to SYCL type";
// }

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
  if(!(t.lanes()==2||t.lanes()==4||t.lanes()==8||t.lanes()==16))
    return CodeGenC::GetVecLoad(t,buf,base);

  std::ostringstream os;
  os<<"({\n"; this->PrintIndent();
  os<<"using vec_t = sycl::vec<"; PrintType(t.element_of(),os);
  os<<","<<t.lanes()<<">;\n";

  this->PrintIndent(); os<<"auto _raw_ = "; PrintVecAddr(buf,t,base,os); os<<";\n";
  this->PrintIndent(); os<<"auto _raw_vec_ = reinterpret_cast<vec_t*>(_raw_);\n";

  auto it = alloc_storage_scope_.find(buf->data.get());
  std::string scope = (it!=alloc_storage_scope_.end())? it->second : "global";
  const char* space = (scope=="shared") ?
       "sycl::access::address_space::local_space" :
       "sycl::access::address_space::global_space";

  this->PrintIndent();
  os<<"auto _vptr_ = sycl::address_space_cast<"<<space
    <<", sycl::access::decorated::no>(_raw_vec_);\n";    // ★ 删除 vec_t

  this->PrintIndent(); os<<"vec_t _v_ = *_vptr_;\n";
  this->PrintIndent(); os<<"_v_; })";
  return os.str();
}

// ---------------- PrintVecStore ----------------
void CodeGenSYCL::PrintVecStore(const BufferNode* buf,DataType t,PrimExpr base,
                                const std::string& value){
  if(!(t.lanes()==2||t.lanes()==4||t.lanes()==8||t.lanes()==16)){
    CodeGenC::PrintVecStore(buf,t,base,value); return;
  }
  this->PrintIndent();
  stream<<"{ using vec_t = sycl::vec<"; PrintType(t.element_of(),stream);
  stream<<","<<t.lanes()<<">;\n";

  this->PrintIndent(); stream<<"auto _raw_ = "; PrintVecAddr(buf,t,base,stream);
  stream<<";\n";
  this->PrintIndent(); stream<<"auto _raw_vec_ = reinterpret_cast<vec_t*>(_raw_);\n";

  auto it = alloc_storage_scope_.find(buf->data.get());
  std::string scope = (it!=alloc_storage_scope_.end())? it->second : "global";
  const char* space = (scope=="shared") ?
       "sycl::access::address_space::local_space" :
       "sycl::access::address_space::global_space";

  this->PrintIndent();
  stream<<"auto _vptr_ = sycl::address_space_cast<"<<space
        <<", sycl::access::decorated::no>(_raw_vec_);\n";  // ★ 删除 vec_t

  this->PrintIndent();
  stream<<"*_vptr_ = static_cast<vec_t>("<<value<<"); }\n";
}


// void CodeGenSYCL::PrintStorageSync(const CallNode* op) {
//   const std::string& sync = op->args[0].as<StringImmNode>()->value;
//   if (sync == "warp") {
//     this->PrintIndent();
//     this->stream << "group_barrier(item.get_sub_group());\n";
//   } else if (sync == "shared") {
//     this->PrintIndent();
//     this->stream << "group_barrier(item.get_group());\n";
//   } else if (sync == "global") {
//     LOG(FATAL) << "global barrier is not supported";
//   }
// }

void CodeGenSYCL::PrintStorageSync(const CallNode* op) {
  const std::string& sync = op->args[0].as<StringImmNode>()->value;
  if (sync == "warp") {
    // SYCL: 避免在可能的非一致分支里发 subgroup barrier 造成死锁，退化为 no-op
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
  if(scope =="shared"){
    std::string code = static_cast<std::ostringstream&>(os).str();
    std::string str = code.substr(code.length()-10, std::string::npos);
    if(str!="(volatile "){
      std::cout<<str<<std::endl;
    }
  }
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
  
    // 检查加上这块后总共会不会超
    if (current_shared_memory_usage_ + required_bytes > MAX_SHARED_MEMORY_SIZE) {
      size_t remaining = (MAX_SHARED_MEMORY_SIZE > current_shared_memory_usage_)
                             ? (MAX_SHARED_MEMORY_SIZE - current_shared_memory_usage_)
                             : 0;
      size_t max_elems = remaining / bytes_per_elem;
      if (max_elems == 0) {
        LOG(FATAL) << "Not enough shared memory left. Request " << required_bytes
                   << " bytes, but only " << remaining << " bytes available.";
      }
      LOG(WARNING) << "Trimming shared memory allocation from " << required_bytes
                   << " to " << (max_elems * bytes_per_elem) << " bytes.";
      constant_size = max_elems;
      required_bytes = constant_size * bytes_per_elem;
    }
  
    current_shared_memory_usage_ += required_bytes;
  
    this->PrintIndent();
    stream << "auto " << vid << " = *sycl::ext::oneapi::group_local_memory<";
    PrintType(op->dtype, stream);
    stream << '[' << constant_size << "]>(item.get_group());\n";
  }
   else {
    this->PrintIndent();
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
    if ((func->value == "erf" || func->value == "erff") && op->dtype.is_float() &&
        op->dtype.bits() == 32 && op->args.size() == 2) {
      const std::string x = PrintExpr(op->args[1]);
      os << "((float)sycl::erf((double)(" << x << ")))";
      return;
    }
    if ((func->value == "tanh" || func->value == "tanhf") && op->dtype.is_float() &&
        op->dtype.bits() == 32 && op->args.size() == 2) {
      const std::string x = PrintExpr(op->args[1]);
      const std::string positive_exp = "exp(-2.000000e+00f * (" + x + "))";
      const std::string negative_exp = "exp(2.000000e+00f * (" + x + "))";
      os << "select(((" << negative_exp << " - 1.000000e+00f) / (" << negative_exp
         << " + 1.000000e+00f)), ((1.000000e+00f - " << positive_exp
         << ") / (1.000000e+00f + " << positive_exp << ")), (0.000000e+00f <= " << x
         << "))";
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

void CodeGenSYCL::VisitExpr_(const MinNode* op, std::ostream& os) {
  PrintBinaryExpr(op, "min", os, this);
}

void CodeGenSYCL::VisitExpr_(const MaxNode* op, std::ostream& os) {
  PrintBinaryExpr(op, "max", os, this);
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
  //stream << vec << "[" << std::hex << i << "]" << " = " << value << ";\n" << std::dec;
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

// void CodeGenSYCL::VisitExpr_(const RampNode* op, std::ostream& os) {  // NOLINT(*)
//   // constraint of current logic
//   ICHECK_EQ(op->base.dtype(), DataType::Int(32));
//   os << "((int" << op->lanes << "){";
//   for (int i = 0; i < op->lanes; i++) {
//     os << "(" << PrintExpr(op->base) << ")"
//        << "+(" << PrintExpr(op->stride) << "*" << i << ")";
//     if (i != op->lanes - 1) os << ", ";
//   }
//   os << "})";
// }

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
#include <execinfo.h>
#include <iostream>
#include <cstdlib>

void print_stack_trace() {
    const int max_frames = 64;
    void* buffer[max_frames];
    int num_frames = backtrace(buffer, max_frames);
    
    char** symbols = backtrace_symbols(buffer, num_frames);
    if (symbols) {
        for (int i = 0; i < num_frames; ++i) {
            std::cout << symbols[i] << std::endl;
        }
        free(symbols);
    } else {
        std::cerr << "Failed to get backtrace symbols" << std::endl;
    }
}
// runtime::Module BuildSYCL(IRModule mod, Target target) {
//   //print_stack_trace();
//   //using tvm::runtime::Registry;
//   //std::cout<<"[SYCL] Build SYCL module."<<std::endl;
//   bool output_ssa = false;
//   std::stringstream code;
//   code << "// tvm target: " << target->str() << "\n";
//   code << "#include <sycl/sycl.hpp>\n";
//   code << "using namespace sycl;\n";
//   //const auto fpostproc = tvm::ffi::Function::GetGlobal("tvm_callback_opencl_postproc");

//   // Debug: Log target and initial module information
//   VLOG(0) << "Target: " << target->str();
  
//   Map<GlobalVar, PrimFunc> functions;
//   for (auto [gvar, base_func] : mod->functions) {
//     ICHECK(base_func->IsInstance<PrimFuncNode>()) << "CodeGenOpenCL: Can only take PrimFunc";
//     auto prim_func = Downcast<PrimFunc>(base_func);
//     auto calling_conv = prim_func->GetAttr<Integer>(tvm::attr::kCallingConv);
//     ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch)
//         << "CodeGenSYCL: expect calling_conv equals CallingConv::kDeviceKernelLaunch";
//     functions.Set(gvar, prim_func);
//   }

//   //std::stringstream code;
//   const auto fpostproc = tvm::ffi::Function::GetGlobal("tvm_callback_opencl_postproc");
//   for (auto [gvar, prim_func] : functions) {
//     code << "// Function: " << gvar->name_hint << std::endl;
//     CodeGenSYCL cg;
//     cg.Init(output_ssa);
//     for (auto [other_gvar, other_prim_func] : functions) {
//       cg.DeclareFunction(other_gvar, other_prim_func);
//     }
//     cg.AddFunction(prim_func);
//     std::string fsource = cg.Finish();
//     if (fpostproc) {
//       fsource = (*fpostproc)(fsource, target).cast<std::string>();
//     }
//     code << fsource;
//   // for (auto kv : mod->functions) {
//   //   ICHECK(kv.second->IsInstance<PrimFuncNode>()) << "CodeGenSYCL: Can only take PrimFunc";
//   //   auto f = Downcast<PrimFunc>(kv.second);
//   //   auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
//   //   ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch)
//   //       << "CodeGenSYCL: expect calling_conv equals CallingConv::kDeviceKernelLaunch";
    
//   //   CodeGenSYCL cg;
//   //   cg.Init(output_ssa);
//   //   cg.AddFunction(f);
//   //   std::string fsource = cg.Finish();
    
//   //   // Debug: Print the generated SYCL code before post-processing
//   //   VLOG(0) << "BuildSYCL: code before post-processing:\n" << fsource;
    
//   //   if (fpostproc) {
//   //     // Post-process the SYCL code
//   //     std::string processed_source = (*fpostproc)(fsource, target).cast<std::string>();
      
//   //     // Debug: Log post-processed source
//   //     VLOG(0) << "Post-processed source:\n" << processed_source;
      
//   //     if (processed_source.empty()) {
//   //       // Handle empty post-processed source gracefully
//   //       VLOG(0) << "Warning: Post-processing produced an empty string.";
//   //     } else {
//   //       fsource = processed_source;
//   //     }
//   //   } else {
//   //     VLOG(0) << "Warning: Post-processing function 'fpostproc' is not available.";
//   //   }
    
//     // Add the final function source to the overall code
//     code << fsource;
//     std::cout << fsource << std::endl;
//     // std::ofstream temp_file("temp_output.txt", std::ios::app);

//     // if (!temp_file.is_open()) {
//     //     std::cerr << "Error opening file for writing!" << std::endl;
//     // }

//     // temp_file << "sycl code" << std::endl;

//     // // 将 fsource 的内容追加到临时文件
//     // temp_file << fsource << std::endl;

//     // // 关闭文件流
//     // temp_file.close();

//   }

//   // Return the created SYCL module with the final code
//   return SYCLModuleCreate(code.str(), "sycl", ExtractFuncInfo(mod), code.str());
// }


runtime::Module BuildSYCL(IRModule mod, Target target) {
  //using tvm::runtime::Registry;
  //std::cout<<"[SYCL] Build SYCL module."<<std::endl;
  bool output_ssa = false;
  std::stringstream code;
  code << "// tvm target: " << target->str() << "\n";
  code << "#include <sycl/sycl.hpp>\n";
  code << "using namespace sycl;\n";
  const auto fpostproc = tvm::ffi::Function::GetGlobal("tvm_callback_opencl_postproc");

  //const auto* fpostproc = tvm::ffi::Function::GetGlobal("tvm_callback_opencl_postproc");
  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>()) << "CodeGenSYCL: Can only take PrimFunc";
    auto f = Downcast<PrimFunc>(kv.second);
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch)
        << "CodeGenSYCL: expect calling_conv equals CallingConv::kDeviceKernelLaunch";
    CodeGenSYCL cg;
    cg.Init(output_ssa);
    cg.AddFunction(f);
    std::string fsource = cg.Finish();
    // Debug for SYCL
    VLOG(0) << "BuildSYCL: code:\n" << fsource;
    if (fpostproc) {
      //fsource = (*fpostproc)(fsource).operator std::string();
      fsource = (*fpostproc)(fsource, target).cast<std::string>();
    }
    code << fsource;
  }

  return SYCLModuleCreate(code.str(), "sycl", ExtractFuncInfo(mod), code.str());
}

TVM_FFI_REGISTER_GLOBAL("target.build.sycl").set_body_typed(BuildSYCL);
}  // namespace codegen
}  // namespace tvm
