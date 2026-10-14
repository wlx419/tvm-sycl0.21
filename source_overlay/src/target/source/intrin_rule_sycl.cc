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
 * \file intrin_rule_sycl.cc
 * \brief SYCL lowering of TVM's warp-shuffle builtins to SYCL 2020 sub-group functions.
 *
 * Used when LowerThreadAllreduce takes the warp-shuffle path for SYCL
 * (TVM_SYCL_SUBGROUP_REDUCE=1). The calls are printed as tvm_sycl_shuffle{,_up,_down}(value, n),
 * macros defined in the generated prelude (codegen_sycl.cc) that apply select_from_group,
 * shift_group_right and shift_group_left to item.get_sub_group(). TVM passes width ==
 * warp_size, so no sub-warp segmentation is needed; the sub-group size equals the target's
 * thread_warp_size (32 behind the CUDA plugin, 64 on gfx906).
 */
#include <tvm/tir/builtin.h>
#include <tvm/tir/op_attr_types.h>

#include "../intrin_rule.h"

namespace tvm {
namespace codegen {
namespace intrin {
using tir::FLowerIntrinsic;

namespace {
PrimExpr DispatchSYCLShuffle(const PrimExpr& e) {
  const CallNode* call = e.as<CallNode>();
  ICHECK(call != nullptr);
  ICHECK_EQ(call->args.size(), 5);  // mask, value, lane_or_delta, width, warp_size
  const Op& op = Downcast<Op>(call->op);
  const char* name = op.same_as(builtin::tvm_warp_shuffle())      ? "tvm_sycl_shuffle"
                     : op.same_as(builtin::tvm_warp_shuffle_up()) ? "tvm_sycl_shuffle_up"
                                                                  : "tvm_sycl_shuffle_down";
  return Call(call->dtype, builtin::call_pure_extern(),
              {StringImm(name), call->args[1], call->args[2]});
}
}  // namespace

TVM_REGISTER_OP("tir.tvm_warp_shuffle")
    .set_attr<FLowerIntrinsic>("sycl.FLowerIntrinsic", DispatchSYCLShuffle);
TVM_REGISTER_OP("tir.tvm_warp_shuffle_up")
    .set_attr<FLowerIntrinsic>("sycl.FLowerIntrinsic", DispatchSYCLShuffle);
TVM_REGISTER_OP("tir.tvm_warp_shuffle_down")
    .set_attr<FLowerIntrinsic>("sycl.FLowerIntrinsic", DispatchSYCLShuffle);

}  // namespace intrin
}  // namespace codegen
}  // namespace tvm
