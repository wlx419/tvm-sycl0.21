# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""The Relax SYCL backend pipeline: the generic GPU pipeline plus graph capture.

When the pass context sets ``relax.backend.use_cuda_graph`` to True, RewriteCUDAGraph lifts
static regions into capture functions exactly as for CUDA, and the call sites are redirected to
the SYCL command-graph builtins (``vm.builtin.sycl_graph.*``, src/runtime/sycl/
sycl_graph_builtin.cc). Without that option the pipeline is identical to the generic GPU one.
"""
import tvm
from tvm import relax

from ..gpu_generic import pipeline as _generic

library_dispatch_passes = _generic.library_dispatch_passes
legalize_passes = _generic.legalize_passes
dataflow_lower_passes = _generic.dataflow_lower_passes

_CUDA_GRAPH_PREFIX = "vm.builtin.cuda_graph."
_SYCL_GRAPH_PREFIX = "vm.builtin.sycl_graph."


@relax.expr_functor.mutator
class _RenameGraphBuiltins(relax.PyExprMutator):
    def visit_extern_func_(self, op: relax.ExternFunc) -> relax.Expr:
        symbol = str(op.global_symbol)
        if symbol.startswith(_CUDA_GRAPH_PREFIX):
            return relax.ExternFunc(_SYCL_GRAPH_PREFIX + symbol[len(_CUDA_GRAPH_PREFIX):])
        return op


@tvm.transform.module_pass(opt_level=0, name="RedirectGraphBuiltinsToSYCL")
def redirect_graph_builtins(mod: tvm.IRModule, _ctx: tvm.transform.PassContext) -> tvm.IRModule:
    """Point RewriteCUDAGraph's runtime calls at the SYCL command-graph builtins.

    No-op unless the current target is a SYCL target, so it can also be placed in the
    target-agnostic default build pipeline used by tvm.compile.
    """
    target = tvm.target.Target.current(allow_none=True)
    if target is None or target.kind.name != "sycl":
        return mod
    updates = {}
    for gvar, func in mod.functions.items():
        if isinstance(func, relax.Function):
            updates[gvar] = _RenameGraphBuiltins(mod).visit_expr(func)
    for gvar, func in updates.items():
        mod[gvar] = func
    return mod


def finalize_passes(target: tvm.target.Target):  # pylint: disable=unused-argument
    """Finalization passes: generic GPU passes plus SYCL graph capture."""
    return [
        relax.transform.StaticPlanBlockMemory(),
        relax.transform.RewriteCUDAGraph(),
        redirect_graph_builtins,
        relax.transform.LowerAllocTensor(),
        relax.transform.KillAfterLastUse(),
        relax.transform.LowerRuntimeBuiltin(),
        relax.transform.ComputePrimValue(),
        relax.transform.VMShapeLower(),
        relax.transform.AttachGlobalSymbol(),
    ]


def get_default_pipeline(target: tvm.target.Target):
    """Return the default compilation pipeline for SYCL."""

    @tvm.transform.module_pass(opt_level=0)
    def _pipeline(mod: tvm.ir.IRModule, _ctx: tvm.transform.PassContext):
        with target:
            seq = tvm.transform.Sequential(
                library_dispatch_passes(target)
                + legalize_passes(target)
                + dataflow_lower_passes(target)
                + finalize_passes(target)
            )
            mod = seq(mod)
        return mod

    return _pipeline
