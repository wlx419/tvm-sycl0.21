# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(lv3: T.Buffer((T.int64(1), T.int64(64), T.int64(112), T.int64(112)), "float32"), pool_max: T.Buffer((T.int64(1), T.int64(64), T.int64(56), T.int64(56)), "float32")):
        T.func_attr({"op_pattern": 4, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(196), thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step": 1024, "pragma_unroll_explicit": 1}):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("pool_max_init"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(3136))
                    v_ax2 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(3136) // T.int64(56))
                    v_ax3 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(56))
                    T.reads()
                    T.writes(pool_max[v_ax0, v_ax1, v_ax2, v_ax3])
                    pool_max[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(-340282346638528859811704183484516925440.0)
                for rv0, rv1 in T.grid(T.int64(3), T.int64(3)):
                    with T.block("pool_max_update"):
                        v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ax1 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(3136))
                        v_ax2 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(3136) // T.int64(56))
                        v_ax3 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(56))
                        v_rv0, v_rv1 = T.axis.remap("RR", [rv0, rv1])
                        T.reads(pool_max[v_ax0, v_ax1, v_ax2, v_ax3], lv3[v_ax0, v_ax1, v_ax2 * T.int64(2) + v_rv0 - T.int64(1), v_ax3 * T.int64(2) + v_rv1 - T.int64(1)])
                        T.writes(pool_max[v_ax0, v_ax1, v_ax2, v_ax3])
                        pool_max[v_ax0, v_ax1, v_ax2, v_ax3] = T.max(pool_max[v_ax0, v_ax1, v_ax2, v_ax3], T.if_then_else(T.int64(1) <= v_ax2 * T.int64(2) + v_rv0 and v_ax2 * T.int64(2) + v_rv0 < T.int64(113) and T.int64(1) <= v_ax3 * T.int64(2) + v_rv1 and v_ax3 * T.int64(2) + v_rv1 < T.int64(113), lv3[v_ax0, v_ax1, v_ax2 * T.int64(2) + v_rv0 - T.int64(1), v_ax3 * T.int64(2) + v_rv1 - T.int64(1)], T.float32(-340282346638528859811704183484516925440.0)))
