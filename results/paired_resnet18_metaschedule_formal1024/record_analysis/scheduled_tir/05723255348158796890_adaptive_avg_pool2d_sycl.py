# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(lv85: T.Buffer((T.int64(1), T.int64(512), T.int64(7), T.int64(7)), "float32"), adaptive_pool_avg: T.Buffer((T.int64(1), T.int64(512), T.int64(1), T.int64(1)), "float32")):
        T.func_attr({"op_pattern": 4, "tir.noalias": True})
        # with T.block("root"):
        adaptive_pool_sum_shared = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(1), T.int64(1)), scope="shared")
        for ax0_ax1_ax2_ax3_0_fused in T.thread_binding(T.int64(512), thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step": 512, "pragma_unroll_explicit": 1}):
            for ax0, ax1, ax2, ax3, ax4_ax5_fused_0 in T.grid(T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(4)):
                for ax4_ax5_fused_1 in T.thread_binding(T.int64(16), thread="threadIdx.x"):
                    with T.block("adaptive_pool_sum"):
                        v_ax0 = T.axis.spatial(T.int64(1), ax0)
                        v_ax1 = T.axis.spatial(T.int64(512), ax0_ax1_ax2_ax3_0_fused + ax1)
                        v_ax2, v_ax3 = T.axis.remap("SS", [ax2, ax3])
                        v_rv0 = T.axis.reduce(T.int64(7), (ax4_ax5_fused_0 * T.int64(16) + ax4_ax5_fused_1) // T.int64(7))
                        v_rv1 = T.axis.reduce(T.int64(7), (ax4_ax5_fused_0 * T.int64(16) + ax4_ax5_fused_1) % T.int64(7))
                        T.where(ax4_ax5_fused_0 * T.int64(16) + ax4_ax5_fused_1 < T.int64(49))
                        T.reads(lv85[v_ax0, v_ax1, v_ax2 * T.int64(7) + v_rv0, v_ax3 * T.int64(7) + v_rv1])
                        T.writes(adaptive_pool_sum_shared[v_ax0, v_ax1, v_ax2, v_ax3])
                        with T.init():
                            adaptive_pool_sum_shared[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0.0)
                        adaptive_pool_sum_shared[v_ax0, v_ax1, v_ax2, v_ax3] = adaptive_pool_sum_shared[v_ax0, v_ax1, v_ax2, v_ax3] + lv85[v_ax0, v_ax1, v_ax2 * T.int64(7) + v_rv0, v_ax3 * T.int64(7) + v_rv1]
            for ax3_1 in T.thread_binding(T.int64(16), thread="threadIdx.x"):
                with T.block("adaptive_pool_avg"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(512), ax0_ax1_ax2_ax3_0_fused)
                    v_ax2 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax3 = T.axis.spatial(T.int64(1), ax3_1)
                    T.where(T.Mul(T.int64(0), T.int64(16)) + ax3_1 < T.int64(1))
                    T.reads(adaptive_pool_sum_shared[v_ax0, v_ax1, v_ax2, v_ax3])
                    T.writes(adaptive_pool_avg[v_ax0, v_ax1, v_ax2, v_ax3])
                    adaptive_pool_avg[v_ax0, v_ax1, v_ax2, v_ax3] = adaptive_pool_sum_shared[v_ax0, v_ax1, v_ax2, v_ax3] * T.float32(0.020408163265306121)
