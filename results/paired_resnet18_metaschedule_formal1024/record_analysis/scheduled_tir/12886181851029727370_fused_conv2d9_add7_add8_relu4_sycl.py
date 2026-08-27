# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(lv68: T.Buffer((T.int64(1), T.int64(512), T.int64(7), T.int64(7)), "float32"), param_0: T.Buffer((T.int64(512), T.int64(512), T.int64(3), T.int64(3)), "float32"), param_1: T.Buffer((T.int64(1), T.int64(512), T.int64(1), T.int64(1)), "float32"), lv74: T.Buffer((T.int64(1), T.int64(512), T.int64(7), T.int64(7)), "float32"), compute_intermediate: T.Buffer((T.int64(1), T.int64(512), T.int64(7), T.int64(7)), "float32")):
        T.func_attr({"tir.noalias": True})
        # with T.block("root"):
        conv2d_nchw_intermediate_local = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(7), T.int64(7)), scope="local")
        pad_temp_shared = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(9), T.int64(9)), scope="shared")
        param_0_shared = T.alloc_buffer((T.int64(512), T.int64(512), T.int64(3), T.int64(3)), scope="shared")
        for nn_0_ff_0_yy_0_xx_0_fused in T.thread_binding(T.int64(64), thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step": 16, "pragma_unroll_explicit": 1}):
            for nn_1_ff_1_yy_1_xx_1_fused in T.thread_binding(T.int64(2), thread="vthread.x"):
                for nn_2_ff_2_yy_2_xx_2_fused in T.thread_binding(T.int64(196), thread="threadIdx.x"):
                    for nn_3_init, ff_3_init, yy_3_init, xx_3_init, nn_4_init, ff_4_init, yy_4_init, xx_4_init in T.grid(T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(1)):
                        with T.block("conv2d_nchw_init"):
                            v_nn = T.axis.spatial(T.int64(1), nn_3_init + nn_4_init)
                            v_ff = T.axis.spatial(T.int64(512), nn_0_ff_0_yy_0_xx_0_fused * T.int64(8) + nn_1_ff_1_yy_1_xx_1_fused * T.int64(4) + nn_2_ff_2_yy_2_xx_2_fused // T.int64(49) + ff_3_init + ff_4_init)
                            v_yy = T.axis.spatial(T.int64(7), nn_2_ff_2_yy_2_xx_2_fused % T.int64(49) // T.int64(7) + yy_3_init + yy_4_init)
                            v_xx = T.axis.spatial(T.int64(7), nn_2_ff_2_yy_2_xx_2_fused % T.int64(7) + xx_3_init + xx_4_init)
                            T.reads()
                            T.writes(conv2d_nchw_intermediate_local[v_nn, v_ff, v_yy, v_xx])
                            T.block_attr({"meta_schedule.thread_extent_high_inclusive": 1024, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                            conv2d_nchw_intermediate_local[v_nn, v_ff, v_yy, v_xx] = T.float32(0.0)
                    for rc_0, ry_0, rx_0 in T.grid(T.int64(64), T.int64(1), T.int64(3)):
                        for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(2)):
                            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(196), thread="threadIdx.x"):
                                for ax0_ax1_ax2_ax3_fused_2 in T.vectorized(T.int64(2)):
                                    with T.block("pad_temp_shared"):
                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v1 = T.axis.spatial(T.int64(512), rc_0 * T.int64(8) + (ax0_ax1_ax2_ax3_fused_0 * T.int64(392) + ax0_ax1_ax2_ax3_fused_1 * T.int64(2) + ax0_ax1_ax2_ax3_fused_2) // T.int64(63))
                                        v2 = T.axis.spatial(T.int64(9), (ax0_ax1_ax2_ax3_fused_0 * T.int64(392) + ax0_ax1_ax2_ax3_fused_1 * T.int64(2) + ax0_ax1_ax2_ax3_fused_2) % T.int64(63) // T.int64(7))
                                        v3 = T.axis.spatial(T.int64(9), rx_0 + (ax0_ax1_ax2_ax3_fused_0 * T.int64(392) + ax0_ax1_ax2_ax3_fused_1 * T.int64(2) + ax0_ax1_ax2_ax3_fused_2) % T.int64(7))
                                        T.where((ax0_ax1_ax2_ax3_fused_0 * T.int64(196) + ax0_ax1_ax2_ax3_fused_1) * T.int64(2) + ax0_ax1_ax2_ax3_fused_2 < T.int64(504))
                                        T.reads(lv68[v0, v1, v2 - T.int64(1), v3 - T.int64(1)])
                                        T.writes(pad_temp_shared[v0, v1, v2, v3])
                                        pad_temp_shared[v0, v1, v2, v3] = T.if_then_else(T.int64(1) <= v2 and v2 < T.int64(8) and T.int64(1) <= v3 and v3 < T.int64(8), lv68[v0, v1, v2 - T.int64(1), v3 - T.int64(1)], T.float32(0.0))
                        for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(1)):
                            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(196), thread="threadIdx.x"):
                                for ax0_ax1_ax2_ax3_fused_2 in T.vectorized(T.int64(4)):
                                    with T.block("param_0_shared"):
                                        v0 = T.axis.spatial(T.int64(512), nn_0_ff_0_yy_0_xx_0_fused * T.int64(8) + (ax0_ax1_ax2_ax3_fused_0 * T.int64(784) + ax0_ax1_ax2_ax3_fused_1 * T.int64(4) + ax0_ax1_ax2_ax3_fused_2) // T.int64(24))
                                        v1 = T.axis.spatial(T.int64(512), rc_0 * T.int64(8) + (ax0_ax1_ax2_ax3_fused_0 * T.int64(784) + ax0_ax1_ax2_ax3_fused_1 * T.int64(4) + ax0_ax1_ax2_ax3_fused_2) % T.int64(24) // T.int64(3))
                                        v2 = T.axis.spatial(T.int64(3), (ax0_ax1_ax2_ax3_fused_0 * T.int64(784) + ax0_ax1_ax2_ax3_fused_1 * T.int64(4) + ax0_ax1_ax2_ax3_fused_2) % T.int64(3))
                                        v3 = T.axis.spatial(T.int64(3), rx_0)
                                        T.where((ax0_ax1_ax2_ax3_fused_0 * T.int64(196) + ax0_ax1_ax2_ax3_fused_1) * T.int64(4) + ax0_ax1_ax2_ax3_fused_2 < T.int64(192))
                                        T.reads(param_0[v0, v1, v2, v3])
                                        T.writes(param_0_shared[v0, v1, v2, v3])
                                        param_0_shared[v0, v1, v2, v3] = param_0[v0, v1, v2, v3]
                        for rc_1, ry_1, rx_1, nn_3, ff_3, yy_3, xx_3, rc_2, ry_2, rx_2, nn_4, ff_4, yy_4, xx_4 in T.grid(T.int64(1), T.int64(3), T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(8), T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(1)):
                            with T.block("conv2d_nchw_update"):
                                v_nn = T.axis.spatial(T.int64(1), nn_3 + nn_4)
                                v_ff = T.axis.spatial(T.int64(512), nn_0_ff_0_yy_0_xx_0_fused * T.int64(8) + nn_1_ff_1_yy_1_xx_1_fused * T.int64(4) + nn_2_ff_2_yy_2_xx_2_fused // T.int64(49) + ff_3 + ff_4)
                                v_yy = T.axis.spatial(T.int64(7), nn_2_ff_2_yy_2_xx_2_fused % T.int64(49) // T.int64(7) + yy_3 + yy_4)
                                v_xx = T.axis.spatial(T.int64(7), nn_2_ff_2_yy_2_xx_2_fused % T.int64(7) + xx_3 + xx_4)
                                v_rc = T.axis.reduce(T.int64(512), rc_0 * T.int64(8) + rc_1 * T.int64(8) + rc_2)
                                v_ry = T.axis.reduce(T.int64(3), ry_0 * T.int64(3) + ry_1 + ry_2)
                                v_rx = T.axis.reduce(T.int64(3), rx_0 + rx_1 + rx_2)
                                T.reads(conv2d_nchw_intermediate_local[v_nn, v_ff, v_yy, v_xx], pad_temp_shared[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], param_0_shared[v_ff, v_rc, v_ry, v_rx])
                                T.writes(conv2d_nchw_intermediate_local[v_nn, v_ff, v_yy, v_xx])
                                T.block_attr({"meta_schedule.thread_extent_high_inclusive": 1024, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                                conv2d_nchw_intermediate_local[v_nn, v_ff, v_yy, v_xx] = conv2d_nchw_intermediate_local[v_nn, v_ff, v_yy, v_xx] + pad_temp_shared[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * param_0_shared[v_ff, v_rc, v_ry, v_rx]
                    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(1), T.int64(1), T.int64(1)):
                        with T.block("conv2d_nchw_intermediate_local"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(512), nn_0_ff_0_yy_0_xx_0_fused * T.int64(8) + nn_1_ff_1_yy_1_xx_1_fused * T.int64(4) + nn_2_ff_2_yy_2_xx_2_fused // T.int64(49) + ax1)
                            v2 = T.axis.spatial(T.int64(7), nn_2_ff_2_yy_2_xx_2_fused % T.int64(49) // T.int64(7) + ax2)
                            v3 = T.axis.spatial(T.int64(7), nn_2_ff_2_yy_2_xx_2_fused % T.int64(7) + ax3)
                            T.reads(conv2d_nchw_intermediate_local[v0, v1, v2, v3], param_1[v0, v1, T.int64(0), T.int64(0)], lv74[v0, v1, v2, v3])
                            T.writes(compute_intermediate[v0, v1, v2, v3])
                            compute_intermediate[v0, v1, v2, v3] = T.max(conv2d_nchw_intermediate_local[v0, v1, v2, v3] + param_1[v0, v1, T.int64(0), T.int64(0)] + lv74[v0, v1, v2, v3], T.float32(0.0))
