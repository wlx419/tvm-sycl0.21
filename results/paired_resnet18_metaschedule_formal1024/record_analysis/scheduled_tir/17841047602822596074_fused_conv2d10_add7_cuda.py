# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(lv64: T.Buffer((T.int64(1), T.int64(256), T.int64(14), T.int64(14)), "float32"), param_0: T.Buffer((T.int64(512), T.int64(256), T.int64(1), T.int64(1)), "float32"), param_1: T.Buffer((T.int64(1), T.int64(512), T.int64(1), T.int64(1)), "float32"), T_add_intermediate: T.Buffer((T.int64(1), T.int64(512), T.int64(7), T.int64(7)), "float32")):
        T.func_attr({"tir.noalias": True})
        # with T.block("root"):
        conv2d_nchw_intermediate_local = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(7), T.int64(7)), scope="local")
        pad_temp_shared = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(14), T.int64(14)), scope="shared")
        param_0_shared = T.alloc_buffer((T.int64(512), T.int64(256), T.int64(1), T.int64(1)), scope="shared")
        for nn_0_ff_0_yy_0_xx_0_fused in T.thread_binding(T.int64(28), thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step": 16, "pragma_unroll_explicit": 1}):
            for nn_1_ff_1_yy_1_xx_1_fused in T.thread_binding(T.int64(1), thread="vthread.x"):
                for nn_2_ff_2_yy_2_xx_2_fused in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                    for nn_3_init, ff_3_init, yy_3_init, xx_3_init, nn_4_init, ff_4_init, yy_4_init, xx_4_init in T.grid(T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(7)):
                        with T.block("conv2d_nchw_init"):
                            v_nn = T.axis.spatial(T.int64(1), nn_3_init + nn_4_init)
                            v_ff = T.axis.spatial(T.int64(512), nn_0_ff_0_yy_0_xx_0_fused // T.int64(7) * T.int64(128) + nn_2_ff_2_yy_2_xx_2_fused + ff_3_init + ff_4_init)
                            v_yy = T.axis.spatial(T.int64(7), nn_0_ff_0_yy_0_xx_0_fused % T.int64(7) + yy_3_init + yy_4_init)
                            v_xx = T.axis.spatial(T.int64(7), xx_3_init * T.int64(7) + xx_4_init)
                            T.reads()
                            T.writes(conv2d_nchw_intermediate_local[v_nn, v_ff, v_yy, v_xx])
                            T.block_attr({"meta_schedule.thread_extent_high_inclusive": 1024, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                            conv2d_nchw_intermediate_local[v_nn, v_ff, v_yy, v_xx] = T.float32(0.0)
                    for rc_0_ry_0_rx_0_fused in T.serial(T.int64(16), annotations={"software_pipeline_async_stages": [0], "software_pipeline_order": [0, 1, 2], "software_pipeline_stage": [0, 0, 3]}):
                        for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(1)):
                            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                                for ax0_ax1_ax2_ax3_fused_2 in T.vectorized(T.int64(4)):
                                    with T.block("pad_temp_shared"):
                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v1 = T.axis.spatial(T.int64(256), rc_0_ry_0_rx_0_fused * T.int64(16) + (ax0_ax1_ax2_ax3_fused_0 * T.int64(512) + ax0_ax1_ax2_ax3_fused_1 * T.int64(4) + ax0_ax1_ax2_ax3_fused_2) // T.int64(13))
                                        v2 = T.axis.spatial(T.int64(14), nn_0_ff_0_yy_0_xx_0_fused % T.int64(7) * T.int64(2))
                                        v3 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(512) + ax0_ax1_ax2_ax3_fused_1 * T.int64(4) + ax0_ax1_ax2_ax3_fused_2) % T.int64(13))
                                        T.where((ax0_ax1_ax2_ax3_fused_0 * T.int64(128) + ax0_ax1_ax2_ax3_fused_1) * T.int64(4) + ax0_ax1_ax2_ax3_fused_2 < T.int64(208))
                                        T.reads(lv64[v0, v1, v2, v3])
                                        T.writes(pad_temp_shared[v0, v1, v2, v3])
                                        pad_temp_shared[v0, v1, v2, v3] = lv64[v0, v1, v2, v3]
                        for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(16)):
                            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                                with T.block("param_0_shared"):
                                    v0 = T.axis.spatial(T.int64(512), nn_0_ff_0_yy_0_xx_0_fused // T.int64(7) * T.int64(128) + (ax0_ax1_ax2_ax3_fused_0 * T.int64(128) + ax0_ax1_ax2_ax3_fused_1) // T.int64(16))
                                    v1 = T.axis.spatial(T.int64(256), rc_0_ry_0_rx_0_fused * T.int64(16) + (ax0_ax1_ax2_ax3_fused_0 * T.int64(128) + ax0_ax1_ax2_ax3_fused_1) % T.int64(16))
                                    v2 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v3 = T.axis.spatial(T.int64(1), T.int64(0))
                                    T.reads(param_0[v0, v1, v2, v3])
                                    T.writes(param_0_shared[v0, v1, v2, v3])
                                    param_0_shared[v0, v1, v2, v3] = param_0[v0, v1, v2, v3]
                        for rc_1, ry_1, rx_1, nn_3, ff_3, yy_3, xx_3, rc_2, ry_2, rx_2, nn_4, ff_4, yy_4, xx_4 in T.grid(T.int64(2), T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(8), T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(7)):
                            with T.block("conv2d_nchw_update"):
                                v_nn = T.axis.spatial(T.int64(1), nn_3 + nn_4)
                                v_ff = T.axis.spatial(T.int64(512), nn_0_ff_0_yy_0_xx_0_fused // T.int64(7) * T.int64(128) + nn_2_ff_2_yy_2_xx_2_fused + ff_3 + ff_4)
                                v_yy = T.axis.spatial(T.int64(7), nn_0_ff_0_yy_0_xx_0_fused % T.int64(7) + yy_3 + yy_4)
                                v_xx = T.axis.spatial(T.int64(7), xx_3 * T.int64(7) + xx_4)
                                v_rc = T.axis.reduce(T.int64(256), rc_0_ry_0_rx_0_fused * T.int64(16) + rc_1 * T.int64(8) + rc_2)
                                v_ry = T.axis.reduce(T.int64(1), ry_1 + ry_2)
                                v_rx = T.axis.reduce(T.int64(1), rx_1 + rx_2)
                                T.reads(conv2d_nchw_intermediate_local[v_nn, v_ff, v_yy, v_xx], pad_temp_shared[v_nn, v_rc, v_yy * T.int64(2) + v_ry, v_xx * T.int64(2) + v_rx], param_0_shared[v_ff, v_rc, v_ry, v_rx])
                                T.writes(conv2d_nchw_intermediate_local[v_nn, v_ff, v_yy, v_xx])
                                T.block_attr({"meta_schedule.thread_extent_high_inclusive": 1024, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                                conv2d_nchw_intermediate_local[v_nn, v_ff, v_yy, v_xx] = conv2d_nchw_intermediate_local[v_nn, v_ff, v_yy, v_xx] + pad_temp_shared[v_nn, v_rc, v_yy * T.int64(2) + v_ry, v_xx * T.int64(2) + v_rx] * param_0_shared[v_ff, v_rc, v_ry, v_rx]
                    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(1), T.int64(1), T.int64(7)):
                        with T.block("conv2d_nchw_intermediate_local"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(512), nn_0_ff_0_yy_0_xx_0_fused // T.int64(7) * T.int64(128) + nn_2_ff_2_yy_2_xx_2_fused + ax1)
                            v2 = T.axis.spatial(T.int64(7), nn_0_ff_0_yy_0_xx_0_fused % T.int64(7) + ax2)
                            v3 = T.axis.spatial(T.int64(7), ax3)
                            T.reads(conv2d_nchw_intermediate_local[v0, v1, v2, v3], param_1[v0, v1, T.int64(0), T.int64(0)])
                            T.writes(T_add_intermediate[v0, v1, v2, v3])
                            T_add_intermediate[v0, v1, v2, v3] = conv2d_nchw_intermediate_local[v0, v1, v2, v3] + param_1[v0, v1, T.int64(0), T.int64(0)]
