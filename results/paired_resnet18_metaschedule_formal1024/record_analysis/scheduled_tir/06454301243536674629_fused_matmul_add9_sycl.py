# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(lv87: T.Buffer((T.int64(1), T.int64(512)), "float32"), param_0: T.Buffer((T.int64(512), T.int64(1000)), "float32"), param_1: T.Buffer((T.int64(1000),), "float32"), T_add_intermediate: T.Buffer((T.int64(1), T.int64(1000)), "float32")):
        T.func_attr({"tir.noalias": True})
        # with T.block("root"):
        matmul_intermediate_local = T.alloc_buffer((T.int64(1), T.int64(1000)), scope="local")
        lv87_shared = T.alloc_buffer((T.int64(1), T.int64(512)), scope="shared")
        param_0_shared = T.alloc_buffer((T.int64(512), T.int64(1000)), scope="shared")
        for i0_0_i1_0_fused in T.thread_binding(T.int64(5), thread="blockIdx.x"):
            for i0_1_i1_1_fused in T.thread_binding(T.int64(1), thread="vthread.x"):
                for i0_2_i1_2_fused in T.thread_binding(T.int64(100), thread="threadIdx.x"):
                    for i0_3_init, i1_3_init, i0_4_init, i1_4_init in T.grid(T.int64(1), T.int64(1), T.int64(1), T.int64(2)):
                        with T.block("matmul_init"):
                            v_i0 = T.axis.spatial(T.int64(1), i0_3_init + i0_4_init)
                            v_i1 = T.axis.spatial(T.int64(1000), i0_0_i1_0_fused * T.int64(200) + i0_2_i1_2_fused * T.int64(2) + i1_3_init * T.int64(2) + i1_4_init)
                            T.reads()
                            T.writes(matmul_intermediate_local[v_i0, v_i1])
                            T.block_attr({"meta_schedule.thread_extent_high_inclusive": 1024, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                            matmul_intermediate_local[v_i0, v_i1] = T.float32(0.0)
                    for k_0 in range(T.int64(128)):
                        for ax0_ax1_fused_0 in range(T.int64(1)):
                            for ax0_ax1_fused_1 in T.thread_binding(T.int64(100), thread="threadIdx.x"):
                                for ax0_ax1_fused_2 in T.vectorized(T.int64(4)):
                                    with T.block("lv87_shared"):
                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                        v1 = T.axis.spatial(T.int64(512), k_0 * T.int64(4) + (ax0_ax1_fused_0 * T.int64(400) + ax0_ax1_fused_1 * T.int64(4) + ax0_ax1_fused_2))
                                        T.where((ax0_ax1_fused_0 * T.int64(100) + ax0_ax1_fused_1) * T.int64(4) + ax0_ax1_fused_2 < T.int64(4))
                                        T.reads(lv87[v0, v1])
                                        T.writes(lv87_shared[v0, v1])
                                        lv87_shared[v0, v1] = lv87[v0, v1]
                        for ax0_ax1_fused_0 in range(T.int64(2)):
                            for ax0_ax1_fused_1 in T.thread_binding(T.int64(100), thread="threadIdx.x"):
                                for ax0_ax1_fused_2 in T.vectorized(T.int64(4)):
                                    with T.block("param_0_shared"):
                                        v0 = T.axis.spatial(T.int64(512), k_0 * T.int64(4) + (ax0_ax1_fused_0 * T.int64(400) + ax0_ax1_fused_1 * T.int64(4) + ax0_ax1_fused_2) // T.int64(200))
                                        v1 = T.axis.spatial(T.int64(1000), i0_0_i1_0_fused * T.int64(200) + (ax0_ax1_fused_0 * T.int64(400) + ax0_ax1_fused_1 * T.int64(4) + ax0_ax1_fused_2) % T.int64(200))
                                        T.reads(param_0[v0, v1])
                                        T.writes(param_0_shared[v0, v1])
                                        param_0_shared[v0, v1] = param_0[v0, v1]
                        for k_1, i0_3, i1_3, k_2, i0_4, i1_4 in T.grid(T.int64(2), T.int64(1), T.int64(1), T.int64(2), T.int64(1), T.int64(2)):
                            with T.block("matmul_update"):
                                v_i0 = T.axis.spatial(T.int64(1), i0_3 + i0_4)
                                v_i1 = T.axis.spatial(T.int64(1000), i0_0_i1_0_fused * T.int64(200) + i0_2_i1_2_fused * T.int64(2) + i1_3 * T.int64(2) + i1_4)
                                v_k = T.axis.reduce(T.int64(512), k_0 * T.int64(4) + k_1 * T.int64(2) + k_2)
                                T.reads(matmul_intermediate_local[v_i0, v_i1], lv87_shared[v_i0, v_k], param_0_shared[v_k, v_i1])
                                T.writes(matmul_intermediate_local[v_i0, v_i1])
                                T.block_attr({"meta_schedule.thread_extent_high_inclusive": 1024, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                                matmul_intermediate_local[v_i0, v_i1] = matmul_intermediate_local[v_i0, v_i1] + lv87_shared[v_i0, v_k] * param_0_shared[v_k, v_i1]
                    for ax0, ax1 in T.grid(T.int64(1), T.int64(2)):
                        with T.block("matmul_intermediate_local"):
                            v0 = T.axis.spatial(T.int64(1), ax0)
                            v1 = T.axis.spatial(T.int64(1000), i0_0_i1_0_fused * T.int64(200) + i0_2_i1_2_fused * T.int64(2) + ax1)
                            T.reads(matmul_intermediate_local[v0, v1], param_1[v1])
                            T.writes(T_add_intermediate[v0, v1])
                            T_add_intermediate[v0, v1] = matmul_intermediate_local[v0, v1] + param_1[v1]
