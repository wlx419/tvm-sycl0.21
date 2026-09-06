	.text
	.amdgcn_target "amdgcn-amd-amdhsa-hcc-gfx906"
	.globl	layer_norm_kernel
	.p2align	8
	.type	layer_norm_kernel,@function
layer_norm_kernel:
	s_load_dwordx2 s[0:1], s[4:5], 0x18
	v_lshl_add_u32 v1, s6, 6, v0
	v_ashrrev_i32_e32 v2, 31, v1
	v_lshlrev_b64 v[3:4], 2, v[1:2]
	s_waitcnt lgkmcnt(0)
	s_barrier 
	v_mov_b32_e32 v5, s1
	v_add_co_u32_e32 v3, vcc, s0, v3
	v_addc_co_u32_e32 v4, vcc, v5, v4, vcc
	global_load_dword v3, v[3:4], off
	v_lshlrev_b32_e32 v4, 2, v0
	v_add_u32_e32 v5, 0x100, v4
	v_cmp_gt_i32_e32 vcc, 32, v0
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v6, 0, v3
	v_fma_f32 v7, v3, v3, 0
	ds_write_b32 v4, v6
	ds_write_b32 v4, v7 offset:256
	s_waitcnt lgkmcnt(0)
	s_barrier 
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_2
	ds_read_b32 v6, v4 offset:128
	ds_read_b32 v7, v4
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v6, v6, v7
	ds_write_b32 v4, v6
	ds_read_b32 v6, v5 offset:128
	ds_read_b32 v7, v5
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v6, v6, v7
	ds_write_b32 v5, v6
.LBB0_2:
	s_or_b64 exec, exec, s[0:1]
	s_load_dwordx2 s[0:1], s[4:5], 0x0
	v_cmp_gt_i32_e32 vcc, 16, v0
	s_waitcnt lgkmcnt(0)
	s_barrier 
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_4
	ds_read_b32 v6, v5 offset:64
	ds_read_b32 v7, v5
	ds_read_b32 v8, v4 offset:64
	ds_read_b32 v9, v4
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v6, v6, v7
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v7, v8, v9
	ds_write_b32 v4, v7
	ds_write_b32 v5, v6
	ds_read_b32 v6, v5 offset:32
	ds_read_b32 v7, v5
	ds_read_b32 v8, v4 offset:32
	ds_read_b32 v9, v4
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v6, v6, v7
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v7, v8, v9
	ds_write_b32 v4, v7
	ds_write_b32 v5, v6
	ds_read_b32 v6, v5 offset:16
	ds_read_b32 v7, v5
	ds_read_b32 v8, v4 offset:16
	ds_read_b32 v9, v4
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v6, v6, v7
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v7, v8, v9
	ds_write_b32 v4, v7
	ds_write_b32 v5, v6
	ds_read_b32 v6, v5 offset:8
	ds_read_b32 v7, v5
	ds_read_b32 v8, v4 offset:8
	ds_read_b32 v9, v4
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v6, v6, v7
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v7, v8, v9
	ds_write_b32 v4, v7
	ds_write_b32 v5, v6
	ds_read_b32 v6, v5 offset:4
	ds_read_b32 v7, v5
	ds_read_b32 v8, v4 offset:4
	ds_read_b32 v9, v4
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v6, v6, v7
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v7, v8, v9
	ds_write_b32 v4, v7
	ds_write_b32 v5, v6
.LBB0_4:
	s_or_b64 exec, exec, s[2:3]
	s_load_dwordx2 s[2:3], s[4:5], 0x8
	s_load_dwordx2 s[6:7], s[4:5], 0x10
	v_cmp_eq_u32_e32 vcc, 0, v0
	s_waitcnt lgkmcnt(0)
	s_barrier 
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_6
	v_mov_b32_e32 v6, 0
	ds_read_b32 v4, v6
	ds_read_b32 v5, v6 offset:256
	s_waitcnt lgkmcnt(0)
	ds_write_b64 v6, v[4:5] offset:512
.LBB0_6:
	s_or_b64 exec, exec, s[4:5]
	v_lshlrev_b32_e32 v0, 2, v0
	global_load_dword v6, v0, s[6:7]
	global_load_dword v7, v0, s[2:3]
	v_mov_b32_e32 v0, 0
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier 
	ds_read_b64 v[4:5], v0 offset:512
	s_mov_b32 s2, 0x3c800000
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v0, 0x3c800000, v4
	v_mul_f32_e32 v0, v0, v0
	v_fma_f32 v0, v5, s2, -v0
	v_add_f32_e32 v0, 0x3727c5ac, v0
	v_sqrt_f32_e32 v5, v0
	v_fmac_f32_e32 v3, 0xbc800000, v4
	v_div_scale_f32 v0, s[2:3], v5, v5, 1.0
	v_div_scale_f32 v8, vcc, 1.0, v5, 1.0
	v_rcp_f32_e32 v9, v0
	v_fma_f32 v10, -v0, v9, 1.0
	v_fmac_f32_e32 v9, v10, v9
	v_mul_f32_e32 v10, v8, v9
	v_fma_f32 v11, -v0, v10, v8
	v_fmac_f32_e32 v10, v11, v9
	v_fma_f32 v0, -v0, v10, v8
	v_div_fmas_f32 v8, v0, v9, v10
	v_lshlrev_b64 v[0:1], 2, v[1:2]
	v_mov_b32_e32 v2, s1
	v_add_co_u32_e32 v0, vcc, s0, v0
	v_addc_co_u32_e32 v1, vcc, v2, v1, vcc
	v_div_fixup_f32 v4, v8, v5, 1.0
	v_mul_f32_e32 v3, v3, v4
	v_fmac_f32_e32 v7, v3, v6
	global_store_dword v[0:1], v7, off
	s_endpgm
	.section	.rodata,#alloc
	.p2align	6
	.amdhsa_kernel layer_norm_kernel
		.amdhsa_group_segment_fixed_size 520
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 32
		.amdhsa_user_sgpr_count 6
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 12
		.amdhsa_next_free_sgpr 8
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	layer_norm_kernel, .Lfunc_end0-layer_norm_kernel

	.section	.text.tvm.fp16.conv,#alloc,#execinstr
	.weak	__truncsfhf2
	.p2align	2
	.type	__truncsfhf2,@function
__truncsfhf2:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	v_and_b32_e32 v2, 0x7fffffff, v0
	v_add_u32_e32 v1, 0xc7800000, v2
	v_add_u32_e32 v3, 0xb8800000, v2
	v_cmp_ge_u32_e32 vcc, v1, v3
	s_and_saveexec_b64 s[4:5], vcc
	s_xor_b64 s[4:5], exec, s[4:5]
	s_cbranch_execz .LBB1_14
	s_mov_b32 s6, 0x7f800001
	v_cmp_gt_u32_e32 vcc, s6, v2
	s_and_saveexec_b64 s[6:7], vcc
	s_xor_b64 s[6:7], exec, s[6:7]
	s_cbranch_execz .LBB1_11
	s_mov_b32 s8, 0x47800000
	v_cmp_gt_u32_e32 vcc, s8, v2
	v_mov_b32_e32 v1, 0x7c00
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB1_10
	s_mov_b32 s10, 0x2cffffff
	v_cmp_lt_u32_e32 vcc, s10, v2
	v_mov_b32_e32 v1, 0
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB1_9
	v_lshrrev_b32_e32 v1, 23, v2
	v_and_b32_e32 v3, 0x7fffff, v0
	v_sub_u32_e32 v2, 0x71, v1
	v_or_b32_e32 v3, 0x800000, v3
	v_add_u32_e32 v1, 0xffffffaf, v1
	v_lshlrev_b32_e32 v1, v1, v3
	v_cmp_ne_u32_e32 vcc, 0, v1
	v_lshrrev_b32_e32 v3, v2, v3
	v_cndmask_b32_e64 v1, 0, 1, vcc
	s_movk_i32 s12, 0x1fff
	v_and_or_b32 v4, v3, s12, v1
	s_movk_i32 s12, 0x1001
	v_lshrrev_b32_e32 v2, 13, v3
	v_cmp_gt_u32_e32 vcc, s12, v4
	s_and_saveexec_b64 s[12:13], vcc
	s_xor_b64 s[12:13], exec, s[12:13]
	s_movk_i32 s14, 0x1000
	v_bfe_u32 v1, v3, 13, 1
	v_cmp_eq_u32_e32 vcc, s14, v4
	v_cndmask_b32_e32 v1, 0, v1, vcc
	v_add_u32_e32 v1, v1, v2
	s_andn2_saveexec_b64 s[12:13], s[12:13]
	v_add_u32_e32 v1, 1, v2
	s_or_b64 exec, exec, s[12:13]
.LBB1_9:
	s_or_b64 exec, exec, s[10:11]
.LBB1_10:
	s_or_b64 exec, exec, s[8:9]
.LBB1_11:
	s_andn2_saveexec_b64 s[6:7], s[6:7]
	v_bfe_u32 v1, v0, 13, 9
	v_or_b32_e32 v1, 0x7e00, v1
	s_or_b64 exec, exec, s[6:7]
.LBB1_14:
	s_andn2_saveexec_b64 s[4:5], s[4:5]
	s_cbranch_execz .LBB1_20
	v_and_b32_e32 v3, 0x1fff, v0
	s_movk_i32 s6, 0x1001
	v_bfe_u32 v2, v0, 13, 16
	v_cmp_gt_u32_e32 vcc, s6, v3
	s_and_saveexec_b64 s[6:7], vcc
	s_xor_b64 s[6:7], exec, s[6:7]
	v_add_u32_e32 v1, 0xfffe4000, v2
	s_movk_i32 s8, 0x1000
	v_bfe_u32 v2, v0, 13, 1
	v_add_u32_sdwa v2, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_cmp_eq_u32_e32 vcc, s8, v3
	v_cndmask_b32_e32 v1, v1, v2, vcc
	s_andn2_saveexec_b64 s[6:7], s[6:7]
	v_add_u32_e32 v1, 0xfffe4001, v2
	s_or_b64 exec, exec, s[6:7]
.LBB1_20:
	s_or_b64 exec, exec, s[4:5]
	v_lshrrev_b32_e32 v0, 16, v0
	s_mov_b32 s4, 0x8000
	v_and_or_b32 v0, v0, s4, v1
	s_setpc_b64 s[30:31]
.Lfunc_end1:
	.size	__truncsfhf2, .Lfunc_end1-__truncsfhf2

	.weak	__extendhfsf2
	.p2align	2
	.type	__extendhfsf2,@function
__extendhfsf2:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	v_and_b32_e32 v2, 0x7fff, v0
	v_and_b32_e32 v1, 0xffff, v2
	v_add_u16_e32 v2, 0xfc00, v2
	s_movk_i32 s4, 0x77ff
	v_cmp_lt_u16_e32 vcc, s4, v2
	s_and_saveexec_b64 s[4:5], vcc
	s_xor_b64 s[4:5], exec, s[4:5]
	s_cbranch_execz .LBB2_8
	s_movk_i32 s6, 0x7c00
	v_cmp_gt_u16_e32 vcc, s6, v1
	s_and_saveexec_b64 s[6:7], vcc
	s_xor_b64 s[6:7], exec, s[6:7]
	s_cbranch_execz .LBB2_5
	v_cmp_ne_u16_e32 vcc, 0, v1
	v_mov_b32_e32 v2, 0
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB2_4
	s_movk_i32 s10, 0x100
	v_lshrrev_b32_e32 v2, 8, v1
	v_cmp_gt_u16_e32 vcc, s10, v1
	v_cndmask_b32_e32 v2, v2, v1, vcc
	v_cndmask_b32_e64 v3, 24, 32, vcc
	v_lshrrev_b32_e32 v4, 4, v2
	v_cmp_gt_u32_e32 vcc, 16, v2
	v_add_u32_e32 v5, -4, v3
	v_cndmask_b32_e32 v2, v4, v2, vcc
	v_cndmask_b32_e32 v3, v5, v3, vcc
	v_lshrrev_b32_e32 v4, 2, v2
	v_cmp_gt_u32_e32 vcc, 4, v2
	v_add_u32_e32 v5, -2, v3
	v_cndmask_b32_e32 v2, v4, v2, vcc
	v_cndmask_b32_e32 v3, v5, v3, vcc
	v_sub_u32_e32 v4, 0, v2
	v_cmp_gt_u32_e32 vcc, 2, v2
	v_cndmask_b32_e32 v2, -2, v4, vcc
	v_add_u32_e32 v2, v2, v3
	v_add_u32_e32 v3, -8, v2
	v_lshlrev_b32_e32 v1, v3, v1
	v_lshlrev_b32_e32 v2, 23, v2
	v_xor_b32_e32 v1, 0x800000, v1
	v_sub_u32_e32 v2, 0x43000000, v2
	v_or_b32_e32 v2, v1, v2
.LBB2_4:
	s_or_b64 exec, exec, s[8:9]
.LBB2_5:
	s_andn2_saveexec_b64 s[6:7], s[6:7]
	v_mov_b32_e32 v2, 0x7f800000
	v_lshl_or_b32 v2, v1, 13, v2
	s_or_b64 exec, exec, s[6:7]
.LBB2_8:
	s_andn2_saveexec_b64 s[4:5], s[4:5]
	v_bfrev_b32_e32 v2, 28
	v_lshl_add_u32 v2, v1, 13, v2
	s_or_b64 exec, exec, s[4:5]
	s_movk_i32 s4, 0x8000
	v_and_b32_sdwa v0, v0, s4 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_or_b32_e32 v0, v2, v0
	s_setpc_b64 s[30:31]
.Lfunc_end2:
	.size	__extendhfsf2, .Lfunc_end2-__extendhfsf2

	.ident	"clang version 15.0.0 (http://10.15.3.7/dcutoolkit/driverruntime/llvm-project.git 340750feeda88c9c2ce8ad481b11d9aa7f033d39)"
	.section	".note.GNU-stack"
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .name:           T_layer_norm
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .access:         read_only
        .address_space:  global
        .name:           ln1.bias
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .access:         read_only
        .address_space:  global
        .name:           ln1.weight
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .access:         read_only
        .address_space:  global
        .name:           x
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 520
    .kernarg_segment_align: 8
    .kernarg_segment_size: 32
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           layer_norm_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         layer_norm_kernel.kd
    .uses_dynamic_stack: false
    .vgpr_count:     12
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa-hcc-gfx906
amdhsa.version:
  - 1
  - 1
...

	.end_amdgpu_metadata
