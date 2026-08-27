// tvm target: sycl -keys=sycl,gpu -l2_cache_size_bytes=75497472 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -texture_spatial_limit=16384 -thread_warp_size=32
#include <sycl/sycl.hpp>
using namespace sycl;
#ifndef TVM_SYCL_MODULE_ID
#define TVM_SYCL_MODULE_ID m0
#endif
#define TVM_SYCL_KERNEL_NAME_IMPL(func, module_id) func##_##module_id
#define TVM_SYCL_KERNEL_NAME(func, module_id) TVM_SYCL_KERNEL_NAME_IMPL(func, module_id)
#define TVM_SYCL_WRAPPER_NAME_IMPL(func, module_id) w_##func##_##module_id
#define TVM_SYCL_WRAPPER_NAME(func, module_id) TVM_SYCL_WRAPPER_NAME_IMPL(func, module_id)
// CodeGenSYCL: NOTE: Auto-generated packed function
// Function: main_kernel
class TVM_SYCL_KERNEL_NAME(h874766378, TVM_SYCL_MODULE_ID);
#ifdef __cplusplus
extern "C"
#endif
void TVM_SYCL_WRAPPER_NAME(h874766378, TVM_SYCL_MODULE_ID)(queue &Qh, sycl::range<3> dimGrid, sycl::range<3> dimBlock, void** void_args) {
  float* compute = (float*)(*(int64_t *)(void_args[0]));
  float* param_0 = (float*)(*(int64_t *)(void_args[1]));
  float* param_1 = (float*)(*(int64_t *)(void_args[2]));
  float* x = (float*)(*(int64_t *)(void_args[3]));
  Qh.submit([=](handler &h) {
    h.parallel_for<TVM_SYCL_KERNEL_NAME(h874766378, TVM_SYCL_MODULE_ID)>(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](sycl::nd_item<3> item) [[intel::kernel_args_restrict]] {
      float conv2d_nchw_local[32];
      auto pad_temp_shared = *sycl::ext::oneapi::group_local_memory<float[147]>(item.get_group());
      auto param_0_shared = *sycl::ext::oneapi::group_local_memory<float[224]>(item.get_group());
      conv2d_nchw_local[0] = 0.000000e+00f;
      conv2d_nchw_local[8] = 0.000000e+00f;
      conv2d_nchw_local[16] = 0.000000e+00f;
      conv2d_nchw_local[24] = 0.000000e+00f;
      conv2d_nchw_local[1] = 0.000000e+00f;
      conv2d_nchw_local[9] = 0.000000e+00f;
      conv2d_nchw_local[17] = 0.000000e+00f;
      conv2d_nchw_local[25] = 0.000000e+00f;
      conv2d_nchw_local[2] = 0.000000e+00f;
      conv2d_nchw_local[10] = 0.000000e+00f;
      conv2d_nchw_local[18] = 0.000000e+00f;
      conv2d_nchw_local[26] = 0.000000e+00f;
      conv2d_nchw_local[3] = 0.000000e+00f;
      conv2d_nchw_local[11] = 0.000000e+00f;
      conv2d_nchw_local[19] = 0.000000e+00f;
      conv2d_nchw_local[27] = 0.000000e+00f;
      conv2d_nchw_local[4] = 0.000000e+00f;
      conv2d_nchw_local[12] = 0.000000e+00f;
      conv2d_nchw_local[20] = 0.000000e+00f;
      conv2d_nchw_local[28] = 0.000000e+00f;
      conv2d_nchw_local[5] = 0.000000e+00f;
      conv2d_nchw_local[13] = 0.000000e+00f;
      conv2d_nchw_local[21] = 0.000000e+00f;
      conv2d_nchw_local[29] = 0.000000e+00f;
      conv2d_nchw_local[6] = 0.000000e+00f;
      conv2d_nchw_local[14] = 0.000000e+00f;
      conv2d_nchw_local[22] = 0.000000e+00f;
      conv2d_nchw_local[30] = 0.000000e+00f;
      conv2d_nchw_local[7] = 0.000000e+00f;
      conv2d_nchw_local[15] = 0.000000e+00f;
      conv2d_nchw_local[23] = 0.000000e+00f;
      conv2d_nchw_local[31] = 0.000000e+00f;
      for (int rc_0 = 0; rc_0 < 3; ++rc_0) {
        for (int ry_0 = 0; ry_0 < 7; ++ry_0) {
          group_barrier(item.get_group());
          float condval;
          if ((((bool)(((bool)(3 <= (((((((int)item.get_group(0)) % 392) / 14) * 8) + (((int)item.get_local_id(2)) / 21)) + ry_0))) && ((bool)(3 <= (((((int)item.get_group(0)) % 14) * 16) + (((int)item.get_local_id(2)) % 21)))))) && ((bool)((((((int)item.get_group(0)) % 14) * 16) + (((int)item.get_local_id(2)) % 21)) < 227)))) {
            condval = x[(((((((rc_0 * 50176) + (((((int)item.get_group(0)) % 392) / 14) * 1792)) + ((((int)item.get_local_id(2)) / 21) * 224)) + (ry_0 * 224)) + ((((int)item.get_group(0)) % 14) * 16)) + (((int)item.get_local_id(2)) % 21)) - 675)];
          } else {
            condval = 0.000000e+00f;
          }
          pad_temp_shared[((int)item.get_local_id(2))] = condval;
          float condval_1;
          if ((((bool)(((bool)(3 <= (((((((int)item.get_group(0)) % 392) / 14) * 8) + ((((int)item.get_local_id(2)) + 32) / 21)) + ry_0))) && ((bool)(3 <= (((((int)item.get_group(0)) % 14) * 16) + ((((int)item.get_local_id(2)) + 11) % 21)))))) && ((bool)((((((int)item.get_group(0)) % 14) * 16) + ((((int)item.get_local_id(2)) + 11) % 21)) < 227)))) {
            condval_1 = x[(((((((rc_0 * 50176) + (((((int)item.get_group(0)) % 392) / 14) * 1792)) + (((((int)item.get_local_id(2)) + 32) / 21) * 224)) + (ry_0 * 224)) + ((((int)item.get_group(0)) % 14) * 16)) + ((((int)item.get_local_id(2)) + 11) % 21)) - 675)];
          } else {
            condval_1 = 0.000000e+00f;
          }
          pad_temp_shared[((((((int)item.get_local_id(2)) + 32) / 21) * 21) + ((((int)item.get_local_id(2)) + 11) % 21))] = condval_1;
          float condval_2;
          if ((((bool)(3 <= (((((int)item.get_group(0)) % 14) * 16) + ((((int)item.get_local_id(2)) + 1) % 21)))) && ((bool)((((((int)item.get_group(0)) % 14) * 16) + ((((int)item.get_local_id(2)) + 1) % 21)) < 227)))) {
            condval_2 = x[(((((((rc_0 * 50176) + (((((int)item.get_group(0)) % 392) / 14) * 1792)) + (((((int)item.get_local_id(2)) + 64) / 21) * 224)) + (ry_0 * 224)) + ((((int)item.get_group(0)) % 14) * 16)) + ((((int)item.get_local_id(2)) + 1) % 21)) - 675)];
          } else {
            condval_2 = 0.000000e+00f;
          }
          pad_temp_shared[((((((int)item.get_local_id(2)) + 64) / 21) * 21) + ((((int)item.get_local_id(2)) + 1) % 21))] = condval_2;
          float condval_3;
          if ((((bool)(((bool)((((((((int)item.get_group(0)) % 392) / 14) * 8) + ((((int)item.get_local_id(2)) + 96) / 21)) + ry_0) < 227)) && ((bool)(3 <= (((((int)item.get_group(0)) % 14) * 16) + ((((int)item.get_local_id(2)) + 12) % 21)))))) && ((bool)((((((int)item.get_group(0)) % 14) * 16) + ((((int)item.get_local_id(2)) + 12) % 21)) < 227)))) {
            condval_3 = x[(((((((rc_0 * 50176) + (((((int)item.get_group(0)) % 392) / 14) * 1792)) + (((((int)item.get_local_id(2)) + 96) / 21) * 224)) + (ry_0 * 224)) + ((((int)item.get_group(0)) % 14) * 16)) + ((((int)item.get_local_id(2)) + 12) % 21)) - 675)];
          } else {
            condval_3 = 0.000000e+00f;
          }
          pad_temp_shared[((((((int)item.get_local_id(2)) + 96) / 21) * 21) + ((((int)item.get_local_id(2)) + 12) % 21))] = condval_3;
          if (((int)item.get_local_id(2)) < 19) {
            float condval_4;
            if ((((bool)(((bool)((((((((int)item.get_group(0)) % 392) / 14) * 8) + ((((int)item.get_local_id(2)) + 128) / 21)) + ry_0) < 227)) && ((bool)(1 <= (((((int)item.get_group(0)) % 14) * 16) + ((int)item.get_local_id(2))))))) && ((bool)((((((int)item.get_group(0)) % 14) * 16) + ((int)item.get_local_id(2))) < 225)))) {
              condval_4 = x[(((((((rc_0 * 50176) + (((((int)item.get_group(0)) % 392) / 14) * 1792)) + (((((int)item.get_local_id(2)) + 128) / 21) * 224)) + (ry_0 * 224)) + ((((int)item.get_group(0)) % 14) * 16)) + ((int)item.get_local_id(2))) - 673)];
            } else {
              condval_4 = 0.000000e+00f;
            }
            pad_temp_shared[(((((((int)item.get_local_id(2)) + 128) / 21) * 21) + ((int)item.get_local_id(2))) + 2)] = condval_4;
          }
          param_0_shared[((int)item.get_local_id(2))] = param_0[((((((((int)item.get_group(0)) / 392) * 4704) + ((((int)item.get_local_id(2)) / 7) * 147)) + (rc_0 * 49)) + (ry_0 * 7)) + (((int)item.get_local_id(2)) % 7))];
          param_0_shared[(((int)item.get_local_id(2)) + 32)] = param_0[((((((((int)item.get_group(0)) / 392) * 4704) + (((((int)item.get_local_id(2)) + 32) / 7) * 147)) + (rc_0 * 49)) + (ry_0 * 7)) + ((((int)item.get_local_id(2)) + 4) % 7))];
          param_0_shared[(((int)item.get_local_id(2)) + 64)] = param_0[((((((((int)item.get_group(0)) / 392) * 4704) + (((((int)item.get_local_id(2)) + 64) / 7) * 147)) + (rc_0 * 49)) + (ry_0 * 7)) + ((((int)item.get_local_id(2)) + 1) % 7))];
          param_0_shared[(((int)item.get_local_id(2)) + 96)] = param_0[((((((((int)item.get_group(0)) / 392) * 4704) + (((((int)item.get_local_id(2)) + 96) / 7) * 147)) + (rc_0 * 49)) + (ry_0 * 7)) + ((((int)item.get_local_id(2)) + 5) % 7))];
          param_0_shared[(((int)item.get_local_id(2)) + 128)] = param_0[((((((((int)item.get_group(0)) / 392) * 4704) + (((((int)item.get_local_id(2)) + 128) / 7) * 147)) + (rc_0 * 49)) + (ry_0 * 7)) + ((((int)item.get_local_id(2)) + 2) % 7))];
          param_0_shared[(((int)item.get_local_id(2)) + 160)] = param_0[((((((((int)item.get_group(0)) / 392) * 4704) + (((((int)item.get_local_id(2)) + 160) / 7) * 147)) + (rc_0 * 49)) + (ry_0 * 7)) + ((((int)item.get_local_id(2)) + 6) % 7))];
          param_0_shared[(((int)item.get_local_id(2)) + 192)] = param_0[((((((((int)item.get_group(0)) / 392) * 4704) + (((((int)item.get_local_id(2)) + 192) / 7) * 147)) + (rc_0 * 49)) + (ry_0 * 7)) + ((((int)item.get_local_id(2)) + 3) % 7))];
          group_barrier(item.get_group());
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)item.get_local_id(2)) & 7) * 2)] * param_0_shared[((((int)item.get_local_id(2)) >> 3) * 56)]));
          conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 42)] * param_0_shared[((((int)item.get_local_id(2)) >> 3) * 56)]));
          conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 84)] * param_0_shared[((((int)item.get_local_id(2)) >> 3) * 56)]));
          conv2d_nchw_local[24] = (conv2d_nchw_local[24] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 126)] * param_0_shared[((((int)item.get_local_id(2)) >> 3) * 56)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((int)item.get_local_id(2)) & 7) * 2)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 7)]));
          conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 42)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 7)]));
          conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 84)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 7)]));
          conv2d_nchw_local[25] = (conv2d_nchw_local[25] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 126)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 7)]));
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 1)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 1)]));
          conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 43)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 1)]));
          conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 85)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 1)]));
          conv2d_nchw_local[24] = (conv2d_nchw_local[24] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 127)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 1)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 1)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 8)]));
          conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 43)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 8)]));
          conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 85)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 8)]));
          conv2d_nchw_local[25] = (conv2d_nchw_local[25] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 127)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 8)]));
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 2)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 2)]));
          conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 44)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 2)]));
          conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 86)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 2)]));
          conv2d_nchw_local[24] = (conv2d_nchw_local[24] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 128)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 2)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 2)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 9)]));
          conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 44)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 9)]));
          conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 86)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 9)]));
          conv2d_nchw_local[25] = (conv2d_nchw_local[25] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 128)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 9)]));
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 3)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 3)]));
          conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 45)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 3)]));
          conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 87)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 3)]));
          conv2d_nchw_local[24] = (conv2d_nchw_local[24] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 129)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 3)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 3)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 10)]));
          conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 45)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 10)]));
          conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 87)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 10)]));
          conv2d_nchw_local[25] = (conv2d_nchw_local[25] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 129)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 10)]));
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 4)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 4)]));
          conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 46)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 4)]));
          conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 88)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 4)]));
          conv2d_nchw_local[24] = (conv2d_nchw_local[24] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 130)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 4)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 4)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 11)]));
          conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 46)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 11)]));
          conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 88)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 11)]));
          conv2d_nchw_local[25] = (conv2d_nchw_local[25] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 130)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 11)]));
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 5)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 5)]));
          conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 47)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 5)]));
          conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 89)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 5)]));
          conv2d_nchw_local[24] = (conv2d_nchw_local[24] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 131)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 5)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 5)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 12)]));
          conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 47)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 12)]));
          conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 89)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 12)]));
          conv2d_nchw_local[25] = (conv2d_nchw_local[25] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 131)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 12)]));
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 6)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 6)]));
          conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 48)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 6)]));
          conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 90)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 6)]));
          conv2d_nchw_local[24] = (conv2d_nchw_local[24] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 132)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 6)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 6)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 13)]));
          conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 48)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 13)]));
          conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 90)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 13)]));
          conv2d_nchw_local[25] = (conv2d_nchw_local[25] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 132)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 13)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((int)item.get_local_id(2)) & 7) * 2)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 14)]));
          conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 42)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 14)]));
          conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 84)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 14)]));
          conv2d_nchw_local[26] = (conv2d_nchw_local[26] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 126)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 14)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((int)item.get_local_id(2)) & 7) * 2)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 21)]));
          conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 42)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 21)]));
          conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 84)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 21)]));
          conv2d_nchw_local[27] = (conv2d_nchw_local[27] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 126)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 21)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 1)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 15)]));
          conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 43)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 15)]));
          conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 85)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 15)]));
          conv2d_nchw_local[26] = (conv2d_nchw_local[26] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 127)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 15)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 1)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 22)]));
          conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 43)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 22)]));
          conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 85)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 22)]));
          conv2d_nchw_local[27] = (conv2d_nchw_local[27] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 127)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 22)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 2)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 16)]));
          conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 44)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 16)]));
          conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 86)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 16)]));
          conv2d_nchw_local[26] = (conv2d_nchw_local[26] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 128)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 16)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 2)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 23)]));
          conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 44)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 23)]));
          conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 86)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 23)]));
          conv2d_nchw_local[27] = (conv2d_nchw_local[27] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 128)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 23)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 3)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 17)]));
          conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 45)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 17)]));
          conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 87)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 17)]));
          conv2d_nchw_local[26] = (conv2d_nchw_local[26] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 129)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 17)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 3)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 24)]));
          conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 45)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 24)]));
          conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 87)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 24)]));
          conv2d_nchw_local[27] = (conv2d_nchw_local[27] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 129)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 24)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 4)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 18)]));
          conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 46)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 18)]));
          conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 88)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 18)]));
          conv2d_nchw_local[26] = (conv2d_nchw_local[26] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 130)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 18)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 4)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 25)]));
          conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 46)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 25)]));
          conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 88)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 25)]));
          conv2d_nchw_local[27] = (conv2d_nchw_local[27] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 130)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 25)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 5)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 19)]));
          conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 47)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 19)]));
          conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 89)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 19)]));
          conv2d_nchw_local[26] = (conv2d_nchw_local[26] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 131)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 19)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 5)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 26)]));
          conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 47)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 26)]));
          conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 89)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 26)]));
          conv2d_nchw_local[27] = (conv2d_nchw_local[27] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 131)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 26)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 6)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 20)]));
          conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 48)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 20)]));
          conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 90)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 20)]));
          conv2d_nchw_local[26] = (conv2d_nchw_local[26] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 132)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 20)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 6)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 27)]));
          conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 48)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 27)]));
          conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 90)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 27)]));
          conv2d_nchw_local[27] = (conv2d_nchw_local[27] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 132)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 27)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[((((int)item.get_local_id(2)) & 7) * 2)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 28)]));
          conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 42)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 28)]));
          conv2d_nchw_local[20] = (conv2d_nchw_local[20] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 84)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 28)]));
          conv2d_nchw_local[28] = (conv2d_nchw_local[28] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 126)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 28)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[((((int)item.get_local_id(2)) & 7) * 2)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 35)]));
          conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 42)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 35)]));
          conv2d_nchw_local[21] = (conv2d_nchw_local[21] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 84)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 35)]));
          conv2d_nchw_local[29] = (conv2d_nchw_local[29] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 126)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 35)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 1)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 29)]));
          conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 43)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 29)]));
          conv2d_nchw_local[20] = (conv2d_nchw_local[20] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 85)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 29)]));
          conv2d_nchw_local[28] = (conv2d_nchw_local[28] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 127)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 29)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 1)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 36)]));
          conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 43)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 36)]));
          conv2d_nchw_local[21] = (conv2d_nchw_local[21] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 85)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 36)]));
          conv2d_nchw_local[29] = (conv2d_nchw_local[29] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 127)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 36)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 2)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 30)]));
          conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 44)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 30)]));
          conv2d_nchw_local[20] = (conv2d_nchw_local[20] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 86)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 30)]));
          conv2d_nchw_local[28] = (conv2d_nchw_local[28] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 128)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 30)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 2)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 37)]));
          conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 44)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 37)]));
          conv2d_nchw_local[21] = (conv2d_nchw_local[21] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 86)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 37)]));
          conv2d_nchw_local[29] = (conv2d_nchw_local[29] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 128)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 37)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 3)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 31)]));
          conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 45)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 31)]));
          conv2d_nchw_local[20] = (conv2d_nchw_local[20] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 87)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 31)]));
          conv2d_nchw_local[28] = (conv2d_nchw_local[28] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 129)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 31)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 3)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 38)]));
          conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 45)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 38)]));
          conv2d_nchw_local[21] = (conv2d_nchw_local[21] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 87)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 38)]));
          conv2d_nchw_local[29] = (conv2d_nchw_local[29] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 129)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 38)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 4)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 32)]));
          conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 46)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 32)]));
          conv2d_nchw_local[20] = (conv2d_nchw_local[20] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 88)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 32)]));
          conv2d_nchw_local[28] = (conv2d_nchw_local[28] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 130)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 32)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 4)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 39)]));
          conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 46)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 39)]));
          conv2d_nchw_local[21] = (conv2d_nchw_local[21] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 88)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 39)]));
          conv2d_nchw_local[29] = (conv2d_nchw_local[29] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 130)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 39)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 5)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 33)]));
          conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 47)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 33)]));
          conv2d_nchw_local[20] = (conv2d_nchw_local[20] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 89)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 33)]));
          conv2d_nchw_local[28] = (conv2d_nchw_local[28] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 131)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 33)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 5)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 40)]));
          conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 47)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 40)]));
          conv2d_nchw_local[21] = (conv2d_nchw_local[21] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 89)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 40)]));
          conv2d_nchw_local[29] = (conv2d_nchw_local[29] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 131)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 40)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 6)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 34)]));
          conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 48)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 34)]));
          conv2d_nchw_local[20] = (conv2d_nchw_local[20] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 90)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 34)]));
          conv2d_nchw_local[28] = (conv2d_nchw_local[28] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 132)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 34)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 6)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 41)]));
          conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 48)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 41)]));
          conv2d_nchw_local[21] = (conv2d_nchw_local[21] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 90)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 41)]));
          conv2d_nchw_local[29] = (conv2d_nchw_local[29] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 132)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 41)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[((((int)item.get_local_id(2)) & 7) * 2)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 42)]));
          conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 42)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 42)]));
          conv2d_nchw_local[22] = (conv2d_nchw_local[22] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 84)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 42)]));
          conv2d_nchw_local[30] = (conv2d_nchw_local[30] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 126)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 42)]));
          conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[((((int)item.get_local_id(2)) & 7) * 2)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 49)]));
          conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 42)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 49)]));
          conv2d_nchw_local[23] = (conv2d_nchw_local[23] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 84)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 49)]));
          conv2d_nchw_local[31] = (conv2d_nchw_local[31] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 126)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 49)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 1)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 43)]));
          conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 43)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 43)]));
          conv2d_nchw_local[22] = (conv2d_nchw_local[22] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 85)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 43)]));
          conv2d_nchw_local[30] = (conv2d_nchw_local[30] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 127)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 43)]));
          conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 1)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 50)]));
          conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 43)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 50)]));
          conv2d_nchw_local[23] = (conv2d_nchw_local[23] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 85)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 50)]));
          conv2d_nchw_local[31] = (conv2d_nchw_local[31] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 127)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 50)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 2)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 44)]));
          conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 44)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 44)]));
          conv2d_nchw_local[22] = (conv2d_nchw_local[22] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 86)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 44)]));
          conv2d_nchw_local[30] = (conv2d_nchw_local[30] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 128)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 44)]));
          conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 2)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 51)]));
          conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 44)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 51)]));
          conv2d_nchw_local[23] = (conv2d_nchw_local[23] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 86)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 51)]));
          conv2d_nchw_local[31] = (conv2d_nchw_local[31] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 128)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 51)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 3)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 45)]));
          conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 45)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 45)]));
          conv2d_nchw_local[22] = (conv2d_nchw_local[22] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 87)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 45)]));
          conv2d_nchw_local[30] = (conv2d_nchw_local[30] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 129)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 45)]));
          conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 3)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 52)]));
          conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 45)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 52)]));
          conv2d_nchw_local[23] = (conv2d_nchw_local[23] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 87)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 52)]));
          conv2d_nchw_local[31] = (conv2d_nchw_local[31] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 129)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 52)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 4)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 46)]));
          conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 46)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 46)]));
          conv2d_nchw_local[22] = (conv2d_nchw_local[22] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 88)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 46)]));
          conv2d_nchw_local[30] = (conv2d_nchw_local[30] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 130)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 46)]));
          conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 4)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 53)]));
          conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 46)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 53)]));
          conv2d_nchw_local[23] = (conv2d_nchw_local[23] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 88)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 53)]));
          conv2d_nchw_local[31] = (conv2d_nchw_local[31] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 130)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 53)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 5)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 47)]));
          conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 47)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 47)]));
          conv2d_nchw_local[22] = (conv2d_nchw_local[22] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 89)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 47)]));
          conv2d_nchw_local[30] = (conv2d_nchw_local[30] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 131)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 47)]));
          conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 5)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 54)]));
          conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 47)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 54)]));
          conv2d_nchw_local[23] = (conv2d_nchw_local[23] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 89)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 54)]));
          conv2d_nchw_local[31] = (conv2d_nchw_local[31] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 131)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 54)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 6)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 48)]));
          conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 48)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 48)]));
          conv2d_nchw_local[22] = (conv2d_nchw_local[22] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 90)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 48)]));
          conv2d_nchw_local[30] = (conv2d_nchw_local[30] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 132)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 48)]));
          conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 6)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 55)]));
          conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 48)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 55)]));
          conv2d_nchw_local[23] = (conv2d_nchw_local[23] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 90)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 55)]));
          conv2d_nchw_local[31] = (conv2d_nchw_local[31] + (pad_temp_shared[(((((int)item.get_local_id(2)) & 7) * 2) + 132)] * param_0_shared[(((((int)item.get_local_id(2)) >> 3) * 56) + 55)]));
        }
      }
      compute[((((((((int)item.get_group(0)) / 392) * 401408) + ((((int)item.get_local_id(2)) >> 3) * 100352)) + (((((int)item.get_group(0)) % 392) / 14) * 448)) + ((((int)item.get_group(0)) % 14) * 8)) + (((int)item.get_local_id(2)) & 7))] = max((float)(conv2d_nchw_local[0] + param_1[(((((int)item.get_group(0)) / 392) * 32) + ((((int)item.get_local_id(2)) >> 3) * 8))]), (float)0.000000e+00f);
      compute[(((((((((int)item.get_group(0)) / 392) * 401408) + ((((int)item.get_local_id(2)) >> 3) * 100352)) + (((((int)item.get_group(0)) % 392) / 14) * 448)) + ((((int)item.get_group(0)) % 14) * 8)) + (((int)item.get_local_id(2)) & 7)) + 112)] = max((float)(conv2d_nchw_local[8] + param_1[(((((int)item.get_group(0)) / 392) * 32) + ((((int)item.get_local_id(2)) >> 3) * 8))]), (float)0.000000e+00f);
      compute[(((((((((int)item.get_group(0)) / 392) * 401408) + ((((int)item.get_local_id(2)) >> 3) * 100352)) + (((((int)item.get_group(0)) % 392) / 14) * 448)) + ((((int)item.get_group(0)) % 14) * 8)) + (((int)item.get_local_id(2)) & 7)) + 224)] = max((float)(conv2d_nchw_local[16] + param_1[(((((int)item.get_group(0)) / 392) * 32) + ((((int)item.get_local_id(2)) >> 3) * 8))]), (float)0.000000e+00f);
      compute[(((((((((int)item.get_group(0)) / 392) * 401408) + ((((int)item.get_local_id(2)) >> 3) * 100352)) + (((((int)item.get_group(0)) % 392) / 14) * 448)) + ((((int)item.get_group(0)) % 14) * 8)) + (((int)item.get_local_id(2)) & 7)) + 336)] = max((float)(conv2d_nchw_local[24] + param_1[(((((int)item.get_group(0)) / 392) * 32) + ((((int)item.get_local_id(2)) >> 3) * 8))]), (float)0.000000e+00f);
      compute[(((((((((int)item.get_group(0)) / 392) * 401408) + ((((int)item.get_local_id(2)) >> 3) * 100352)) + (((((int)item.get_group(0)) % 392) / 14) * 448)) + ((((int)item.get_group(0)) % 14) * 8)) + (((int)item.get_local_id(2)) & 7)) + 12544)] = max((float)(conv2d_nchw_local[1] + param_1[((((((int)item.get_group(0)) / 392) * 32) + ((((int)item.get_local_id(2)) >> 3) * 8)) + 1)]), (float)0.000000e+00f);
      compute[(((((((((int)item.get_group(0)) / 392) * 401408) + ((((int)item.get_local_id(2)) >> 3) * 100352)) + (((((int)item.get_group(0)) % 392) / 14) * 448)) + ((((int)item.get_group(0)) % 14) * 8)) + (((int)item.get_local_id(2)) & 7)) + 12656)] = max((float)(conv2d_nchw_local[9] + param_1[((((((int)item.get_group(0)) / 392) * 32) + ((((int)item.get_local_id(2)) >> 3) * 8)) + 1)]), (float)0.000000e+00f);
      compute[(((((((((int)item.get_group(0)) / 392) * 401408) + ((((int)item.get_local_id(2)) >> 3) * 100352)) + (((((int)item.get_group(0)) % 392) / 14) * 448)) + ((((int)item.get_group(0)) % 14) * 8)) + (((int)item.get_local_id(2)) & 7)) + 12768)] = max((float)(conv2d_nchw_local[17] + param_1[((((((int)item.get_group(0)) / 392) * 32) + ((((int)item.get_local_id(2)) >> 3) * 8)) + 1)]), (float)0.000000e+00f);
      compute[(((((((((int)item.get_group(0)) / 392) * 401408) + ((((int)item.get_local_id(2)) >> 3) * 100352)) + (((((int)item.get_group(0)) % 392) / 14) * 448)) + ((((int)item.get_group(0)) % 14) * 8)) + (((int)item.get_local_id(2)) & 7)) + 12880)] = max((float)(conv2d_nchw_local[25] + param_1[((((((int)item.get_group(0)) / 392) * 32) + ((((int)item.get_local_id(2)) >> 3) * 8)) + 1)]), (float)0.000000e+00f);
      compute[(((((((((int)item.get_group(0)) / 392) * 401408) + ((((int)item.get_local_id(2)) >> 3) * 100352)) + (((((int)item.get_group(0)) % 392) / 14) * 448)) + ((((int)item.get_group(0)) % 14) * 8)) + (((int)item.get_local_id(2)) & 7)) + 25088)] = max((float)(conv2d_nchw_local[2] + param_1[((((((int)item.get_group(0)) / 392) * 32) + ((((int)item.get_local_id(2)) >> 3) * 8)) + 2)]), (float)0.000000e+00f);
      compute[(((((((((int)item.get_group(0)) / 392) * 401408) + ((((int)item.get_local_id(2)) >> 3) * 100352)) + (((((int)item.get_group(0)) % 392) / 14) * 448)) + ((((int)item.get_group(0)) % 14) * 8)) + (((int)item.get_local_id(2)) & 7)) + 25200)] = max((float)(conv2d_nchw_local[10] + param_1[((((((int)item.get_group(0)) / 392) * 32) + ((((int)item.get_local_id(2)) >> 3) * 8)) + 2)]), (float)0.000000e+00f);
      compute[(((((((((int)item.get_group(0)) / 392) * 401408) + ((((int)item.get_local_id(2)) >> 3) * 100352)) + (((((int)item.get_group(0)) % 392) / 14) * 448)) + ((((int)item.get_group(0)) % 14) * 8)) + (((int)item.get_local_id(2)) & 7)) + 25312)] = max((float)(conv2d_nchw_local[18] + param_1[((((((int)item.get_group(0)) / 392) * 32) + ((((int)item.get_local_id(2)) >> 3) * 8)) + 2)]), (float)0.000000e+00f);
      compute[(((((((((int)item.get_group(0)) / 392) * 401408) + ((((int)item.get_local_id(2)) >> 3) * 100352)) + (((((int)item.get_group(0)) % 392) / 14) * 448)) + ((((int)item.get_group(0)) % 14) * 8)) + (((int)item.get_local_id(2)) & 7)) + 25424)] = max((float)(conv2d_nchw_local[26] + param_1[((((((int)item.get_group(0)) / 392) * 32) + ((((int)item.get_local_id(2)) >> 3) * 8)) + 2)]), (float)0.000000e+00f);
      compute[(((((((((int)item.get_group(0)) / 392) * 401408) + ((((int)item.get_local_id(2)) >> 3) * 100352)) + (((((int)item.get_group(0)) % 392) / 14) * 448)) + ((((int)item.get_group(0)) % 14) * 8)) + (((int)item.get_local_id(2)) & 7)) + 37632)] = max((float)(conv2d_nchw_local[3] + param_1[((((((int)item.get_group(0)) / 392) * 32) + ((((int)item.get_local_id(2)) >> 3) * 8)) + 3)]), (float)0.000000e+00f);
      compute[(((((((((int)item.get_group(0)) / 392) * 401408) + ((((int)item.get_local_id(2)) >> 3) * 100352)) + (((((int)item.get_group(0)) % 392) / 14) * 448)) + ((((int)item.get_group(0)) % 14) * 8)) + (((int)item.get_local_id(2)) & 7)) + 37744)] = max((float)(conv2d_nchw_local[11] + param_1[((((((int)item.get_group(0)) / 392) * 32) + ((((int)item.get_local_id(2)) >> 3) * 8)) + 3)]), (float)0.000000e+00f);
      compute[(((((((((int)item.get_group(0)) / 392) * 401408) + ((((int)item.get_local_id(2)) >> 3) * 100352)) + (((((int)item.get_group(0)) % 392) / 14) * 448)) + ((((int)item.get_group(0)) % 14) * 8)) + (((int)item.get_local_id(2)) & 7)) + 37856)] = max((float)(conv2d_nchw_local[19] + param_1[((((((int)item.get_group(0)) / 392) * 32) + ((((int)item.get_local_id(2)) >> 3) * 8)) + 3)]), (float)0.000000e+00f);
      compute[(((((((((int)item.get_group(0)) / 392) * 401408) + ((((int)item.get_local_id(2)) >> 3) * 100352)) + (((((int)item.get_group(0)) % 392) / 14) * 448)) + ((((int)item.get_group(0)) % 14) * 8)) + (((int)item.get_local_id(2)) & 7)) + 37968)] = max((float)(conv2d_nchw_local[27] + param_1[((((((int)item.get_group(0)) / 392) * 32) + ((((int)item.get_local_id(2)) >> 3) * 8)) + 3)]), (float)0.000000e+00f);
      compute[(((((((((int)item.get_group(0)) / 392) * 401408) + ((((int)item.get_local_id(2)) >> 3) * 100352)) + (((((int)item.get_group(0)) % 392) / 14) * 448)) + ((((int)item.get_group(0)) % 14) * 8)) + (((int)item.get_local_id(2)) & 7)) + 50176)] = max((float)(conv2d_nchw_local[4] + param_1[((((((int)item.get_group(0)) / 392) * 32) + ((((int)item.get_local_id(2)) >> 3) * 8)) + 4)]), (float)0.000000e+00f);
      compute[(((((((((int)item.get_group(0)) / 392) * 401408) + ((((int)item.get_local_id(2)) >> 3) * 100352)) + (((((int)item.get_group(0)) % 392) / 14) * 448)) + ((((int)item.get_group(0)) % 14) * 8)) + (((int)item.get_local_id(2)) & 7)) + 50288)] = max((float)(conv2d_nchw_local[12] + param_1[((((((int)item.get_group(0)) / 392) * 32) + ((((int)item.get_local_id(2)) >> 3) * 8)) + 4)]), (float)0.000000e+00f);
      compute[(((((((((int)item.get_group(0)) / 392) * 401408) + ((((int)item.get_local_id(2)) >> 3) * 100352)) + (((((int)item.get_group(0)) % 392) / 14) * 448)) + ((((int)item.get_group(0)) % 14) * 8)) + (((int)item.get_local_id(2)) & 7)) + 50400)] = max((float)(conv2d_nchw_local[20] + param_1[((((((int)item.get_group(0)) / 392) * 32) + ((((int)item.get_local_id(2)) >> 3) * 8)) + 4)]), (float)0.000000e+00f);
      compute[(((((((((int)item.get_group(0)) / 392) * 401408) + ((((int)item.get_local_id(2)) >> 3) * 100352)) + (((((int)item.get_group(0)) % 392) / 14) * 448)) + ((((int)item.get_group(0)) % 14) * 8)) + (((int)item.get_local_id(2)) & 7)) + 50512)] = max((float)(conv2d_nchw_local[28] + param_1[((((((int)item.get_group(0)) / 392) * 32) + ((((int)item.get_local_id(2)) >> 3) * 8)) + 4)]), (float)0.000000e+00f);
      compute[(((((((((int)item.get_group(0)) / 392) * 401408) + ((((int)item.get_local_id(2)) >> 3) * 100352)) + (((((int)item.get_group(0)) % 392) / 14) * 448)) + ((((int)item.get_group(0)) % 14) * 8)) + (((int)item.get_local_id(2)) & 7)) + 62720)] = max((float)(conv2d_nchw_local[5] + param_1[((((((int)item.get_group(0)) / 392) * 32) + ((((int)item.get_local_id(2)) >> 3) * 8)) + 5)]), (float)0.000000e+00f);
      compute[(((((((((int)item.get_group(0)) / 392) * 401408) + ((((int)item.get_local_id(2)) >> 3) * 100352)) + (((((int)item.get_group(0)) % 392) / 14) * 448)) + ((((int)item.get_group(0)) % 14) * 8)) + (((int)item.get_local_id(2)) & 7)) + 62832)] = max((float)(conv2d_nchw_local[13] + param_1[((((((int)item.get_group(0)) / 392) * 32) + ((((int)item.get_local_id(2)) >> 3) * 8)) + 5)]), (float)0.000000e+00f);
      compute[(((((((((int)item.get_group(0)) / 392) * 401408) + ((((int)item.get_local_id(2)) >> 3) * 100352)) + (((((int)item.get_group(0)) % 392) / 14) * 448)) + ((((int)item.get_group(0)) % 14) * 8)) + (((int)item.get_local_id(2)) & 7)) + 62944)] = max((float)(conv2d_nchw_local[21] + param_1[((((((int)item.get_group(0)) / 392) * 32) + ((((int)item.get_local_id(2)) >> 3) * 8)) + 5)]), (float)0.000000e+00f);
      compute[(((((((((int)item.get_group(0)) / 392) * 401408) + ((((int)item.get_local_id(2)) >> 3) * 100352)) + (((((int)item.get_group(0)) % 392) / 14) * 448)) + ((((int)item.get_group(0)) % 14) * 8)) + (((int)item.get_local_id(2)) & 7)) + 63056)] = max((float)(conv2d_nchw_local[29] + param_1[((((((int)item.get_group(0)) / 392) * 32) + ((((int)item.get_local_id(2)) >> 3) * 8)) + 5)]), (float)0.000000e+00f);
      compute[(((((((((int)item.get_group(0)) / 392) * 401408) + ((((int)item.get_local_id(2)) >> 3) * 100352)) + (((((int)item.get_group(0)) % 392) / 14) * 448)) + ((((int)item.get_group(0)) % 14) * 8)) + (((int)item.get_local_id(2)) & 7)) + 75264)] = max((float)(conv2d_nchw_local[6] + param_1[((((((int)item.get_group(0)) / 392) * 32) + ((((int)item.get_local_id(2)) >> 3) * 8)) + 6)]), (float)0.000000e+00f);
      compute[(((((((((int)item.get_group(0)) / 392) * 401408) + ((((int)item.get_local_id(2)) >> 3) * 100352)) + (((((int)item.get_group(0)) % 392) / 14) * 448)) + ((((int)item.get_group(0)) % 14) * 8)) + (((int)item.get_local_id(2)) & 7)) + 75376)] = max((float)(conv2d_nchw_local[14] + param_1[((((((int)item.get_group(0)) / 392) * 32) + ((((int)item.get_local_id(2)) >> 3) * 8)) + 6)]), (float)0.000000e+00f);
      compute[(((((((((int)item.get_group(0)) / 392) * 401408) + ((((int)item.get_local_id(2)) >> 3) * 100352)) + (((((int)item.get_group(0)) % 392) / 14) * 448)) + ((((int)item.get_group(0)) % 14) * 8)) + (((int)item.get_local_id(2)) & 7)) + 75488)] = max((float)(conv2d_nchw_local[22] + param_1[((((((int)item.get_group(0)) / 392) * 32) + ((((int)item.get_local_id(2)) >> 3) * 8)) + 6)]), (float)0.000000e+00f);
      compute[(((((((((int)item.get_group(0)) / 392) * 401408) + ((((int)item.get_local_id(2)) >> 3) * 100352)) + (((((int)item.get_group(0)) % 392) / 14) * 448)) + ((((int)item.get_group(0)) % 14) * 8)) + (((int)item.get_local_id(2)) & 7)) + 75600)] = max((float)(conv2d_nchw_local[30] + param_1[((((((int)item.get_group(0)) / 392) * 32) + ((((int)item.get_local_id(2)) >> 3) * 8)) + 6)]), (float)0.000000e+00f);
      compute[(((((((((int)item.get_group(0)) / 392) * 401408) + ((((int)item.get_local_id(2)) >> 3) * 100352)) + (((((int)item.get_group(0)) % 392) / 14) * 448)) + ((((int)item.get_group(0)) % 14) * 8)) + (((int)item.get_local_id(2)) & 7)) + 87808)] = max((float)(conv2d_nchw_local[7] + param_1[((((((int)item.get_group(0)) / 392) * 32) + ((((int)item.get_local_id(2)) >> 3) * 8)) + 7)]), (float)0.000000e+00f);
      compute[(((((((((int)item.get_group(0)) / 392) * 401408) + ((((int)item.get_local_id(2)) >> 3) * 100352)) + (((((int)item.get_group(0)) % 392) / 14) * 448)) + ((((int)item.get_group(0)) % 14) * 8)) + (((int)item.get_local_id(2)) & 7)) + 87920)] = max((float)(conv2d_nchw_local[15] + param_1[((((((int)item.get_group(0)) / 392) * 32) + ((((int)item.get_local_id(2)) >> 3) * 8)) + 7)]), (float)0.000000e+00f);
      compute[(((((((((int)item.get_group(0)) / 392) * 401408) + ((((int)item.get_local_id(2)) >> 3) * 100352)) + (((((int)item.get_group(0)) % 392) / 14) * 448)) + ((((int)item.get_group(0)) % 14) * 8)) + (((int)item.get_local_id(2)) & 7)) + 88032)] = max((float)(conv2d_nchw_local[23] + param_1[((((((int)item.get_group(0)) / 392) * 32) + ((((int)item.get_local_id(2)) >> 3) * 8)) + 7)]), (float)0.000000e+00f);
      compute[(((((((((int)item.get_group(0)) / 392) * 401408) + ((((int)item.get_local_id(2)) >> 3) * 100352)) + (((((int)item.get_group(0)) % 392) / 14) * 448)) + ((((int)item.get_group(0)) % 14) * 8)) + (((int)item.get_local_id(2)) & 7)) + 88144)] = max((float)(conv2d_nchw_local[31] + param_1[((((((int)item.get_group(0)) / 392) * 32) + ((((int)item.get_local_id(2)) >> 3) * 8)) + 7)]), (float)0.000000e+00f);
    });
  });
  }

