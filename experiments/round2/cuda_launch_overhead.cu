// CUDA counterpart of sycl_launch_overhead.cpp: N empty <<<64, 64>>> launches on one stream.
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void Empty(float* p) {
  if (blockIdx.x * blockDim.x + threadIdx.x == (1u << 30)) p[0] = 1.0f;
}

int main(int argc, char** argv) {
  const int n = argc > 1 ? std::atoi(argv[1]) : 20000;
  cudaStream_t s;
  cudaStreamCreate(&s);
  float* p;
  cudaMalloc(&p, sizeof(float));
  for (int i = 0; i < 2000; ++i) Empty<<<64, 64, 0, s>>>(p);
  cudaStreamSynchronize(s);
  std::vector<double> sub, tot;
  for (int rep = 0; rep < 7; ++rep) {
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; ++i) Empty<<<64, 64, 0, s>>>(p);
    auto t1 = std::chrono::steady_clock::now();
    cudaStreamSynchronize(s);
    auto t2 = std::chrono::steady_clock::now();
    sub.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count() / n);
    tot.push_back(std::chrono::duration<double, std::micro>(t2 - t0).count() / n);
  }
  std::sort(sub.begin(), sub.end());
  std::sort(tot.begin(), tot.end());
  std::printf("%-28s submit_us=%.3f total_us=%.3f\n", "cuda_stream", sub[3], tot[3]);
  cudaFree(p);
  return 0;
}
