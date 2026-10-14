// CUDA counterpart of sycl_graph_replay_overhead.cpp: a captured graph of K small kernels
// (64 blocks of 64 threads), replayed with cudaGraphLaunch and a stream synchronize per iteration.
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void Small(float* p, int k) {
  if (blockIdx.x * blockDim.x + threadIdx.x == (1u << 30)) p[k] = 1.0f;
}

int main(int argc, char** argv) {
  const int iters = argc > 1 ? std::atoi(argv[1]) : 5000;
  const int kernels = argc > 2 ? std::atoi(argv[2]) : 10;
  cudaStream_t s;
  cudaStreamCreate(&s);
  float* p;
  cudaMalloc(&p, kernels * sizeof(float));
  for (int i = 0; i < 100; ++i) Small<<<64, 64, 0, s>>>(p, 0);
  cudaStreamSynchronize(s);
  cudaGraph_t g;
  cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
  for (int k = 0; k < kernels; ++k) Small<<<64, 64, 0, s>>>(p, k);
  cudaStreamEndCapture(s, &g);
  cudaGraphExec_t exec;
  cudaGraphInstantiate(&exec, g, 0);
  for (int i = 0; i < 200; ++i) { cudaGraphLaunch(exec, s); cudaStreamSynchronize(s); }
  std::vector<double> sub, it;
  for (int rep = 0; rep < 5; ++rep) {
    double acc = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
      auto a = std::chrono::steady_clock::now();
      cudaGraphLaunch(exec, s);
      auto b = std::chrono::steady_clock::now();
      acc += std::chrono::duration<double, std::micro>(b - a).count();
      cudaStreamSynchronize(s);
    }
    auto t1 = std::chrono::steady_clock::now();
    sub.push_back(acc / iters);
    it.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count() / iters);
  }
  std::sort(sub.begin(), sub.end());
  std::sort(it.begin(), it.end());
  std::printf("cuda graph   submit_us=%.3f iter_us=%.3f (%d kernels)\n", sub[2], it[2], kernels);
  std::vector<double> ev;
  for (int rep = 0; rep < 5; ++rep) {
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
      for (int k = 0; k < kernels; ++k) Small<<<64, 64, 0, s>>>(p, k);
      cudaStreamSynchronize(s);
    }
    auto t1 = std::chrono::steady_clock::now();
    ev.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count() / iters);
  }
  std::sort(ev.begin(), ev.end());
  std::printf("cuda eager   iter_us=%.3f\n", ev[2]);
  return 0;
}
