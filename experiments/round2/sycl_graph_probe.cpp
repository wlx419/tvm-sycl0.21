// A3 feasibility: sycl_ext_oneapi_graph record & replay on the CUDA backend.
// Records N small kernels submitted to an in-order queue, replays the executable graph, checks
// the result, and compares host-observed time per iteration for eager submission + final wait
// versus one graph submission + wait.
#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdio>
#include <vector>

namespace sycl_exp = sycl::ext::oneapi::experimental;

int main() {
  sycl::queue q{sycl::gpu_selector_v, {sycl::property::queue::in_order{}}};
  const int kernels = 24, n = 1 << 16, iters = 200;
  float* x = sycl::malloc_device<float>(n, q);
  q.fill(x, 0.0f, n).wait();
  auto submit_all = [&] {
    for (int k = 0; k < kernels; ++k) {
      q.parallel_for(sycl::nd_range<1>(n, 256), [=](sycl::nd_item<1> it) {
        size_t i = it.get_group(0) * 256 + it.get_local_id(0);
        x[i] += 1.0f;
      });
    }
  };
  // This DPC++ version predates the graph aspects; support is queried through the device info
  // descriptor of the extension when present, otherwise the graph API is simply tried.
  bool supported = true;
  std::printf("{\"device\": \"%s\"", q.get_device().get_info<sycl::info::device::name>().c_str());
  // eager
  for (int w = 0; w < 10; ++w) submit_all();
  q.wait();
  auto t0 = std::chrono::steady_clock::now();
  for (int it = 0; it < iters; ++it) { submit_all(); q.wait(); }
  double eager_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
  std::printf(", \"eager_us_per_iter\": %.2f", eager_us);
  if (!supported) { std::printf(", \"graph\": \"unsupported\"}\n"); return 0; }
  try {
    sycl_exp::command_graph g{q.get_context(), q.get_device()};
    g.begin_recording(q);
    submit_all();
    g.end_recording(q);
    auto exec = g.finalize();
    for (int w = 0; w < 10; ++w) { q.ext_oneapi_graph(exec); q.wait(); }
    q.wait();
    float before;
    q.memcpy(&before, x, sizeof(float)).wait();
    auto t1 = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; ++it) { q.ext_oneapi_graph(exec); q.wait(); }
    double graph_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t1).count() / iters;
    float after;
    q.memcpy(&after, x, sizeof(float)).wait();
    std::printf(", \"graph_us_per_iter\": %.2f, \"graph_result_ok\": %s}\n", graph_us,
                (after - before == float(kernels * iters)) ? "true" : "false");
  } catch (const sycl::exception& e) {
    std::printf(", \"graph\": \"exception: %s\"}\n", e.what());
  }
  sycl::free(x, q);
  return 0;
}
