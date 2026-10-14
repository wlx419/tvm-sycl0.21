// Host cost of a kernel launch on an in-order SYCL queue, as used by the TVM SYCL runtime.
// Submits N empty nd_range kernels (64 work-groups of 64 work-items) and reports
//   submit_us : host time per submission (loop of submits, no wait)
//   total_us  : time per kernel including the final wait (device-bound when kernels queue up)
// for a plain in-order queue and for one with discard_events (no event per submission).
#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

class Empty;

template <typename Fn>
static void Measure(const char* name, sycl::queue& q, float* p, int n, Fn submit) {
  for (int i = 0; i < 2000; ++i) submit(q, p);  // warm-up (kernel bundle, driver JIT)
  q.wait();
  std::vector<double> sub, tot;
  for (int rep = 0; rep < 7; ++rep) {
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; ++i) submit(q, p);
    auto t1 = std::chrono::steady_clock::now();
    q.wait();
    auto t2 = std::chrono::steady_clock::now();
    sub.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count() / n);
    tot.push_back(std::chrono::duration<double, std::micro>(t2 - t0).count() / n);
  }
  std::sort(sub.begin(), sub.end());
  std::sort(tot.begin(), tot.end());
  std::printf("%-28s submit_us=%.3f total_us=%.3f\n", name, sub[3], tot[3]);
}

int main(int argc, char** argv) {
  const int n = argc > 1 ? std::atoi(argv[1]) : 20000;
  auto submit = [](sycl::queue& q, float* p) {
    q.submit([&](sycl::handler& h) {
      h.parallel_for<Empty>(sycl::nd_range<1>(64 * 64, 64), [=](sycl::nd_item<1> it) {
        if (it.get_global_linear_id() == 1u << 30) p[0] = 1.0f;
      });
    });
  };
  #if __LIBSYCL_MAJOR_VERSION >= 6
  sycl::device dev{sycl::gpu_selector_v};
#else
  sycl::device dev{sycl::gpu_selector{}};
#endif
  std::printf("device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
  {
    sycl::queue q(dev, sycl::property::queue::in_order{});
    float* p = sycl::malloc_device<float>(1, q);
    Measure("in_order", q, p, n, submit);
    sycl::free(p, q);
  }
  {
    sycl::queue q(dev, {sycl::property::queue::in_order{},
                        sycl::ext::oneapi::property::queue::discard_events{}});
    float* p = sycl::malloc_device<float>(1, q);
    Measure("in_order+discard_events", q, p, n, submit);
    sycl::free(p, q);
  }
  return 0;
}
