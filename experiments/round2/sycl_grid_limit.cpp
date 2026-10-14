// E8: does the inherited TVM-SYCL launch mapping limit blockIdx.x?
//
// The runtime builds grid(g0,g1,g2) unreversed and block(b2,b1,b0) reversed, launches
// nd_range<3>(grid*block, block), and the kernel reads blockIdx.d as get_group(d) and
// threadIdx.d as get_local_id(2-d). DPC++ maps SYCL dimension 0 to the backend's
// slowest grid axis (CUDA/HIP grid z), so blockIdx.x is bounded by that axis' limit.
// This probe launches a 1-D elementwise kernel with G blocks of 64 threads in both the
// inherited mapping (blockIdx.x on SYCL dim 0) and the reversed mapping (dim 2), and
// checks that every element is written exactly once.
#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

template <int MAPPING>
class GridK;

template <int MAPPING>
static std::string run(sycl::queue& q, size_t G, size_t B) {
  const size_t n = G * B;
  uint32_t* out = sycl::malloc_device<uint32_t>(n, q);
  q.memset(out, 0xff, n * sizeof(uint32_t)).wait();
  sycl::range<3> grid = MAPPING == 0 ? sycl::range<3>(G, 1, 1) : sycl::range<3>(1, 1, G);
  sycl::range<3> block(1, 1, B);
  std::string status = "pass";
  try {
    q.submit([&](sycl::handler& h) {
       h.parallel_for<GridK<MAPPING>>(sycl::nd_range<3>(grid * block, block), [=](sycl::nd_item<3> it) {
         size_t bx = MAPPING == 0 ? it.get_group(0) : it.get_group(2);
         size_t tx = it.get_local_id(2);
         size_t i = bx * B + tx;
         out[i] = (uint32_t)i;
       });
     }).wait_and_throw();
    std::vector<uint32_t> h(n);
    q.memcpy(h.data(), out, n * sizeof(uint32_t)).wait();
    for (size_t i = 0; i < n; ++i) {
      if (h[i] != (uint32_t)i) {
        status = "wrong_result_at_" + std::to_string(i);
        break;
      }
    }
  } catch (const sycl::exception& e) {
    status = std::string("exception: ") + e.what();
  }
  sycl::free(out, q);
  return status;
}

int main() {
  sycl::queue q{sycl::gpu_selector{}, sycl::property::queue::in_order{}};
  auto dev = q.get_device();
  std::printf("{\"device\": \"%s\", \"platform\": \"%s\", \n \"results\": [\n",
              dev.get_info<sycl::info::device::name>().c_str(),
              dev.get_platform().get_info<sycl::info::platform::name>().c_str());
  const size_t B = 64;
  const size_t Gs[] = {1024, 65535, 65536, 131072, 1048576};
  bool first = true;
  for (size_t G : Gs) {
    for (int m = 0; m < 2; ++m) {
      std::string s = m == 0 ? run<0>(q, G, B) : run<1>(q, G, B);
      for (auto& c : s) if (c == '"' || c == '\n') c = ' ';
      std::printf("%s  {\"blocks\": %zu, \"threads\": %zu, \"mapping\": \"%s\", \"status\": \"%s\"}",
                  first ? "" : ",\n", G, B, m == 0 ? "inherited_blockIdx.x_on_dim0" : "reversed_blockIdx.x_on_dim2",
                  s.c_str());
      first = false;
    }
  }
  std::printf("\n]}\n");
  return 0;
}
