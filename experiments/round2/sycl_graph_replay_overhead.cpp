// Host cost of replaying a recorded sycl_ext_oneapi_graph on an in-order queue, as the TVM SYCL
// runtime replays a captured Relax region. The graph holds K small kernels (64 work-groups of 64).
// Each iteration replays the graph and waits for the queue (one inference).
//
// DPC++ 646ab08 keeps the event of every replay of an executable graph and, before each replay,
// checks that all of them have completed (exec_graph_impl::previousSubmissionCompleted), so the
// host cost of a replay grows with the number of earlier replays. Modes:
//   growth <K> : replay cost in windows of replays 1-100, 101-1000, 1001-3000 of one executable graph
//   refin <K> <N> : re-finalize the executable graph every N replays (cost of finalize included)
// Output: mean host time per replay (submit_us) and per iteration including the wait (iter_us).
#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>

namespace sycl_exp = sycl::ext::oneapi::experimental;
using Clock = std::chrono::steady_clock;
class Small;

static void SubmitSmall(sycl::queue& q, float* p, int k) {
  q.submit([&](sycl::handler& h) {
    h.parallel_for<Small>(sycl::nd_range<1>(64 * 64, 64), [=](sycl::nd_item<1> it) {
      if (it.get_global_linear_id() == 1u << 30) p[k] = 1.0f;
    });
  });
}

static double Us(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::micro>(b - a).count();
}

int main(int argc, char** argv) {
  const char* mode = argc > 1 ? argv[1] : "growth";
  const int kernels = argc > 2 ? std::atoi(argv[2]) : 10;
  const int every = argc > 3 ? std::atoi(argv[3]) : 0;
  sycl::device dev{sycl::gpu_selector_v};
  sycl::queue q(dev, sycl::property::queue::in_order{});
  float* p = sycl::malloc_device<float>(kernels, q);
  for (int i = 0; i < 100; ++i) SubmitSmall(q, p, 0);
  q.wait();
  sycl_exp::command_graph<sycl_exp::graph_state::modifiable> g{q.get_context(), q.get_device()};
  g.begin_recording(q);
  for (int k = 0; k < kernels; ++k) SubmitSmall(q, p, k);
  g.end_recording(q);

  auto t0 = Clock::now();
  std::optional<sycl_exp::command_graph<sycl_exp::graph_state::executable>> exec(g.finalize());
  auto t1 = Clock::now();
  std::printf("finalize_us=%.1f (first, K=%d)\n", Us(t0, t1), kernels);

  if (std::strcmp(mode, "growth") == 0) {
    const int bounds[] = {0, 100, 1000, 3000};
    int done = 0;
    for (int w = 1; w < 4; ++w) {
      double sub = 0, it = 0;
      for (; done < bounds[w]; ++done) {
        auto a = Clock::now();
        q.ext_oneapi_graph(*exec);
        auto b = Clock::now();
        q.wait();
        auto c = Clock::now();
        sub += Us(a, b);
        it += Us(a, c);
      }
      int n = bounds[w] - bounds[w - 1];
      std::printf("growth K=%d replays %d-%d submit_us=%.2f iter_us=%.2f\n", kernels, bounds[w - 1] + 1,
                  bounds[w], sub / n, it / n);
    }
  } else {
    const int total = 3000;
    double sub = 0, it = 0, fin = 0;
    int since = 0, nfin = 0;
    for (int i = 0; i < total; ++i) {
      auto a = Clock::now();
      if (since == every) {
        exec.reset();  // waits for (completed) earlier replays, releases the command buffer
        exec.emplace(g.finalize());
        since = 0;
        ++nfin;
      }
      auto f = Clock::now();
      q.ext_oneapi_graph(*exec);
      auto b = Clock::now();
      ++since;
      q.wait();
      auto c = Clock::now();
      fin += Us(a, f);
      sub += Us(a, b);
      it += Us(a, c);
    }
    std::printf("refin K=%d every=%d submit_us=%.2f (finalize share %.2f, %d finalizes) iter_us=%.2f\n",
                kernels, every, sub / total, fin / total, nfin, it / total);
  }
  exec.reset();
  sycl::free(p, q);
  return 0;
}
