// E9: accuracy of the float32 a/sqrt(b) -> a*sycl::rsqrt(b) rewrite.
//
// Compares three device forms against a correctly rounded float32 reference computed on
// the host in double precision:
//   div_sqrt   : a / sycl::sqrt(b)          (what TIR asks for)
//   mul_rsqrt  : a * sycl::rsqrt(b)         (what the SYCL codegen emits)
//   mul_recip  : a * (1.0f / sycl::sqrt(b)) (what TVM's LLVM/AMDGPU path emits for rsqrt)
// Input sets: (1) random a in [-8,8], b log-uniform in [1e-6,1e6];
//             (2) LayerNorm-like rows: one b per 256-lane row, broadcast across the row.
// Operand routes: "loaded" (b used straight from memory) and "valu_add_eps"
// (b + 1e-5f computed in-register immediately before the root, as in LayerNorm's
// var + eps), which is the route that exposes the gfx906 v_sqrt_f32 hazard.
// Reports max/mean ULP distance, ULP histogram and non-finite counts.
#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static int64_t ordered(float x) {
  int32_t i;
  std::memcpy(&i, &x, 4);
  return i < 0 ? (int64_t)INT32_MIN - (int64_t)i : (int64_t)i;
}
static int64_t ulp_dist(float a, float b) {
  int64_t d = ordered(a) - ordered(b);
  return d < 0 ? -d : d;
}

struct Stats {
  int64_t max_ulp = 0;
  double sum_ulp = 0;
  int64_t n = 0, nonfinite = 0, hist[5] = {0, 0, 0, 0, 0};  // 0,1,2,3,>3
  double max_rel = 0;
  int64_t bad_lane_mod16[16] = {0};  // non-finite or >3 ulp, by lane % 16
};

template <int FORM, bool EPS>
class K;

template <int FORM, bool EPS>
static void run(sycl::queue& q, const std::vector<float>& a, const std::vector<float>& b,
                std::vector<float>& out) {
  const size_t n = a.size();
  float* da = sycl::malloc_device<float>(n, q);
  float* db = sycl::malloc_device<float>(n, q);
  float* dout = sycl::malloc_device<float>(n, q);
  q.memcpy(da, a.data(), n * 4).wait();
  q.memcpy(db, b.data(), n * 4).wait();
  q.parallel_for<K<FORM, EPS>>(sycl::nd_range<1>(n, 256), [=](sycl::nd_item<1> it) {
     // get_global_id needs the implicit global offset, unsupported by the 2022 HIP plugin;
     // TVM-generated kernels likewise use group and local ids only.
     size_t i = it.get_group(0) * it.get_local_range(0) + it.get_local_id(0);
     float x = da[i], y = db[i];
     if constexpr (EPS) y = y + 1e-5f;
     if constexpr (FORM == 0) dout[i] = x / sycl::sqrt(y);
     if constexpr (FORM == 1) dout[i] = x * sycl::rsqrt(y);
     if constexpr (FORM == 2) dout[i] = x * (1.0f / sycl::sqrt(y));
   }).wait_and_throw();
  q.memcpy(out.data(), dout, n * 4).wait();
  sycl::free(da, q);
  sycl::free(db, q);
  sycl::free(dout, q);
}

static Stats compare(const std::vector<float>& a, const std::vector<float>& b,
                     const std::vector<float>& out, bool eps) {
  Stats s;
  for (size_t i = 0; i < a.size(); ++i) {
    float bb = eps ? (float)(b[i] + 1e-5f) : b[i];
    float ref = (float)((double)a[i] / std::sqrt((double)bb));
    float v = out[i];
    s.n++;
    if (!std::isfinite(v)) {
      s.nonfinite++;
      s.bad_lane_mod16[i % 16]++;
      continue;
    }
    int64_t u = ulp_dist(v, ref);
    if (u > 3) s.bad_lane_mod16[i % 16]++;
    s.max_ulp = std::max(s.max_ulp, u);
    s.sum_ulp += (double)u;
    s.hist[u > 3 ? 4 : u]++;
    if (ref != 0.0f) s.max_rel = std::max(s.max_rel, std::fabs(((double)v - ref) / ref));
  }
  return s;
}

int main(int argc, char** argv) {
  const size_t n = (argc > 1) ? std::stoull(argv[1]) : (size_t(1) << 24);
  sycl::queue q{sycl::gpu_selector{}, sycl::property::queue::in_order{}};
  auto dev = q.get_device();
  std::printf("{\"device\": \"%s\", \"platform\": \"%s\", \"n\": %zu, \"results\": [\n",
              dev.get_info<sycl::info::device::name>().c_str(),
              dev.get_platform().get_info<sycl::info::platform::name>().c_str(), n);

  std::mt19937 rng(20261012);
  std::uniform_real_distribution<float> ua(-8.0f, 8.0f), ue(-6.0f, 6.0f), uv(-5.0f, 1.0f);
  std::vector<float> a(n), b(n), out(n);
  const char* forms[3] = {"div_sqrt", "mul_rsqrt", "mul_recip"};
  bool first = true;
  for (int set = 0; set < 2; ++set) {
    for (size_t i = 0; i < n; ++i) {
      a[i] = ua(rng);
      if (set == 0) {
        b[i] = std::pow(10.0f, ue(rng));
      } else if (i % 256 == 0) {
        b[i] = std::pow(10.0f, uv(rng));
      } else {
        b[i] = b[i - i % 256];
      }
    }
    const char* setname = set == 0 ? "random_logb" : "broadcast_rows256";
    for (int epsi = 0; epsi < 2; ++epsi) {
      const bool eps = epsi == 1;
      for (int form = 0; form < 3; ++form) {
        if (form == 0) { if (eps) run<0, true>(q, a, b, out); else run<0, false>(q, a, b, out); }
        if (form == 1) { if (eps) run<1, true>(q, a, b, out); else run<1, false>(q, a, b, out); }
        if (form == 2) { if (eps) run<2, true>(q, a, b, out); else run<2, false>(q, a, b, out); }
        Stats s = compare(a, b, out, eps);
        std::string lanes;
        for (int l = 0; l < 16; ++l) {
          if (s.bad_lane_mod16[l]) lanes += (lanes.empty() ? "" : ",") + std::to_string(l);
        }
        std::printf("%s  {\"set\": \"%s\", \"operand\": \"%s\", \"form\": \"%s\", \"max_ulp\": %lld, "
                    "\"mean_ulp\": %.4f, \"hist_0_1_2_3_gt3\": [%lld, %lld, %lld, %lld, %lld], "
                    "\"nonfinite\": %lld, \"max_rel\": %.3e, \"bad_lanes_mod16\": [%s]}",
                    first ? "" : ",\n", setname, eps ? "valu_add_eps" : "loaded", forms[form],
                    (long long)s.max_ulp, s.sum_ulp / std::max<int64_t>(1, s.n - s.nonfinite),
                    (long long)s.hist[0], (long long)s.hist[1], (long long)s.hist[2],
                    (long long)s.hist[3], (long long)s.hist[4], (long long)s.nonfinite, s.max_rel,
                    lanes.c_str());
        first = false;
      }
    }
  }
  std::printf("\n]}\n");
  return 0;
}
