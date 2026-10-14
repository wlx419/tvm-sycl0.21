// E11: compute-bound shapes for the oneDNN comparison.
// oneDNN with the SYCL runtime and the NVIDIA vendor backend (cuBLAS/cuDNN), USM memory, the same
// timing as onednn_sycl_baseline.cpp (host wall time of execute + stream wait, 10 warm-ups, 30
// measurements). Correctness against a float64 reference on 4096 sampled output elements.
//   matmul 4096x4096x4096 (row-major A[M,K] B[K,N])
//   conv2d NCHW 16x64x56x56, 64 filters 3x3, stride 1, padding 1 (ResNet stage-1 convolution)
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

namespace {
constexpr int kWarmup = 10;
constexpr int kMeasure = 30;
constexpr int kSamples = 4096;

std::vector<float> Filled(size_t n, float scale) {
  std::vector<float> x(n);
  for (size_t i = 0; i < n; ++i) {
    const int v = static_cast<int>((i * 1315423911u + 2654435761u) % 257u) - 128;
    x[i] = scale * static_cast<float>(v) / 257.0f;
  }
  return x;
}

template <typename Fn>
std::vector<double> Measure(Fn&& fn) {
  for (int i = 0; i < kWarmup; ++i) fn();
  std::vector<double> t;
  for (int i = 0; i < kMeasure; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    fn();
    auto t1 = std::chrono::steady_clock::now();
    t.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  std::sort(t.begin(), t.end());
  return t;
}

float* DeviceCopy(sycl::queue& q, const std::vector<float>& h) {
  float* d = sycl::malloc_device<float>(h.size(), q);
  q.memcpy(d, h.data(), h.size() * sizeof(float)).wait();
  return d;
}

std::vector<float> HostCopy(sycl::queue& q, const float* d, size_t n) {
  std::vector<float> h(n);
  q.memcpy(h.data(), d, n * sizeof(float)).wait();
  return h;
}

struct Row {
  std::string op, shape;
  double max_abs, max_rel_masked, p50, p90;
};

void Report(std::ofstream& out, const Row& r) {
  std::printf("%s %s max_abs=%.3g max_rel_masked=%.3g p50_ms=%.4f p90_ms=%.4f\n", r.op.c_str(),
              r.shape.c_str(), r.max_abs, r.max_rel_masked, r.p50, r.p90);
  out << r.op << ',' << r.shape << ',' << r.max_abs << ',' << r.max_rel_masked << ',' << r.p50
      << ',' << r.p90 << '\n';
}

Row Matmul(dnnl::engine& eng, dnnl::stream& strm, sycl::queue& q) {
  const int M = 4096, K = 4096, N = 4096;
  auto ha = Filled(size_t(M) * K, 1.0f), hb = Filled(size_t(K) * N, 0.5f);
  float* a = DeviceCopy(q, ha);
  float* b = DeviceCopy(q, hb);
  float* c = sycl::malloc_device<float>(size_t(M) * N, q);
  using tag = dnnl::memory::format_tag;
  using dt = dnnl::memory::data_type;
  dnnl::memory::desc amd({M, K}, dt::f32, tag::ab), bmd({K, N}, dt::f32, tag::ab),
      cmd({M, N}, dt::f32, tag::ab);
  auto am = dnnl::sycl_interop::make_memory(amd, eng, dnnl::sycl_interop::memory_kind::usm, a);
  auto bm = dnnl::sycl_interop::make_memory(bmd, eng, dnnl::sycl_interop::memory_kind::usm, b);
  auto cm = dnnl::sycl_interop::make_memory(cmd, eng, dnnl::sycl_interop::memory_kind::usm, c);
  auto prim = dnnl::matmul(dnnl::matmul::primitive_desc(eng, amd, bmd, cmd));
  auto t = Measure([&] {
    prim.execute(strm, {{DNNL_ARG_SRC, am}, {DNNL_ARG_WEIGHTS, bm}, {DNNL_ARG_DST, cm}});
    strm.wait();
  });
  auto hc = HostCopy(q, c, size_t(M) * N);
  double max_abs = 0, max_rel = 0, absmax = 0;
  std::vector<std::pair<double, double>> s;
  for (int i = 0; i < kSamples; ++i) {
    const size_t idx = (size_t(i) * 2654435761u) % (size_t(M) * N);
    const int m = int(idx / N), n = int(idx % N);
    double ref = 0;
    for (int k = 0; k < K; ++k) ref += double(ha[size_t(m) * K + k]) * double(hb[size_t(k) * N + n]);
    s.push_back({hc[idx], ref});
    absmax = std::max(absmax, std::abs(ref));
  }
  for (auto [got, ref] : s) {
    const double d = std::abs(got - ref);
    max_abs = std::max(max_abs, d);
    if (std::abs(ref) >= 1e-3 * absmax) max_rel = std::max(max_rel, d / std::abs(ref));
  }
  sycl::free(a, q); sycl::free(b, q); sycl::free(c, q);
  return {"matmul", "4096x4096x4096", max_abs, max_rel, t[t.size() / 2], t[size_t(0.9 * (t.size() - 1))]};
}

Row Conv(dnnl::engine& eng, dnnl::stream& strm, sycl::queue& q) {
  const int B = 16, C = 64, H = 56, W = 56, F = 64, R = 3, S = 3;
  auto hx = Filled(size_t(B) * C * H * W, 1.0f), hw = Filled(size_t(F) * C * R * S, 0.25f);
  float* x = DeviceCopy(q, hx);
  float* w = DeviceCopy(q, hw);
  float* y = sycl::malloc_device<float>(size_t(B) * F * H * W, q);
  using tag = dnnl::memory::format_tag;
  using dt = dnnl::memory::data_type;
  dnnl::memory::desc xmd({B, C, H, W}, dt::f32, tag::nchw), wmd({F, C, R, S}, dt::f32, tag::oihw),
      ymd({B, F, H, W}, dt::f32, tag::nchw);
  auto xm = dnnl::sycl_interop::make_memory(xmd, eng, dnnl::sycl_interop::memory_kind::usm, x);
  auto wm = dnnl::sycl_interop::make_memory(wmd, eng, dnnl::sycl_interop::memory_kind::usm, w);
  auto ym = dnnl::sycl_interop::make_memory(ymd, eng, dnnl::sycl_interop::memory_kind::usm, y);
  auto pd = dnnl::convolution_forward::primitive_desc(
      eng, dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct, xmd, wmd, ymd,
      {1, 1}, {1, 1}, {1, 1});
  auto prim = dnnl::convolution_forward(pd);
  auto t = Measure([&] {
    prim.execute(strm, {{DNNL_ARG_SRC, xm}, {DNNL_ARG_WEIGHTS, wm}, {DNNL_ARG_DST, ym}});
    strm.wait();
  });
  auto hy = HostCopy(q, y, size_t(B) * F * H * W);
  double max_abs = 0, max_rel = 0, absmax = 0;
  std::vector<std::pair<double, double>> s;
  for (int i = 0; i < kSamples; ++i) {
    const size_t idx = (size_t(i) * 2654435761u) % (size_t(B) * F * H * W);
    const int b = int(idx / (size_t(F) * H * W)), f = int(idx / (H * W) % F);
    const int oh = int(idx / W % H), ow = int(idx % W);
    double ref = 0;
    for (int c = 0; c < C; ++c)
      for (int r = 0; r < R; ++r)
        for (int sx = 0; sx < S; ++sx) {
          const int ih = oh + r - 1, iw = ow + sx - 1;
          if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
          ref += double(hx[((size_t(b) * C + c) * H + ih) * W + iw]) *
                 double(hw[((size_t(f) * C + c) * R + r) * S + sx]);
        }
    s.push_back({hy[idx], ref});
    absmax = std::max(absmax, std::abs(ref));
  }
  for (auto [got, ref] : s) {
    const double d = std::abs(got - ref);
    max_abs = std::max(max_abs, d);
    if (std::abs(ref) >= 1e-3 * absmax) max_rel = std::max(max_rel, d / std::abs(ref));
  }
  sycl::free(x, q); sycl::free(w, q); sycl::free(y, q);
  return {"conv2d_nchw", "16x64x56x56_f64_k3_p1", max_abs, max_rel, t[t.size() / 2], t[size_t(0.9 * (t.size() - 1))]};
}
}  // namespace

int main(int argc, char** argv) {
  std::ofstream out(argc > 1 ? argv[1] : "onednn_large_shapes.csv");
  out << "op,shape,max_abs,max_rel_masked,p50_ms,p90_ms\n";
  dnnl::engine eng(dnnl::engine::kind::gpu, 0);
  dnnl::stream strm(eng);
  sycl::queue q = dnnl::sycl_interop::get_queue(strm);
  Report(out, Matmul(eng, strm, q));
  Report(out, Conv(eng, strm, q));
  return 0;
}
