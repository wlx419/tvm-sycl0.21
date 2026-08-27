#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kWarmup = 10;
constexpr int kMeasure = 30;

struct Timing {
  double p50_ms;
  double mean_ms;
  double p90_ms;
};

struct Result {
  std::string op;
  std::string shape;
  std::string status;
  double max_abs;
  double max_rel;
  double p50_ms;
  double mean_ms;
  double p90_ms;
  std::string notes;
};

class UsmFloat {
 public:
  UsmFloat(size_t n, const sycl::device& dev, const sycl::context& ctx)
      : ctx_(ctx), n_(n) {
    ptr_ = sycl::malloc_device<float>(n_, dev, ctx_);
    if (ptr_ == nullptr) {
      throw std::runtime_error("sycl::malloc_shared failed");
    }
  }

  UsmFloat(const UsmFloat&) = delete;
  UsmFloat& operator=(const UsmFloat&) = delete;

  ~UsmFloat() {
    if (ptr_ != nullptr) {
      sycl::free(ptr_, ctx_);
    }
  }

  float* get() { return ptr_; }
  const float* get() const { return ptr_; }
  size_t size() const { return n_; }

 private:
  sycl::context ctx_;
  float* ptr_ = nullptr;
  size_t n_ = 0;
};

double Percentile(std::vector<double> xs, double q) {
  std::sort(xs.begin(), xs.end());
  if (xs.empty()) return std::numeric_limits<double>::quiet_NaN();
  const double pos = q * static_cast<double>(xs.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(pos));
  const size_t hi = static_cast<size_t>(std::ceil(pos));
  if (lo == hi) return xs[lo];
  const double t = pos - static_cast<double>(lo);
  return xs[lo] * (1.0 - t) + xs[hi] * t;
}

double Mean(const std::vector<double>& xs) {
  return std::accumulate(xs.begin(), xs.end(), 0.0) /
         static_cast<double>(xs.size());
}

void Fill(float* x, size_t n, float scale = 1.0f) {
  for (size_t i = 0; i < n; ++i) {
    const int v =
        static_cast<int>((i * 1315423911u + 2654435761u) % 257u) - 128;
    x[i] = scale * static_cast<float>(v) / 257.0f;
  }
}

std::vector<float> Filled(size_t n, float scale = 1.0f) {
  std::vector<float> x(n);
  Fill(x.data(), x.size(), scale);
  return x;
}

std::pair<double, double> Error(const std::vector<float>& got,
                               const std::vector<float>& ref) {
  double max_abs = 0.0;
  double max_rel = 0.0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double diff =
        std::abs(static_cast<double>(got[i]) - static_cast<double>(ref[i]));
    max_abs = std::max(max_abs, diff);
    max_rel = std::max(
        max_rel, diff / std::max(std::abs(static_cast<double>(ref[i])), 1e-12));
  }
  return {max_abs, max_rel};
}

template <typename Fn>
Timing Measure(Fn&& fn) {
  std::vector<double> times;
  times.reserve(kMeasure);
  for (int i = 0; i < kWarmup; ++i) fn();
  for (int i = 0; i < kMeasure; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    fn();
    const auto t1 = std::chrono::steady_clock::now();
    times.push_back(
        std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  return {Percentile(times, 0.5), Mean(times), Percentile(times, 0.9)};
}

std::string StatusFor(double max_abs, double max_rel) {
  const bool pass = max_abs <= 1e-3 || (max_abs <= 5e-3 && max_rel <= 1e-2);
  return pass ? "pass" : "fail";
}

std::string CleanNotes(const std::string& s) {
  std::string out = s;
  for (char& c : out) {
    if (c == ',' || c == '\n' || c == '\r') c = ' ';
  }
  return out;
}

dnnl::memory MakeUsmMemory(const dnnl::memory::desc& md,
                           const dnnl::engine& eng, float* ptr) {
  return dnnl::sycl_interop::make_memory(
      md, eng, dnnl::sycl_interop::memory_kind::usm, ptr);
}

void CopyToDevice(sycl::queue& q, float* dst, const std::vector<float>& src) {
  q.memcpy(dst, src.data(), src.size() * sizeof(float)).wait();
}

std::vector<float> CopyFromDevice(sycl::queue& q, const float* src, size_t n) {
  std::vector<float> dst(n);
  q.memcpy(dst.data(), src, n * sizeof(float)).wait();
  return dst;
}

Result RunMatmul(dnnl::engine& eng, dnnl::stream& strm) {
  constexpr int M = 256;
  constexpr int K = 256;
  constexpr int N = 256;
  const size_t a_size = M * K;
  const size_t b_size = K * N;
  const size_t c_size = M * N;

  auto dev = dnnl::sycl_interop::get_device(eng);
  auto ctx = dnnl::sycl_interop::get_context(eng);
  auto q = dnnl::sycl_interop::get_queue(strm);
  UsmFloat a(a_size, dev, ctx);
  UsmFloat b(b_size, dev, ctx);
  UsmFloat c(c_size, dev, ctx);
  auto host_a = Filled(a_size, 1.0f);
  auto host_b = Filled(b_size, 0.5f);
  CopyToDevice(q, a.get(), host_a);
  CopyToDevice(q, b.get(), host_b);

  std::vector<float> ref(c_size, 0.0f);
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < K; ++k) {
        sum += host_a[i * K + k] * host_b[k * N + j];
      }
      ref[i * N + j] = sum;
    }
  }

  const auto src_md = dnnl::memory::desc(
      {M, K}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);
  const auto wei_md = dnnl::memory::desc(
      {K, N}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);
  const auto dst_md = dnnl::memory::desc(
      {M, N}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);
  auto src = MakeUsmMemory(src_md, eng, a.get());
  auto wei = MakeUsmMemory(wei_md, eng, b.get());
  auto dst = MakeUsmMemory(dst_md, eng, c.get());
  auto pd = dnnl::matmul::primitive_desc(eng, src_md, wei_md, dst_md);
  auto prim = dnnl::matmul(pd);

  auto run = [&]() {
    prim.execute(strm, {{DNNL_ARG_SRC, src},
                        {DNNL_ARG_WEIGHTS, wei},
                        {DNNL_ARG_DST, dst}});
    strm.wait();
  };
  const Timing timing = Measure(run);
  std::vector<float> got = CopyFromDevice(q, c.get(), c_size);
  const auto [max_abs, max_rel] = Error(got, ref);
  return {"matmul", "256x256x256", StatusFor(max_abs, max_rel),
          max_abs, max_rel, timing.p50_ms, timing.mean_ms, timing.p90_ms, ""};
}

Result RunRowSum(dnnl::engine& eng, dnnl::stream& strm) {
  constexpr int R = 1024;
  constexpr int C = 1024;
  const size_t in_size = R * C;
  const size_t out_size = R;

  auto dev = dnnl::sycl_interop::get_device(eng);
  auto ctx = dnnl::sycl_interop::get_context(eng);
  auto q = dnnl::sycl_interop::get_queue(strm);
  UsmFloat x(in_size, dev, ctx);
  UsmFloat y(out_size, dev, ctx);
  auto host_x = Filled(in_size, 1.0f);
  CopyToDevice(q, x.get(), host_x);

  std::vector<float> ref(out_size, 0.0f);
  for (int r = 0; r < R; ++r) {
    float sum = 0.0f;
    for (int c = 0; c < C; ++c) {
      sum += host_x[r * C + c];
    }
    ref[r] = sum;
  }

  const auto src_md = dnnl::memory::desc(
      {R, C}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);
  const auto dst_md = dnnl::memory::desc(
      {R, 1}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);
  auto src = MakeUsmMemory(src_md, eng, x.get());
  auto dst = MakeUsmMemory(dst_md, eng, y.get());
  auto pd = dnnl::reduction::primitive_desc(
      eng, dnnl::algorithm::reduction_sum, src_md, dst_md, 0.0f, 0.0f);
  auto prim = dnnl::reduction(pd);

  auto run = [&]() {
    prim.execute(strm, {{DNNL_ARG_SRC, src}, {DNNL_ARG_DST, dst}});
    strm.wait();
  };
  const Timing timing = Measure(run);
  std::vector<float> got = CopyFromDevice(q, y.get(), out_size);
  const auto [max_abs, max_rel] = Error(got, ref);
  return {"row_sum", "1024x1024", StatusFor(max_abs, max_rel),
          max_abs, max_rel, timing.p50_ms, timing.mean_ms, timing.p90_ms, ""};
}

Result RunSoftmax(dnnl::engine& eng, dnnl::stream& strm) {
  constexpr int R = 512;
  constexpr int C = 512;
  const size_t n = R * C;

  auto dev = dnnl::sycl_interop::get_device(eng);
  auto ctx = dnnl::sycl_interop::get_context(eng);
  auto q = dnnl::sycl_interop::get_queue(strm);
  UsmFloat x(n, dev, ctx);
  UsmFloat y(n, dev, ctx);
  auto host_x = Filled(n, 1.0f);
  CopyToDevice(q, x.get(), host_x);

  std::vector<float> ref(n, 0.0f);
  for (int r = 0; r < R; ++r) {
    float row_max = host_x[r * C];
    for (int c = 1; c < C; ++c) {
      row_max = std::max(row_max, host_x[r * C + c]);
    }
    float denom = 0.0f;
    for (int c = 0; c < C; ++c) {
      denom += std::exp(host_x[r * C + c] - row_max);
    }
    for (int c = 0; c < C; ++c) {
      ref[r * C + c] = std::exp(host_x[r * C + c] - row_max) / denom;
    }
  }

  const auto md = dnnl::memory::desc(
      {R, C}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);
  auto src = MakeUsmMemory(md, eng, x.get());
  auto dst = MakeUsmMemory(md, eng, y.get());
  auto pd = dnnl::softmax_forward::primitive_desc(
      eng, dnnl::prop_kind::forward_inference, dnnl::algorithm::softmax_accurate,
      md, md, 1);
  auto prim = dnnl::softmax_forward(pd);

  auto run = [&]() {
    prim.execute(strm, {{DNNL_ARG_SRC, src}, {DNNL_ARG_DST, dst}});
    strm.wait();
  };
  const Timing timing = Measure(run);
  std::vector<float> got = CopyFromDevice(q, y.get(), n);
  const auto [max_abs, max_rel] = Error(got, ref);
  return {"softmax", "512x512", StatusFor(max_abs, max_rel),
          max_abs, max_rel, timing.p50_ms, timing.mean_ms, timing.p90_ms, ""};
}

Result RunConv2dNHWC(dnnl::engine& eng, dnnl::stream& strm) {
  constexpr int B = 8;
  constexpr int H = 32;
  constexpr int W = 32;
  constexpr int C = 16;
  constexpr int F = 32;
  constexpr int KH = 3;
  constexpr int KW = 3;
  constexpr int Pad = 1;
  const size_t x_size = B * H * W * C;
  const size_t w_size = KH * KW * C * F;
  const size_t y_size = B * H * W * F;

  auto dev = dnnl::sycl_interop::get_device(eng);
  auto ctx = dnnl::sycl_interop::get_context(eng);
  auto q = dnnl::sycl_interop::get_queue(strm);
  UsmFloat x(x_size, dev, ctx);
  UsmFloat w(w_size, dev, ctx);
  UsmFloat y(y_size, dev, ctx);
  auto host_x = Filled(x_size, 0.5f);
  auto host_w = Filled(w_size, 0.25f);
  CopyToDevice(q, x.get(), host_x);
  CopyToDevice(q, w.get(), host_w);

  std::vector<float> ref(y_size, 0.0f);
  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < H; ++h) {
      for (int ww = 0; ww < W; ++ww) {
        for (int f = 0; f < F; ++f) {
          float sum = 0.0f;
          for (int kh = 0; kh < KH; ++kh) {
            const int ih = h + kh - Pad;
            if (ih < 0 || ih >= H) continue;
            for (int kw = 0; kw < KW; ++kw) {
              const int iw = ww + kw - Pad;
              if (iw < 0 || iw >= W) continue;
              for (int c = 0; c < C; ++c) {
                const size_t x_idx =
                    ((static_cast<size_t>(b) * H + ih) * W + iw) * C + c;
                const size_t w_idx =
                    ((static_cast<size_t>(kh) * KW + kw) * C + c) * F + f;
                sum += host_x[x_idx] * host_w[w_idx];
              }
            }
          }
          ref[((static_cast<size_t>(b) * H + h) * W + ww) * F + f] = sum;
        }
      }
    }
  }

  const auto src_md = dnnl::memory::desc(
      {B, C, H, W}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nhwc);
  const auto wei_md = dnnl::memory::desc(
      {F, C, KH, KW}, dnnl::memory::data_type::f32,
      dnnl::memory::format_tag::hwio);
  const auto dst_md = dnnl::memory::desc(
      {B, F, H, W}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nhwc);
  auto src = MakeUsmMemory(src_md, eng, x.get());
  auto wei = MakeUsmMemory(wei_md, eng, w.get());
  auto dst = MakeUsmMemory(dst_md, eng, y.get());
  auto pd = dnnl::convolution_forward::primitive_desc(
      eng, dnnl::prop_kind::forward_inference,
      dnnl::algorithm::convolution_direct, src_md, wei_md, dst_md, {1, 1},
      {Pad, Pad}, {Pad, Pad});
  auto prim = dnnl::convolution_forward(pd);

  auto run = [&]() {
    prim.execute(strm, {{DNNL_ARG_SRC, src},
                        {DNNL_ARG_WEIGHTS, wei},
                        {DNNL_ARG_DST, dst}});
    strm.wait();
  };
  const Timing timing = Measure(run);
  std::vector<float> got = CopyFromDevice(q, y.get(), y_size);
  const auto [max_abs, max_rel] = Error(got, ref);
  return {"conv2d_nhwc", "8x32x32x16_to_32_k3", StatusFor(max_abs, max_rel),
          max_abs, max_rel, timing.p50_ms, timing.mean_ms, timing.p90_ms, ""};
}

template <typename Fn>
void RunAndAppend(std::vector<Result>& results, const std::string& op,
                  const std::string& shape, Fn&& fn) {
  try {
    results.push_back(fn());
  } catch (const dnnl::error& e) {
    std::ostringstream os;
    os << "oneDNN " << static_cast<int>(e.status) << " " << e.what();
    results.push_back({op, shape, "error",
                       std::numeric_limits<double>::quiet_NaN(),
                       std::numeric_limits<double>::quiet_NaN(),
                       std::numeric_limits<double>::quiet_NaN(),
                       std::numeric_limits<double>::quiet_NaN(),
                       std::numeric_limits<double>::quiet_NaN(),
                       CleanNotes(os.str())});
  } catch (const std::exception& e) {
    results.push_back({op, shape, "error",
                       std::numeric_limits<double>::quiet_NaN(),
                       std::numeric_limits<double>::quiet_NaN(),
                       std::numeric_limits<double>::quiet_NaN(),
                       std::numeric_limits<double>::quiet_NaN(),
                       std::numeric_limits<double>::quiet_NaN(),
                       CleanNotes(e.what())});
  }
}

void WriteCsv(const std::string& path, const std::vector<Result>& results) {
  std::ofstream out(path);
  out << std::setprecision(10);
  out << "op,shape,status,max_abs,max_rel,p50_ms,mean_ms,p90_ms,notes\n";
  for (const auto& r : results) {
    out << r.op << ',' << r.shape << ',' << r.status << ',' << r.max_abs << ','
        << r.max_rel << ',' << r.p50_ms << ',' << r.mean_ms << ',' << r.p90_ms
        << ',' << CleanNotes(r.notes) << '\n';
  }
}

}  // namespace

int main(int argc, char** argv) {
  const std::string out_path =
      argc > 1 ? argv[1] : "onednn_sycl_baseline_20260707.csv";

  try {
    dnnl::engine eng(dnnl::engine::kind::gpu, 0);
    dnnl::stream strm(eng);
    std::cout << "oneDNN engine kind: gpu\n";

    std::vector<Result> results;
    RunAndAppend(results, "matmul", "256x256x256",
                 [&]() { return RunMatmul(eng, strm); });
    RunAndAppend(results, "row_sum", "1024x1024",
                 [&]() { return RunRowSum(eng, strm); });
    RunAndAppend(results, "softmax", "512x512",
                 [&]() { return RunSoftmax(eng, strm); });
    RunAndAppend(results, "conv2d_nhwc", "8x32x32x16_to_32_k3",
                 [&]() { return RunConv2dNHWC(eng, strm); });

    WriteCsv(out_path, results);
    for (const auto& r : results) {
      std::cout << r.op << ' ' << r.status << " p50_ms=" << r.p50_ms
                << " max_abs=" << r.max_abs << " notes=" << r.notes << '\n';
    }
  } catch (const std::exception& e) {
    std::cerr << "fatal: " << e.what() << '\n';
    return 1;
  }

  return 0;
}
