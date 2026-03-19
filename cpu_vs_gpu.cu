#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <cmath>
#include <fstream>
#include <ctime>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <sstream>
#include <iomanip>

#define BLOCK_SIZE 16

// ================================================================
// Benchmark configuration
// ================================================================
static const int NUM_RUNS     = 6;    // run 0 = cold, 1-5 = warm
static const int CPU_REPEATS  = 50;   // timed iterations for CPU avg
static const int KERNEL_ITERS = 50;   // timed iterations for GPU avg

// ================================================================
// NAIVE GPU KERNELS — global memory only
// ================================================================

__global__ void gaussianBlurNaive(const unsigned char* input, unsigned char* output,
                                   const float* kernel5x5, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 2 || y < 2 || x >= width - 2 || y >= height - 2) return;
    float sum = 0.0f;
    for (int ky = -2; ky <= 2; ky++)
        for (int kx = -2; kx <= 2; kx++)
            sum += input[(y + ky) * width + (x + kx)] * kernel5x5[(ky + 2) * 5 + (kx + 2)];
    output[y * width + x] = static_cast<unsigned char>(sum);
}

__global__ void sobelGradientNaive(const unsigned char* input, float* grad, float* angle,
                                    int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 1 || y < 1 || x >= width - 1 || y >= height - 1) return;
    int Gx = -input[(y-1)*width+(x-1)] - 2*input[y*width+(x-1)] - input[(y+1)*width+(x-1)]
             +input[(y-1)*width+(x+1)] + 2*input[y*width+(x+1)] + input[(y+1)*width+(x+1)];
    int Gy = -input[(y-1)*width+(x-1)] - 2*input[(y-1)*width+x] - input[(y-1)*width+(x+1)]
             +input[(y+1)*width+(x-1)] + 2*input[(y+1)*width+x] + input[(y+1)*width+(x+1)];
    float fx = static_cast<float>(Gx), fy = static_cast<float>(Gy);
    grad[y * width + x]  = sqrtf(fx * fx + fy * fy);
    angle[y * width + x] = atan2f(fy, fx);
}

__global__ void nonMaxSuppressionNaive(const float* grad, const float* angle,
                                        unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 1 || y < 1 || x >= width - 1 || y >= height - 1) return;
    float a = angle[y * width + x], g = grad[y * width + x];
    float q = 0.0f, r = 0.0f;
    if      ((a >= -M_PI/8 && a <= M_PI/8) || (a <= -7*M_PI/8) || (a >= 7*M_PI/8))
        { q = grad[y*width+(x+1)];     r = grad[y*width+(x-1)]; }
    else if ((a >= M_PI/8 && a < 3*M_PI/8) || (a <= -5*M_PI/8 && a > -7*M_PI/8))
        { q = grad[(y+1)*width+(x-1)]; r = grad[(y-1)*width+(x+1)]; }
    else if ((a >= 3*M_PI/8 && a <= 5*M_PI/8) || (a <= -3*M_PI/8 && a >= -5*M_PI/8))
        { q = grad[(y+1)*width+x];     r = grad[(y-1)*width+x]; }
    else
        { q = grad[(y-1)*width+(x-1)]; r = grad[(y+1)*width+(x+1)]; }
    output[y * width + x] = (g >= q && g >= r) ? static_cast<unsigned char>(fminf(g, 255.0f)) : 0;
}

__global__ void hysteresisNaive(unsigned char* edges, int width, int height,
                                 unsigned char low, unsigned char high) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    unsigned char val = edges[y * width + x];
    if      (val >= high) edges[y * width + x] = 255;
    else if (val < low)   edges[y * width + x] = 0;
}

// ================================================================
// OPTIMIZED GPU KERNELS — __constant__ + shared memory
// ================================================================

__constant__ float d_kernel[25];

__global__ void gaussianBlurGPUShared(const unsigned char* input, unsigned char* output,
                                       int width, int height) {
    constexpr int R = 2, TILE_W = BLOCK_SIZE + 2*R, TILE_H = BLOCK_SIZE + 2*R;
    __shared__ unsigned char tile[TILE_H][TILE_W];
    const int blockX = blockIdx.x * BLOCK_SIZE, blockY = blockIdx.y * BLOCK_SIZE;
    const int threadLinear = threadIdx.y * blockDim.x + threadIdx.x;
    const int tileSize = TILE_W * TILE_H;
    for (int i = threadLinear; i < tileSize; i += blockDim.x * blockDim.y) {
        const int localY = i / TILE_W, localX = i - localY * TILE_W;
        const int gx = blockX + localX - R, gy = blockY + localY - R;
        tile[localY][localX] = (gx >= 0 && gy >= 0 && gx < width && gy < height)
                               ? input[gy * width + gx] : 0;
    }
    __syncthreads();
    const int x = blockX + threadIdx.x, y = blockY + threadIdx.y;
    if (x < 2 || y < 2 || x >= width - 2 || y >= height - 2) return;
    float sum = 0.0f;
    #pragma unroll
    for (int ky = -R; ky <= R; ++ky)
        #pragma unroll
        for (int kx = -R; kx <= R; ++kx)
            sum += tile[threadIdx.y+ky+R][threadIdx.x+kx+R] * d_kernel[(ky+R)*5+(kx+R)];
    output[y * width + x] = static_cast<unsigned char>(sum);
}

__global__ void sobelGradientGPU(const unsigned char* input, float* grad, float* angle,
                                  int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 1 || y < 1 || x >= width - 1 || y >= height - 1) return;
    int Gx = -input[(y-1)*width+(x-1)] - 2*input[y*width+(x-1)] - input[(y+1)*width+(x-1)]
             +input[(y-1)*width+(x+1)] + 2*input[y*width+(x+1)] + input[(y+1)*width+(x+1)];
    int Gy = -input[(y-1)*width+(x-1)] - 2*input[(y-1)*width+x] - input[(y-1)*width+(x+1)]
             +input[(y+1)*width+(x-1)] + 2*input[(y+1)*width+x] + input[(y+1)*width+(x+1)];
    float fx = static_cast<float>(Gx), fy = static_cast<float>(Gy);
    grad[y * width + x]  = sqrtf(fx * fx + fy * fy);
    angle[y * width + x] = atan2f(fy, fx);
}

__global__ void nonMaxSuppressionGPU(const float* grad, const float* angle,
                                      unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 1 || y < 1 || x >= width - 1 || y >= height - 1) return;
    float a = angle[y * width + x], g = grad[y * width + x];
    float q = 0.0f, r = 0.0f;
    if      ((a >= -M_PI/8 && a <= M_PI/8) || (a <= -7*M_PI/8) || (a >= 7*M_PI/8))
        { q = grad[y*width+(x+1)];     r = grad[y*width+(x-1)]; }
    else if ((a >= M_PI/8 && a < 3*M_PI/8) || (a <= -5*M_PI/8 && a > -7*M_PI/8))
        { q = grad[(y+1)*width+(x-1)]; r = grad[(y-1)*width+(x+1)]; }
    else if ((a >= 3*M_PI/8 && a <= 5*M_PI/8) || (a <= -3*M_PI/8 && a >= -5*M_PI/8))
        { q = grad[(y+1)*width+x];     r = grad[(y-1)*width+x]; }
    else
        { q = grad[(y-1)*width+(x-1)]; r = grad[(y+1)*width+(x+1)]; }
    output[y * width + x] = (g >= q && g >= r) ? static_cast<unsigned char>(fminf(g, 255.0f)) : 0;
}

__global__ void hysteresisGPU(unsigned char* edges, int width, int height,
                               unsigned char low, unsigned char high) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    unsigned char val = edges[y * width + x];
    if      (val >= high) edges[y * width + x] = 255;
    else if (val < low)   edges[y * width + x] = 0;
}

// ================================================================
// Helper functions
// ================================================================

static cv::Mat makeRoadRoiMask(cv::Size sz) {
    cv::Mat mask = cv::Mat::zeros(sz, CV_8UC1);
    int w = sz.width, h = sz.height;
    std::vector<cv::Point> poly = {
        cv::Point(static_cast<int>(w * 0.05), h),
        cv::Point(static_cast<int>(w * 0.15), static_cast<int>(h * 0.60)),
        cv::Point(static_cast<int>(w * 0.55), static_cast<int>(h * 0.60)),
        cv::Point(static_cast<int>(w * 0.95), h),
    };
    cv::fillConvexPoly(mask, poly, cv::Scalar(255));
    return mask;
}

static cv::Mat whiteYellowMask(const cv::Mat& bgr) {
    cv::Mat hsv;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
    cv::Mat yellowMask;
    cv::inRange(hsv, cv::Scalar(15, 60, 60), cv::Scalar(45, 255, 255), yellowMask);
    std::vector<cv::Mat> ch(3);
    cv::split(bgr, ch);
    cv::Mat whiteMask = (ch[0] > 180) & (ch[1] > 180) & (ch[2] > 180);
    cv::Mat mask;
    cv::bitwise_or(yellowMask, whiteMask, mask);
    cv::Mat k = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, k);
    return mask;
}

static cv::Mat extractLaneLinesFromEdges(const cv::Mat& edges, const cv::Mat& origBgr,
                                          int houghThresh, int minLen, int maxGap) {
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(edges, lines, 1, CV_PI / 180, houghThresh, minLen, maxGap);
    cv::Mat laneImg = cv::Mat::zeros(origBgr.size(), CV_8UC3);
    for (const auto& l : lines)
        cv::line(laneImg, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 0, 255), 3);
    cv::Mat out;
    cv::addWeighted(origBgr, 0.85, laneImg, 1.0, 0.0, out);
    return out;
}

// stddev of a vector of doubles
static double stddev(const std::vector<double>& v) {
    if (v.size() < 2) return 0.0;
    double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    double sq = 0.0;
    for (double x : v) sq += (x - mean) * (x - mean);
    return std::sqrt(sq / (v.size() - 1));
}

static std::string csvEscape(const std::string& s) {
    std::string out;
    out.push_back('"');
    for (char c : s) { if (c == '"') out.push_back('"'); out.push_back(c); }
    out.push_back('"');
    return out;
}

// ================================================================
// appendTimingCsvRow — expanded schema with all new metrics
//
// New columns vs. original:
//   run_index            — 0=cold, 1-5=warm
//   throughput_*_mpps    — megapixels per second per pipeline
//   fps_*                — effective frames per second per pipeline
//   -- per-stage GPU timings (naive and optimized) --
//   naive_blur_ms        — Gaussian blur kernel alone (naive)
//   naive_sobel_ms       — Sobel gradient kernel alone (naive)
//   naive_nms_ms         — NMS kernel alone (naive)
//   naive_hyst_ms        — hysteresis kernel alone (naive)
//   opt_blur_ms          — Gaussian blur kernel alone (optimized)
//   opt_sobel_ms         — Sobel gradient kernel alone (optimized)
//   opt_nms_ms           — NMS kernel alone (optimized)
//   opt_hyst_ms          — hysteresis kernel alone (optimized)
//   -- quality / correctness --
//   cpu_edge_pixels      — non-zero pixels in CPU edge output
//   naive_edge_pixels    — non-zero pixels in GPU naive edge output
//   opt_edge_pixels      — non-zero pixels in GPU opt edge output
//   -- consistency --
//   cpu_stddev_ms        — std dev across CPU_REPEATS iterations
//   naive_stddev_ms      — std dev across KERNEL_ITERS iterations
//   opt_stddev_ms        — std dev across KERNEL_ITERS iterations
//   -- bandwidth (analytical, no profiler needed) --
//   naive_blur_bw_gbs    — effective GB/s for naive Gaussian blur
//   opt_blur_bw_gbs      — effective GB/s for optimized Gaussian blur
// ================================================================
struct RunMetrics {
    // timing
    double cpu_ms, naive_ms, opt_ms;
    // per-stage (GPU only — CPU stages not separable via chrono easily)
    double naive_blur_ms, naive_sobel_ms, naive_nms_ms, naive_hyst_ms;
    double opt_blur_ms,   opt_sobel_ms,   opt_nms_ms,   opt_hyst_ms;
    // quality
    int cpu_edge_px, naive_edge_px, opt_edge_px;
    // consistency
    double cpu_stddev_ms, naive_stddev_ms, opt_stddev_ms;
    // derived (filled after timing)
    double cpu_mpps, naive_mpps, opt_mpps;
    double cpu_fps,  naive_fps,  opt_fps;
    double naive_blur_bw_gbs, opt_blur_bw_gbs;
    // speedups
    double cpu_over_naive, cpu_over_opt, naive_over_opt;
};

static void appendCsvRow(const std::string& csvPath,
                          const std::string& inputPath,
                          int width, int height, int runIndex,
                          const RunMetrics& m) {
    bool writeHeader = true;
    {
        std::ifstream fin(csvPath.c_str());
        if (fin.good()) { fin.seekg(0, std::ios::end); writeHeader = (fin.tellg() == 0); }
    }
    std::ofstream f(csvPath.c_str(), std::ios::out | std::ios::app);
    if (!f.good()) { std::cerr << "Warning: cannot open " << csvPath << "\n"; return; }

    if (writeHeader) {
        f << "epoch_s,run_index,input_path,width,height,"
          // overall timing
          << "cpu_edge_ms,gpu_naive_edge_ms,gpu_opt_edge_ms,"
          // speedups
          << "cpu_over_gpu_naive,cpu_over_gpu_opt,gpu_naive_over_gpu_opt,"
          // throughput
          << "cpu_mpps,naive_mpps,opt_mpps,"
          // fps
          << "cpu_fps,naive_fps,opt_fps,"
          // per-stage naive
          << "naive_blur_ms,naive_sobel_ms,naive_nms_ms,naive_hyst_ms,"
          // per-stage opt
          << "opt_blur_ms,opt_sobel_ms,opt_nms_ms,opt_hyst_ms,"
          // quality / correctness
          << "cpu_edge_pixels,naive_edge_pixels,opt_edge_pixels,"
          // consistency
          << "cpu_stddev_ms,naive_stddev_ms,opt_stddev_ms,"
          // bandwidth
          << "naive_blur_bw_gbs,opt_blur_bw_gbs\n";
    }

    const std::time_t epoch = std::time(nullptr);
    auto p = [](double v) { return std::to_string(v); };

    f << epoch                   << ','
      << runIndex                << ','
      << csvEscape(inputPath)    << ','
      << width << ',' << height  << ','
      // timing
      << m.cpu_ms   << ',' << m.naive_ms << ',' << m.opt_ms << ','
      // speedups
      << m.cpu_over_naive << ',' << m.cpu_over_opt << ',' << m.naive_over_opt << ','
      // throughput
      << m.cpu_mpps   << ',' << m.naive_mpps << ',' << m.opt_mpps << ','
      // fps
      << m.cpu_fps    << ',' << m.naive_fps  << ',' << m.opt_fps  << ','
      // per-stage naive
      << m.naive_blur_ms << ',' << m.naive_sobel_ms << ','
      << m.naive_nms_ms  << ',' << m.naive_hyst_ms  << ','
      // per-stage opt
      << m.opt_blur_ms   << ',' << m.opt_sobel_ms   << ','
      << m.opt_nms_ms    << ',' << m.opt_hyst_ms    << ','
      // quality
      << m.cpu_edge_px << ',' << m.naive_edge_px << ',' << m.opt_edge_px << ','
      // consistency
      << m.cpu_stddev_ms << ',' << m.naive_stddev_ms << ',' << m.opt_stddev_ms << ','
      // bandwidth
      << m.naive_blur_bw_gbs << ',' << m.opt_blur_bw_gbs
      << "\n";
}

// ================================================================
// timeSingleKernel — fires one kernel KERNEL_ITERS times between
// a pair of CUDA events and returns the per-iteration average in ms.
// The lambda captures everything the kernel needs.
// ================================================================
template<typename KernelFn>
static double timeSingleKernel(KernelFn fn, std::vector<double>& perIterMs) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // one warm-up
    fn();
    cudaDeviceSynchronize();

    perIterMs.clear();
    perIterMs.reserve(KERNEL_ITERS);

    // time each iteration individually so we can compute stddev
    for (int i = 0; i < KERNEL_ITERS; ++i) {
        cudaEventRecord(start);
        fn();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        perIterMs.push_back(static_cast<double>(ms));
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    double total = std::accumulate(perIterMs.begin(), perIterMs.end(), 0.0);
    return total / static_cast<double>(KERNEL_ITERS);
}

// ================================================================
// runBenchmark — one complete benchmark pass
// ================================================================
static RunMetrics runBenchmark(const std::string& inputPath,
                                const cv::Mat& imgGray,
                                const cv::Mat& frame,
                                const float* h_kernel,
                                int width, int height,
                                int callRun,
                                bool writeImages) {

    RunMetrics m{};
    const long long totalPixels = static_cast<long long>(width) * height;

    size_t imgSize   = static_cast<size_t>(width) * height * sizeof(uchar);
    size_t floatSize = static_cast<size_t>(width) * height * sizeof(float);

    const int   CPU_CANNY_LOW  = 50,  CPU_CANNY_HIGH = 150;
    const uchar GPU_LOW_T      = 20,  GPU_HIGH_T     = 80;

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width  + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // ============================================================
    // 1) CPU PIPELINE
    //    Warm-up → CPU_REPEATS timed iterations → avg + stddev
    // ============================================================
    cv::Mat cpuEdges;
    {
        cv::Mat blurWarm;
        cv::GaussianBlur(imgGray, blurWarm, cv::Size(5, 5), 1.2);
        cv::Canny(blurWarm, cpuEdges, CPU_CANNY_LOW, CPU_CANNY_HIGH);
    }

    std::vector<double> cpuSamples;
    cpuSamples.reserve(CPU_REPEATS);
    for (int i = 0; i < CPU_REPEATS; ++i) {
        cv::Mat blur;
        auto t1 = std::chrono::high_resolution_clock::now();
        cv::GaussianBlur(imgGray, blur, cv::Size(5, 5), 1.2);
        cv::Canny(blur, cpuEdges, CPU_CANNY_LOW, CPU_CANNY_HIGH);
        auto t2 = std::chrono::high_resolution_clock::now();
        cpuSamples.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
    }
    m.cpu_ms         = std::accumulate(cpuSamples.begin(), cpuSamples.end(), 0.0) / CPU_REPEATS;
    m.cpu_stddev_ms  = stddev(cpuSamples);
    m.cpu_edge_px    = cv::countNonZero(cpuEdges);
    m.cpu_mpps       = (totalPixels / 1.0e6) / (m.cpu_ms / 1000.0);
    m.cpu_fps        = 1000.0 / m.cpu_ms;

    if (writeImages) {
        cv::imwrite("cpu_edges.png", cpuEdges);
        cv::imwrite("cpu_lane_detection.png",
                    extractLaneLinesFromEdges(cpuEdges, frame, 40, 40, 150));
    }

    // ============================================================
    // 2) NAIVE GPU PIPELINE — per-stage timing
    // ============================================================
    uchar *dn_input, *dn_blur, *dn_nms;
    float *dn_grad, *dn_angle, *dn_gkernel;
    cudaMalloc(&dn_input,   imgSize);
    cudaMalloc(&dn_blur,    imgSize);
    cudaMalloc(&dn_nms,     imgSize);
    cudaMalloc(&dn_grad,    floatSize);
    cudaMalloc(&dn_angle,   floatSize);
    cudaMalloc(&dn_gkernel, 25 * sizeof(float));
    cudaMemcpy(dn_input,   imgGray.data, imgSize,           cudaMemcpyHostToDevice);
    cudaMemcpy(dn_gkernel, h_kernel,     25 * sizeof(float), cudaMemcpyHostToDevice);

    // Time each stage individually (per-iteration samples for stddev)
    std::vector<double> nBlurPer, nSobelPer, nNmsPer, nHystPer;

    m.naive_blur_ms  = timeSingleKernel([&]{ gaussianBlurNaive     <<<grid,block>>>(dn_input,dn_blur,dn_gkernel,width,height); }, nBlurPer);
    m.naive_sobel_ms = timeSingleKernel([&]{ sobelGradientNaive    <<<grid,block>>>(dn_blur, dn_grad,dn_angle,  width,height); }, nSobelPer);
    m.naive_nms_ms   = timeSingleKernel([&]{ nonMaxSuppressionNaive<<<grid,block>>>(dn_grad, dn_angle,dn_nms,   width,height); }, nNmsPer);
    m.naive_hyst_ms  = timeSingleKernel([&]{ hysteresisNaive       <<<grid,block>>>(dn_nms,  width,height,GPU_LOW_T,GPU_HIGH_T); }, nHystPer);

    // Combined pipeline stddev — time full pipeline end-to-end
    {
        std::vector<double> pipelineSamples;
        pipelineSamples.reserve(KERNEL_ITERS);
        // warm-up
        gaussianBlurNaive     <<<grid,block>>>(dn_input,dn_blur,dn_gkernel,width,height);
        sobelGradientNaive    <<<grid,block>>>(dn_blur, dn_grad,dn_angle,  width,height);
        nonMaxSuppressionNaive<<<grid,block>>>(dn_grad, dn_angle,dn_nms,   width,height);
        hysteresisNaive       <<<grid,block>>>(dn_nms,  width,height,GPU_LOW_T,GPU_HIGH_T);
        cudaDeviceSynchronize();

        cudaEvent_t pS, pE;
        cudaEventCreate(&pS); cudaEventCreate(&pE);
        for (int i = 0; i < KERNEL_ITERS; ++i) {
            cudaEventRecord(pS);
            gaussianBlurNaive     <<<grid,block>>>(dn_input,dn_blur,dn_gkernel,width,height);
            sobelGradientNaive    <<<grid,block>>>(dn_blur, dn_grad,dn_angle,  width,height);
            nonMaxSuppressionNaive<<<grid,block>>>(dn_grad, dn_angle,dn_nms,   width,height);
            hysteresisNaive       <<<grid,block>>>(dn_nms,  width,height,GPU_LOW_T,GPU_HIGH_T);
            cudaEventRecord(pE);
            cudaEventSynchronize(pE);
            float ms = 0.0f; cudaEventElapsedTime(&ms, pS, pE);
            pipelineSamples.push_back(static_cast<double>(ms));
        }
        cudaEventDestroy(pS); cudaEventDestroy(pE);
        m.naive_ms        = std::accumulate(pipelineSamples.begin(), pipelineSamples.end(), 0.0) / KERNEL_ITERS;
        m.naive_stddev_ms = stddev(pipelineSamples);
    }

    // Edge pixel count
    {
        cv::Mat naiveEdges(height, width, CV_8UC1);
        cudaMemcpy(naiveEdges.data, dn_nms, imgSize, cudaMemcpyDeviceToHost);
        m.naive_edge_px = cv::countNonZero(naiveEdges);
        if (writeImages) {
            cv::imwrite("gpu_naive_edges.png", naiveEdges);
            cv::Mat dil;
            cv::Mat k = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
            cv::dilate(naiveEdges, dil, k, cv::Point(-1,-1), 1);
            cv::imwrite("gpu_naive_edges_dilated.png", dil);
            cv::imwrite("gpu_naive_lane_detection.png",
                        extractLaneLinesFromEdges(dil, frame, 25, 20, 200));
        }
    }

    // Analytical bandwidth for Gaussian blur (naive):
    // Each output pixel reads 25 input bytes + reads 25 kernel floats from global mem
    // + writes 1 output byte. Total bytes moved per pixel = 25 + 25*4 + 1 = 126 bytes.
    // Naive reads kernel from global memory every call.
    {
        double bytesPerPixel_naive = 25.0 + 25.0 * 4.0 + 1.0; // 126 bytes
        double totalBytes_naive = static_cast<double>(totalPixels) * bytesPerPixel_naive;
        m.naive_blur_bw_gbs = (totalBytes_naive / 1.0e9) / (m.naive_blur_ms / 1000.0);
    }

    cudaFree(dn_input); cudaFree(dn_blur); cudaFree(dn_nms);
    cudaFree(dn_grad);  cudaFree(dn_angle); cudaFree(dn_gkernel);

    // ============================================================
    // 3) OPTIMIZED GPU PIPELINE — per-stage timing
    // ============================================================
    cudaMemcpyToSymbol(d_kernel, h_kernel, 25 * sizeof(float));

    uchar *d_input, *d_blur, *d_nms;
    float *d_grad, *d_angle;
    cudaMalloc(&d_input, imgSize);
    cudaMalloc(&d_blur,  imgSize);
    cudaMalloc(&d_nms,   imgSize);
    cudaMalloc(&d_grad,  floatSize);
    cudaMalloc(&d_angle, floatSize);
    cudaMemcpy(d_input, imgGray.data, imgSize, cudaMemcpyHostToDevice);

    std::vector<double> oBlurPer, oSobelPer, oNmsPer, oHystPer;

    m.opt_blur_ms  = timeSingleKernel([&]{ gaussianBlurGPUShared  <<<grid,block>>>(d_input,d_blur,width,height); }, oBlurPer);
    m.opt_sobel_ms = timeSingleKernel([&]{ sobelGradientGPU       <<<grid,block>>>(d_blur, d_grad,d_angle,width,height); }, oSobelPer);
    m.opt_nms_ms   = timeSingleKernel([&]{ nonMaxSuppressionGPU   <<<grid,block>>>(d_grad, d_angle,d_nms,width,height); }, oNmsPer);
    m.opt_hyst_ms  = timeSingleKernel([&]{ hysteresisGPU          <<<grid,block>>>(d_nms,  width,height,GPU_LOW_T,GPU_HIGH_T); }, oHystPer);

    // Combined pipeline timing
    {
        std::vector<double> pipelineSamples;
        pipelineSamples.reserve(KERNEL_ITERS);
        gaussianBlurGPUShared<<<grid,block>>>(d_input,d_blur,width,height);
        sobelGradientGPU     <<<grid,block>>>(d_blur, d_grad,d_angle,width,height);
        nonMaxSuppressionGPU <<<grid,block>>>(d_grad, d_angle,d_nms,width,height);
        hysteresisGPU        <<<grid,block>>>(d_nms,  width,height,GPU_LOW_T,GPU_HIGH_T);
        cudaDeviceSynchronize();

        cudaEvent_t pS, pE;
        cudaEventCreate(&pS); cudaEventCreate(&pE);
        for (int i = 0; i < KERNEL_ITERS; ++i) {
            cudaEventRecord(pS);
            gaussianBlurGPUShared<<<grid,block>>>(d_input,d_blur,width,height);
            sobelGradientGPU     <<<grid,block>>>(d_blur, d_grad,d_angle,width,height);
            nonMaxSuppressionGPU <<<grid,block>>>(d_grad, d_angle,d_nms,width,height);
            hysteresisGPU        <<<grid,block>>>(d_nms,  width,height,GPU_LOW_T,GPU_HIGH_T);
            cudaEventRecord(pE);
            cudaEventSynchronize(pE);
            float ms = 0.0f; cudaEventElapsedTime(&ms, pS, pE);
            pipelineSamples.push_back(static_cast<double>(ms));
        }
        cudaEventDestroy(pS); cudaEventDestroy(pE);
        m.opt_ms        = std::accumulate(pipelineSamples.begin(), pipelineSamples.end(), 0.0) / KERNEL_ITERS;
        m.opt_stddev_ms = stddev(pipelineSamples);
    }

    // Edge pixel count
    {
        cv::Mat gpuEdges(height, width, CV_8UC1);
        cudaMemcpy(gpuEdges.data, d_nms, imgSize, cudaMemcpyDeviceToHost);
        m.opt_edge_px = cv::countNonZero(gpuEdges);
        if (writeImages) {
            cv::imwrite("gpu_optimized_edges.png", gpuEdges);
            cv::Mat dil;
            cv::Mat k = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
            cv::dilate(gpuEdges, dil, k, cv::Point(-1,-1), 1);
            cv::imwrite("gpu_optimized_edges_dilated.png", dil);
            cv::imwrite("gpu_optimized_lane_detection.png",
                        extractLaneLinesFromEdges(dil, frame, 25, 20, 200));
        }
    }

    // Analytical bandwidth for optimized Gaussian blur:
    // Kernel weights come from __constant__ cache (no global reads for kernel).
    // Each output pixel reads 25 input bytes from shared mem (loaded once per block)
    // + writes 1 output byte. Global traffic = tile load + output write.
    // Tile size = (BLOCK_SIZE+4)^2 bytes loaded from global per block.
    // Per output pixel: (TILE_W*TILE_H) / (BLOCK_SIZE^2) input bytes + 1 output byte.
    {
        const double tileW = BLOCK_SIZE + 4.0, tileH = BLOCK_SIZE + 4.0;
        const double inputBytesPerPx = (tileW * tileH) / (BLOCK_SIZE * BLOCK_SIZE); // ~1.5625
        const double bytesPerPixel_opt = inputBytesPerPx + 1.0; // read tile share + write output
        double totalBytes_opt = static_cast<double>(totalPixels) * bytesPerPixel_opt;
        m.opt_blur_bw_gbs = (totalBytes_opt / 1.0e9) / (m.opt_blur_ms / 1000.0);
    }

    cudaFree(d_input); cudaFree(d_blur); cudaFree(d_nms);
    cudaFree(d_grad);  cudaFree(d_angle);

    // ============================================================
    // Derived metrics
    // ============================================================
    m.naive_mpps = (totalPixels / 1.0e6) / (m.naive_ms / 1000.0);
    m.opt_mpps   = (totalPixels / 1.0e6) / (m.opt_ms   / 1000.0);
    m.naive_fps  = 1000.0 / m.naive_ms;
    m.opt_fps    = 1000.0 / m.opt_ms;

    m.cpu_over_naive  = (m.naive_ms > 0.0) ? (m.cpu_ms / m.naive_ms) : 0.0;
    m.cpu_over_opt    = (m.opt_ms   > 0.0) ? (m.cpu_ms / m.opt_ms)   : 0.0;
    m.naive_over_opt  = (m.opt_ms   > 0.0) ? (m.naive_ms / m.opt_ms) : 0.0;

    // ============================================================
    // Console output
    // ============================================================
    const char* tag = (callRun == 0) ? "[COLD]" : "[WARM]";
    std::cout << "\n══ Run " << callRun << " " << tag << " ════════════════════════════════\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  Pipeline        │ Time (ms)  │ ±Stddev │ FPS      │ MPix/s  │ Speedup\n";
    std::cout << "  ────────────────┼────────────┼─────────┼──────────┼─────────┼────────\n";
    std::cout << "  CPU             │ " << std::setw(10) << m.cpu_ms
              << " │ " << std::setw(7) << m.cpu_stddev_ms
              << " │ " << std::setw(8) << m.cpu_fps
              << " │ " << std::setw(7) << m.cpu_mpps
              << " │   1.00x\n";
    std::cout << "  GPU Naive       │ " << std::setw(10) << m.naive_ms
              << " │ " << std::setw(7) << m.naive_stddev_ms
              << " │ " << std::setw(8) << m.naive_fps
              << " │ " << std::setw(7) << m.naive_mpps
              << " │ " << std::setw(6) << m.cpu_over_naive << "x\n";
    std::cout << "  GPU Optimized   │ " << std::setw(10) << m.opt_ms
              << " │ " << std::setw(7) << m.opt_stddev_ms
              << " │ " << std::setw(8) << m.opt_fps
              << " │ " << std::setw(7) << m.opt_mpps
              << " │ " << std::setw(6) << m.cpu_over_opt << "x\n";

    std::cout << "\n  Per-stage GPU timings (avg ms per kernel call):\n";
    std::cout << "  Stage       │ Naive      │ Optimized  │ Stage speedup\n";
    std::cout << "  ────────────┼────────────┼────────────┼──────────────\n";
    auto stageSpeedup = [](double a, double b){ return (b > 0.0) ? a/b : 0.0; };
    std::cout << "  Gauss blur  │ " << std::setw(10) << m.naive_blur_ms
              << " │ " << std::setw(10) << m.opt_blur_ms
              << " │ " << std::setw(12) << stageSpeedup(m.naive_blur_ms, m.opt_blur_ms) << "x\n";
    std::cout << "  Sobel grad  │ " << std::setw(10) << m.naive_sobel_ms
              << " │ " << std::setw(10) << m.opt_sobel_ms
              << " │ " << std::setw(12) << stageSpeedup(m.naive_sobel_ms, m.opt_sobel_ms) << "x\n";
    std::cout << "  NMS         │ " << std::setw(10) << m.naive_nms_ms
              << " │ " << std::setw(10) << m.opt_nms_ms
              << " │ " << std::setw(12) << stageSpeedup(m.naive_nms_ms, m.opt_nms_ms) << "x\n";
    std::cout << "  Hysteresis  │ " << std::setw(10) << m.naive_hyst_ms
              << " │ " << std::setw(10) << m.opt_hyst_ms
              << " │ " << std::setw(12) << stageSpeedup(m.naive_hyst_ms, m.opt_hyst_ms) << "x\n";

    std::cout << "\n  Gaussian blur bandwidth:\n";
    std::cout << "  Naive  blur BW : " << m.naive_blur_bw_gbs << " GB/s  "
              << "(kernel weights from global memory — 126 bytes/px)\n";
    std::cout << "  Opt    blur BW : " << m.opt_blur_bw_gbs   << " GB/s  "
              << "(kernel from __constant__, tile from shared — ~2.6 bytes/px)\n";

    std::cout << "\n  Edge pixel counts (output quality / correctness):\n";
    std::cout << "  CPU: " << m.cpu_edge_px
              << "  |  Naive GPU: " << m.naive_edge_px
              << "  |  Opt GPU: " << m.opt_edge_px << "\n";

    return m;
}

// ================================================================
// main
// ================================================================
int main(int argc, char** argv) {
    const std::string inputPath = (argc >= 2) ? argv[1] : "road.png";

    cv::Mat frame = cv::imread(inputPath, cv::IMREAD_COLOR);
    if (frame.empty()) { std::cerr << "Image not found: " << inputPath << "\n"; return -1; }

    // Preprocessing — done once outside the benchmark loop
    cv::Mat roiMask = makeRoadRoiMask(frame.size());
    cv::Mat roiBgr; frame.copyTo(roiBgr, roiMask);
    cv::imwrite("roi_bgr.png", roiBgr);

    cv::Mat colorMask = whiteYellowMask(roiBgr);
    cv::imwrite("mask_white_yellow.png", colorMask);

    cv::Mat filtered; roiBgr.copyTo(filtered, colorMask);
    cv::imwrite("filtered_white_yellow.png", filtered);

    cv::Mat imgGray;
    cv::cvtColor(filtered, imgGray, cv::COLOR_BGR2GRAY);
    cv::imwrite("original.png", frame);

    const int width = imgGray.cols, height = imgGray.rows;

    float h_kernel[25] = {
         2,  4,  5,  4,  2,
         4,  9, 12,  9,  4,
         5, 12, 15, 12,  5,
         4,  9, 12,  9,  4,
         2,  4,  5,  4,  2
    };
    for (int i = 0; i < 25; i++) h_kernel[i] /= 159.0f;

    std::cout << "Image: " << inputPath << "  (" << width << "x" << height << ")\n";
    std::cout << "Benchmark: " << NUM_RUNS << " runs, "
              << CPU_REPEATS << " CPU iters, " << KERNEL_ITERS << " GPU iters each\n";

    for (int run = 0; run < NUM_RUNS; ++run) {
        RunMetrics m = runBenchmark(inputPath, imgGray, frame, h_kernel,
                                    width, height, run, (run == 0));
        appendCsvRow("timing_results.csv", inputPath, width, height, run, m);
    }

    std::cout << "\nAll " << NUM_RUNS << " runs complete. Results in timing_results.csv\n";
    return 0;
}
