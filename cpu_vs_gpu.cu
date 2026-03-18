#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <cmath>

#define BLOCK_SIZE 16

// ================================================================
// NAIVE GPU KERNELS
// Straightforward CPU-to-GPU port. All data in global memory.
// No __constant__, no shared memory, no tiling.
// ================================================================

__global__ void gaussianBlurNaive(const unsigned char* input, unsigned char* output,
                                  const float* kernel5x5, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 2 || y < 2 || x >= width - 2 || y >= height - 2) return;

    float sum = 0.0f;
    for (int ky = -2; ky <= 2; ky++) {
        for (int kx = -2; kx <= 2; kx++) {
            sum += input[(y + ky) * width + (x + kx)] * kernel5x5[(ky + 2) * 5 + (kx + 2)];
        }
    }
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

    float fx = static_cast<float>(Gx);
    float fy = static_cast<float>(Gy);
    grad[y * width + x]  = sqrtf(fx * fx + fy * fy);
    angle[y * width + x] = atan2f(fy, fx);
}

__global__ void nonMaxSuppressionNaive(const float* grad, const float* angle,
                                       unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 1 || y < 1 || x >= width - 1 || y >= height - 1) return;

    float a = angle[y * width + x];
    float g = grad[y * width + x];
    float q = 0.0f, r = 0.0f;

    if ((a >= -M_PI/8 && a <= M_PI/8) || (a <= -7*M_PI/8) || (a >= 7*M_PI/8)) {
        q = grad[y*width+(x+1)]; r = grad[y*width+(x-1)];
    } else if ((a >= M_PI/8 && a < 3*M_PI/8) || (a <= -5*M_PI/8 && a > -7*M_PI/8)) {
        q = grad[(y+1)*width+(x-1)]; r = grad[(y-1)*width+(x+1)];
    } else if ((a >= 3*M_PI/8 && a <= 5*M_PI/8) || (a <= -3*M_PI/8 && a >= -5*M_PI/8)) {
        q = grad[(y+1)*width+x]; r = grad[(y-1)*width+x];
    } else {
        q = grad[(y-1)*width+(x-1)]; r = grad[(y+1)*width+(x+1)];
    }
    output[y * width + x] = (g >= q && g >= r) ? static_cast<unsigned char>(fminf(g, 255.0f)) : 0;
}

__global__ void hysteresisNaive(unsigned char* edges, int width, int height,
                                unsigned char low, unsigned char high) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    unsigned char val = edges[y * width + x];
    if (val >= high)     edges[y * width + x] = 255;
    else if (val < low)  edges[y * width + x] = 0;
}

// ================================================================
// OPTIMIZED GPU KERNELS
// 1) Gaussian Blur: Use __constant__ memory for kernel, shared memory tiling, and manual unrolling.
// 2) Sobel Gradient: Use register blocking to fetch neighbors once, and manual unrolling for math.
// 3) Non-Max Suppression: Quantize angles to 4 directions and use predication instead of if/else.
// 4) Hysteresis: Use predication to eliminate branches.
// ================================================================

#define TILE_SIZE 16
#define KERNEL_RADIUS 2
#define SH_DIM (TILE_SIZE + 2 * KERNEL_RADIUS)

__constant__ float d_kernel[25];

// Optimized Gaussian Blur with Shared Memory Tiling and Manual Unrolling
__global__ void gaussianBlurGPU(const unsigned char* input, unsigned char* output,
    int width, int height) {
    // 1. Shared memory for the tile plus a 2-pixel "halo" on all sides
    __shared__ unsigned char sh_tile[SH_DIM][SH_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    // 2. Collaborative Loading into Shared Memory
    for (int i = ty; i < SH_DIM; i += TILE_SIZE) {
        for (int j = tx; j < SH_DIM; j += TILE_SIZE) {
            int input_row = (int)(blockIdx.y * TILE_SIZE) + i - KERNEL_RADIUS;
            int input_col = (int)(blockIdx.x * TILE_SIZE) + j - KERNEL_RADIUS;

            if (input_row >= 0 && input_row < height && input_col >= 0 && input_col < width)
                sh_tile[i][j] = input[input_row * width + input_col];
            else
                sh_tile[i][j] = 0;
        }
    }
    __syncthreads();

    // 3. Compute the Blur with Manual Unrolling
    if (row < height && col < width) {
        float sum = 0.0f;

        // Row 0
        sum += (float)sh_tile[ty + 0][tx + 0] * d_kernel[0];
        sum += (float)sh_tile[ty + 0][tx + 1] * d_kernel[1];
        sum += (float)sh_tile[ty + 0][tx + 2] * d_kernel[2];
        sum += (float)sh_tile[ty + 0][tx + 3] * d_kernel[3];
        sum += (float)sh_tile[ty + 0][tx + 4] * d_kernel[4];

        // Row 1
        sum += (float)sh_tile[ty + 1][tx + 0] * d_kernel[5];
        sum += (float)sh_tile[ty + 1][tx + 1] * d_kernel[6];
        sum += (float)sh_tile[ty + 1][tx + 2] * d_kernel[7];
        sum += (float)sh_tile[ty + 1][tx + 3] * d_kernel[8];
        sum += (float)sh_tile[ty + 1][tx + 4] * d_kernel[9];

        // Row 2
        sum += (float)sh_tile[ty + 2][tx + 0] * d_kernel[10];
        sum += (float)sh_tile[ty + 2][tx + 1] * d_kernel[11];
        sum += (float)sh_tile[ty + 2][tx + 2] * d_kernel[12];
        sum += (float)sh_tile[ty + 2][tx + 3] * d_kernel[13];
        sum += (float)sh_tile[ty + 2][tx + 4] * d_kernel[14];

        // Row 3
        sum += (float)sh_tile[ty + 3][tx + 0] * d_kernel[15];
        sum += (float)sh_tile[ty + 3][tx + 1] * d_kernel[16];
        sum += (float)sh_tile[ty + 3][tx + 2] * d_kernel[17];
        sum += (float)sh_tile[ty + 3][tx + 3] * d_kernel[18];
        sum += (float)sh_tile[ty + 3][tx + 4] * d_kernel[19];

        // Row 4
        sum += (float)sh_tile[ty + 4][tx + 0] * d_kernel[20];
        sum += (float)sh_tile[ty + 4][tx + 1] * d_kernel[21];
        sum += (float)sh_tile[ty + 4][tx + 2] * d_kernel[22];
        sum += (float)sh_tile[ty + 4][tx + 3] * d_kernel[23];
        sum += (float)sh_tile[ty + 4][tx + 4] * d_kernel[24];

        output[row * width + col] = static_cast<unsigned char>(sum);
    }
}

// Optimized Sobel Gradient with Register Blocking and Manual Unrolling
__global__ void sobelGradientGPU(const unsigned char* input, float* grad, float* angle,
    int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check for 3x3 kernel
    if (x < 1 || y < 1 || x >= width - 1 || y >= height - 1) return;

    // 1. Fetch each neighbor exactly ONCE and store in registers
    // This reduces global memory transactions significantly
    float s00 = static_cast<float>(input[(y - 1) * width + (x - 1)]);
    float s01 = static_cast<float>(input[(y - 1) * width + x]);
    float s02 = static_cast<float>(input[(y - 1) * width + (x + 1)]);

    float s10 = static_cast<float>(input[y * width + (x - 1)]);
    // s11 (center) is not used in Sobel
    float s12 = static_cast<float>(input[y * width + (x + 1)]);

    float s20 = static_cast<float>(input[(y + 1) * width + (x - 1)]);
    float s21 = static_cast<float>(input[(y + 1) * width + x]);
    float s22 = static_cast<float>(input[(y + 1) * width + (x + 1)]);

    // 2. Perform math using the registers
    // Gx = sum of right column - sum of left column (with weights)
    float fx = -s00 - 2.0f * s10 - s20 + s02 + 2.0f * s12 + s22;

    // Gy = sum of bottom row - sum of top row (with weights)
    float fy = -s00 - 2.0f * s01 - s02 + s20 + 2.0f * s21 + s22;

    // 3. Store results
    int idx = y * width + x;
    grad[idx] = sqrtf(fx * fx + fy * fy);
    angle[idx] = atan2f(fy, fx);
}

// Optimized Non-Max Suppression with Direction Quantization and Predication
__global__ void nonMaxSuppressionGPU(const float* grad, const float* angle,
    unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 1 || y < 1 || x >= width - 1 || y >= height - 1) return;

    int idx = y * width + x;
    float a = angle[idx];
    float g = grad[idx];

    // 1. Map the continuous angle (-PI to PI) into 4 discrete directions (0, 1, 2, 3)
    // We add PI/8 to shift the ranges so that 0 is "centered" around 0 degrees.
    // We use fabsf because symmetry (0 deg is same as 180 deg for NMS)
    float shifted_angle = fabsf(a) + (3.14159265f / 8.0f);
    if (shifted_angle > 3.14159265f) shifted_angle -= 3.14159265f;

    // Map to 0 (0°), 1 (45°), 2 (90°), 3 (135°)
    int dir = static_cast<int>(shifted_angle / (3.14159265f / 4.0f)) % 4;

    float q = 0.0f, r = 0.0f;

    // 2. Simplified logic using the direction index
    // This is much easier for the GPU to schedule than the original if/else blocks
    switch (dir) {
    case 0: // Horizontal (0°)
        q = grad[y * width + (x + 1)];
        r = grad[y * width + (x - 1)];
        break;
    case 1: // Diagonal (45°)
        q = grad[(y + 1) * width + (x - 1)];
        r = grad[(y - 1) * width + (x + 1)];
        break;
    case 2: // Vertical (90°)
        q = grad[(y + 1) * width + x];
        r = grad[(y - 1) * width + x];
        break;
    case 3: // Diagonal (135°)
        q = grad[(y - 1) * width + (x - 1)];
        r = grad[(y + 1) * width + (x + 1)];
        break;
    }

    output[idx] = (g >= q && g >= r) ? static_cast<unsigned char>(fminf(g, 255.0f)) : 0;
}

// Optimized Hysteresis with Predication (No Branches)
__global__ void hysteresisGPU(unsigned char* edges, int width, int height,
    unsigned char low, unsigned char high) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    unsigned char val = edges[idx];

    // Predicated Logic: No 'if' or 'else' blocks
    // The compiler turns these into 'sel' (select) or predicated 'mov' instructions
    unsigned char is_strong = (val >= high);
    unsigned char is_weak = (val >= low);

    // If strong (>= high), result is 255. 
    // If not strong but weak (>= low), result is 127 (or val).
    // If neither, result is 0.
    unsigned char result = is_strong ? 255 : (is_weak ? 127 : 0);

    edges[idx] = result;
}


// ================================================================
// Helper functions (shared by all pipelines)
// ================================================================

static cv::Mat makeRoadRoiMask(cv::Size sz) {
    cv::Mat mask = cv::Mat::zeros(sz, CV_8UC1);
    int w = sz.width, h = sz.height;
    std::vector<cv::Point> poly = {
        cv::Point(static_cast<int>(w * 0.00), h),
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

// ================================================================
// Main — runs all three pipelines: CPU, Naive GPU, Optimized GPU
// ================================================================

int main(int argc, char** argv) {
    const std::string inputPath = (argc >= 2) ? argv[1] : "solidWhiteCurve.jpg";
    cv::Mat frame = cv::imread(inputPath, cv::IMREAD_COLOR);
    if (frame.empty()) { std::cerr << "Image not found: " << inputPath << "\n"; return -1; }

    // Preprocessing (shared)
    cv::Mat roiMask = makeRoadRoiMask(frame.size());
    cv::Mat roiBgr;
    frame.copyTo(roiBgr, roiMask);
    cv::imwrite("roi_bgr.png", roiBgr);

    cv::Mat colorMask = whiteYellowMask(roiBgr);
    cv::imwrite("mask_white_yellow.png", colorMask);
    cv::Mat filtered;
    roiBgr.copyTo(filtered, colorMask);
    cv::imwrite("filtered_white_yellow.png", filtered);

    cv::Mat imgGray;
    cv::cvtColor(filtered, imgGray, cv::COLOR_BGR2GRAY);
    int width  = imgGray.cols;
    int height = imgGray.rows;
    size_t imgSize   = static_cast<size_t>(width) * height * sizeof(uchar);
    size_t floatSize = static_cast<size_t>(width) * height * sizeof(float);

    const int   CPU_CANNY_LOW  = 50;
    const int   CPU_CANNY_HIGH = 150;
    const uchar GPU_LOW_T  = 20;
    const uchar GPU_HIGH_T = 80;

    // Gaussian kernel (host)
    float h_kernel[25] = {
        2, 4, 5, 4, 2,
        4, 9,12, 9, 4,
        5,12,15,12, 5,
        4, 9,12, 9, 4,
        2, 4, 5, 4, 2
    };
    for (int i = 0; i < 25; i++) h_kernel[i] /= 159.0f;

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // ============================================================
    // 1) CPU PIPELINE
    // ============================================================
    cv::Mat cpuEdges;
    auto t1 = std::chrono::high_resolution_clock::now();
    {
        cv::Mat blur;
        cv::GaussianBlur(imgGray, blur, cv::Size(5, 5), 1.2);
        cv::Canny(blur, cpuEdges, CPU_CANNY_LOW, CPU_CANNY_HIGH);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(t2 - t1).count();

    cv::imwrite("cpu_edges.png", cpuEdges);
    cv::Mat cpuLanes = extractLaneLinesFromEdges(cpuEdges, frame, 40, 40, 150);
    cv::imwrite("cpu_lane_detection.png", cpuLanes);

    // ============================================================
    // 2) NAIVE GPU PIPELINE
    //    Gaussian kernel in global memory. No __constant__, no
    //    shared memory, no tiling. Direct port of CPU logic.
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

    auto t5 = std::chrono::high_resolution_clock::now();
    gaussianBlurNaive<<<grid, block>>>(dn_input, dn_blur, dn_gkernel, width, height);
    sobelGradientNaive<<<grid, block>>>(dn_blur, dn_grad, dn_angle, width, height);
    nonMaxSuppressionNaive<<<grid, block>>>(dn_grad, dn_angle, dn_nms, width, height);
    hysteresisNaive<<<grid, block>>>(dn_nms, width, height, GPU_LOW_T, GPU_HIGH_T);
    cudaDeviceSynchronize();
    auto t6 = std::chrono::high_resolution_clock::now();
    double naive_time = std::chrono::duration<double, std::milli>(t6 - t5).count();

    cv::Mat naiveEdges(height, width, CV_8UC1);
    cudaMemcpy(naiveEdges.data, dn_nms, imgSize, cudaMemcpyDeviceToHost);
    cv::imwrite("gpu_naive_edges.png", naiveEdges);

    cv::Mat naiveEdgesDilated;
    {
        cv::Mat k = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::dilate(naiveEdges, naiveEdgesDilated, k, cv::Point(-1, -1), 1);
    }
    cv::imwrite("gpu_naive_edges_dilated.png", naiveEdgesDilated);
    cv::Mat naiveLanes = extractLaneLinesFromEdges(naiveEdgesDilated, frame, 25, 20, 200);
    cv::imwrite("gpu_naive_lane_detection.png", naiveLanes);

    cudaFree(dn_input); cudaFree(dn_blur); cudaFree(dn_nms);
    cudaFree(dn_grad);  cudaFree(dn_angle); cudaFree(dn_gkernel);

    // ============================================================
    // 3) OPTIMIZED GPU PIPELINE
    //    __constant__ memory for Gaussian kernel.
    //    TODO (@CyberKnight): Add shared memory tiling for blur.
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

    auto t3 = std::chrono::high_resolution_clock::now();
    gaussianBlurGPU<<<grid, block>>>(d_input, d_blur, width, height);
    sobelGradientGPU<<<grid, block>>>(d_blur, d_grad, d_angle, width, height);
    nonMaxSuppressionGPU<<<grid, block>>>(d_grad, d_angle, d_nms, width, height);
    hysteresisGPU<<<grid, block>>>(d_nms, width, height, GPU_LOW_T, GPU_HIGH_T);
    cudaDeviceSynchronize();
    auto t4 = std::chrono::high_resolution_clock::now();
    double opt_time = std::chrono::duration<double, std::milli>(t4 - t3).count();

    cv::Mat gpuEdges(height, width, CV_8UC1);
    cudaMemcpy(gpuEdges.data, d_nms, imgSize, cudaMemcpyDeviceToHost);
    cv::imwrite("gpu_optimized_edges.png", gpuEdges);

    cv::Mat gpuEdgesDilated;
    {
        cv::Mat k = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::dilate(gpuEdges, gpuEdgesDilated, k, cv::Point(-1, -1), 1);
    }
    cv::imwrite("gpu_optimized_edges_dilated.png", gpuEdgesDilated);
    cv::Mat gpuLanes = extractLaneLinesFromEdges(gpuEdgesDilated, frame, 25, 20, 200);
    cv::imwrite("gpu_optimized_lane_detection.png", gpuLanes);

    cudaFree(d_input); cudaFree(d_blur); cudaFree(d_nms);
    cudaFree(d_grad);  cudaFree(d_angle);

    // ============================================================
    // Results
    // ============================================================
    std::cout << "\n===== TIMING RESULTS =====\n";
    std::cout << "CPU edge pipeline:           " << cpu_time   << " ms\n";
    std::cout << "GPU NAIVE edge pipeline:     " << naive_time << " ms\n";
    std::cout << "GPU OPTIMIZED edge pipeline: " << opt_time   << " ms\n";
    std::cout << "\n===== SPEEDUPS =====\n";
    if (naive_time > 0.0)
        std::cout << "CPU / GPU Naive:      " << (cpu_time / naive_time) << "x\n";
    if (opt_time > 0.0)
        std::cout << "CPU / GPU Optimized:  " << (cpu_time / opt_time)   << "x\n";
    if (opt_time > 0.0 && naive_time > 0.0)
        std::cout << "GPU Naive / Optimized: " << (naive_time / opt_time) << "x\n";

    cv::imwrite("original.png", frame);
    std::cout << "\nSaved all output images.\n";
    return 0;
}
