#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <cmath>

#define BLOCK_SIZE 16

// ---------------- GPU Kernels ----------------
__constant__ float d_kernel[25]; // 5x5 Gaussian

__global__ void gaussianBlurGPU(const unsigned char* input,
                                unsigned char* output,
                                int width,
                                int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 2 || y < 2 || x >= width - 2 || y >= height - 2) return;

    float sum = 0.0f;
    for (int ky = -2; ky <= 2; ky++) {
        for (int kx = -2; kx <= 2; kx++) {
            sum += input[(y + ky) * width + (x + kx)] * d_kernel[(ky + 2) * 5 + (kx + 2)];
        }
    }
    output[y * width + x] = static_cast<unsigned char>(sum);
}

__global__ void sobelGradientGPU(const unsigned char* input,
                                float* grad,
                                float* angle,
                                int width,
                                int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 1 || y < 1 || x >= width - 1 || y >= height - 1) return;

    int Gx =
        -input[(y - 1) * width + (x - 1)] - 2 * input[y * width + (x - 1)] -
        input[(y + 1) * width + (x - 1)] + input[(y - 1) * width + (x + 1)] +
        2 * input[y * width + (x + 1)] + input[(y + 1) * width + (x + 1)];

    int Gy =
        -input[(y - 1) * width + (x - 1)] - 2 * input[(y - 1) * width + x] -
        input[(y - 1) * width + (x + 1)] + input[(y + 1) * width + (x - 1)] +
        2 * input[(y + 1) * width + x] + input[(y + 1) * width + (x + 1)];

    float fx = static_cast<float>(Gx);
    float fy = static_cast<float>(Gy);

    grad[y * width + x] = sqrtf(fx * fx + fy * fy);
    angle[y * width + x] = atan2f(fy, fx);
}

__global__ void nonMaxSuppressionGPU(const float* grad,
                                    const float* angle,
                                    unsigned char* output,
                                    int width,
                                    int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 1 || y < 1 || x >= width - 1 || y >= height - 1) return;

    float a = angle[y * width + x];
    float g = grad[y * width + x];
    float q = 0.0f, r = 0.0f;

    if ((a >= -M_PI / 8 && a <= M_PI / 8) || (a <= -7 * M_PI / 8) || (a >= 7 * M_PI / 8)) {
        q = grad[y * width + (x + 1)];
        r = grad[y * width + (x - 1)];
    } else if ((a >= M_PI / 8 && a < 3 * M_PI / 8) ||
               (a <= -5 * M_PI / 8 && a > -7 * M_PI / 8)) {
        q = grad[(y + 1) * width + (x - 1)];
        r = grad[(y - 1) * width + (x + 1)];
    } else if ((a >= 3 * M_PI / 8 && a <= 5 * M_PI / 8) ||
               (a <= -3 * M_PI / 8 && a >= -5 * M_PI / 8)) {
        q = grad[(y + 1) * width + x];
        r = grad[(y - 1) * width + x];
    } else {
        q = grad[(y - 1) * width + (x - 1)];
        r = grad[(y + 1) * width + (x + 1)];
    }

    output[y * width + x] = (g >= q && g >= r) ? static_cast<unsigned char>(fminf(g, 255.0f)) : 0;
}

__global__ void hysteresisGPU(unsigned char* edges,
                             int width,
                             int height,
                             unsigned char low,
                             unsigned char high) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    unsigned char val = edges[y * width + x];
    if (val >= high) edges[y * width + x] = 255;
    else if (val < low) edges[y * width + x] = 0;
}

static cv::Mat makeRoadRoiMask(cv::Size sz) {
    cv::Mat mask = cv::Mat::zeros(sz, CV_8UC1);

    int w = sz.width;
    int h = sz.height;

    // Expand ROI to the left by reducing the left x-ratios
    std::vector<cv::Point> poly = {
        cv::Point(static_cast<int>(w * 0.00), h),                    // was 0.05
        cv::Point(static_cast<int>(w * 0.15), static_cast<int>(h * 0.60)), // widened further to the left
        cv::Point(static_cast<int>(w * 0.55), static_cast<int>(h * 0.60)),
        cv::Point(static_cast<int>(w * 0.95), h),
    };

    cv::fillConvexPoly(mask, poly, cv::Scalar(255));
    return mask;
}

static cv::Mat whiteYellowMask(const cv::Mat& bgr) {
    // Yellow in HSV
    cv::Mat hsv;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);

    cv::Mat yellowMask;
    cv::inRange(hsv, cv::Scalar(15, 60, 60), cv::Scalar(45, 255, 255), yellowMask);

    // White in BGR: bright and fairly neutral (all channels high)
    std::vector<cv::Mat> ch(3);
    cv::split(bgr, ch);
    cv::Mat whiteMask = (ch[0] > 180) & (ch[1] > 180) & (ch[2] > 180);

    cv::Mat mask;
    cv::bitwise_or(yellowMask, whiteMask, mask);

    // Light closing to connect broken paint
    cv::Mat k = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, k);

    return mask;
}

// Lane overlay using edges->Hough (parameters configurable)
static cv::Mat extractLaneLinesFromEdges(const cv::Mat& edges,
                                        const cv::Mat& origBgr,
                                        int houghThresh,
                                        int minLen,
                                        int maxGap) {
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(edges, lines, 1, CV_PI / 180, houghThresh, minLen, maxGap);

    cv::Mat laneImg = cv::Mat::zeros(origBgr.size(), CV_8UC3);
    for (const auto& l : lines) {
        cv::line(laneImg, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 0, 255), 3);
    }

    cv::Mat out;
    cv::addWeighted(origBgr, 0.85, laneImg, 1.0, 0.0, out);
    return out;
}

// ---------------- Main ----------------
int main() {
    cv::Mat frame = cv::imread("solidWhiteCurve.jpg", cv::IMREAD_COLOR);
    if (frame.empty()) {
        std::cerr << "Image not found!\n";
        return -1;
    }

    // Whole image (ROI = whole image)
    cv::Mat roiMask = makeRoadRoiMask(frame.size());
    cv::Mat roiBgr;
    frame.copyTo(roiBgr, roiMask);
    cv::imwrite("roi_bgr.png", roiBgr);

    // Color mask on whole image
    cv::Mat colorMask = whiteYellowMask(roiBgr);
    cv::imwrite("mask_white_yellow.png", colorMask);

    cv::Mat filtered;
    roiBgr.copyTo(filtered, colorMask);
    cv::imwrite("filtered_white_yellow.png", filtered);

    // For GPU pipeline (expects grayscale)
    cv::Mat imgGray;
    cv::cvtColor(filtered, imgGray, cv::COLOR_BGR2GRAY);

    int width = imgGray.cols;
    int height = imgGray.rows;
    size_t imgSize = static_cast<size_t>(width) * static_cast<size_t>(height) * sizeof(uchar);

    // CPU: OpenCV Canny (true hysteresis)
    const int CPU_CANNY_LOW = 50;
    const int CPU_CANNY_HIGH = 150;

    // GPU: simplified hysteresis, needs lower thresholds
    const uchar GPU_LOW_T = 20;
    const uchar GPU_HIGH_T = 80;

    // ---------------- CPU edges ----------------
    cv::Mat cpuEdges;
    auto t1 = std::chrono::high_resolution_clock::now();
    {
        cv::Mat blur;
        cv::GaussianBlur(imgGray, blur, cv::Size(5, 5), 1.2);
        cv::Canny(blur, cpuEdges, CPU_CANNY_LOW, CPU_CANNY_HIGH);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(t2 - t1).count();
    std::cout << "CPU edge pipeline time: " << cpu_time << " ms\n";
    cv::imwrite("cpu_edges.png", cpuEdges);

    // CPU Hough (stricter)
    cv::Mat cpuLanes = extractLaneLinesFromEdges(cpuEdges, frame, 40, 40, 150);
    cv::imwrite("cpu_lane_detection.png", cpuLanes);

    // ---------------- GPU edges (custom pipeline) ----------------
    float h_kernel[25] = {
        2, 4, 5, 4, 2,
        4, 9, 12, 9, 4,
        5, 12, 15, 12, 5,
        4, 9, 12, 9, 4,
        2, 4, 5, 4, 2
    };
    for (int i = 0; i < 25; i++) h_kernel[i] /= 159.0f;
    cudaMemcpyToSymbol(d_kernel, h_kernel, 25 * sizeof(float));

    uchar *d_input = nullptr, *d_blur = nullptr, *d_nms = nullptr;
    float *d_grad = nullptr, *d_angle = nullptr;

    cudaMalloc(&d_input, imgSize);
    cudaMalloc(&d_blur, imgSize);
    cudaMalloc(&d_nms, imgSize);
    cudaMalloc(&d_grad, static_cast<size_t>(width) * static_cast<size_t>(height) * sizeof(float));
    cudaMalloc(&d_angle, static_cast<size_t>(width) * static_cast<size_t>(height) * sizeof(float));

    cudaMemcpy(d_input, imgGray.data, imgSize, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    auto t3 = std::chrono::high_resolution_clock::now();
    gaussianBlurGPU<<<grid, block>>>(d_input, d_blur, width, height);
    sobelGradientGPU<<<grid, block>>>(d_blur, d_grad, d_angle, width, height);
    nonMaxSuppressionGPU<<<grid, block>>>(d_grad, d_angle, d_nms, width, height);
    hysteresisGPU<<<grid, block>>>(d_nms, width, height, GPU_LOW_T, GPU_HIGH_T);
    cudaDeviceSynchronize();
    auto t4 = std::chrono::high_resolution_clock::now();

    double gpu_time = std::chrono::duration<double, std::milli>(t4 - t3).count();
    std::cout << "GPU edge pipeline time: " << gpu_time << " ms\n";
    if (gpu_time > 0.0) {
        std::cout << "Edge speedup (CPU/GPU): " << (cpu_time / gpu_time) << "x\n";
    }

    cv::Mat gpuEdges(height, width, CV_8UC1);
    cudaMemcpy(gpuEdges.data, d_nms, imgSize, cudaMemcpyDeviceToHost);
    cv::imwrite("gpu_edges.png", gpuEdges);

    // DILATE GPU edges to connect broken segments (helps Hough)
    cv::Mat gpuEdgesDilated;
    {
        cv::Mat k = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::dilate(gpuEdges, gpuEdgesDilated, k, cv::Point(-1, -1), 1);
    }
    cv::imwrite("gpu_edges_dilated.png", gpuEdgesDilated);

    // GPU Hough (looser)
    cv::Mat gpuLanes = extractLaneLinesFromEdges(gpuEdgesDilated, frame, 25, 20, 200);
    cv::imwrite("gpu_lane_detection.png", gpuLanes);

    cv::imwrite("original.png", frame);

    std::cout << "Saved: original.png, roi_bgr.png, mask_white_yellow.png, filtered_white_yellow.png, "
                 "cpu_edges.png, gpu_edges.png, gpu_edges_dilated.png, cpu_lane_detection.png, "
                 "gpu_lane_detection.png\n";

    cudaFree(d_input);
    cudaFree(d_blur);
    cudaFree(d_nms);
    cudaFree(d_grad);
    cudaFree(d_angle);

    return 0;
}
