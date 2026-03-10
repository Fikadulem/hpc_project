// main.cpp
// Detect (most) white road-marking lines in an image, smooth/merge them, and export as GeoJSON (pixel coordinates).
//
// Build:
//   g++ -std=c++17 detect_lane_lines.cpp -o white_lines `pkg-config --cflags --libs opencv4`
//
// Run:
//   ./white_lines road.png
//
// Outputs:
//   debug_mask_hsv.png
//   debug_mask_lab.png
//   debug_mask_combined.png
//   debug_edges.png
//   white_segments_raw.png
//   white_segments_smooth.png
//   white_lines.geojson

#include <opencv2/opencv.hpp>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

struct Segment {
    int x1, y1, x2, y2;
};

static void saveTextFile(const std::string& path, const std::string& content) {
    std::ofstream f(path, std::ios::out | std::ios::binary);
    if (!f) throw std::runtime_error("Failed to open output file: " + path);
    f << content;
}

static cv::Mat whiteMaskHSV(const cv::Mat& bgr, int sMax = 90, int vMin = 170) {
    cv::Mat hsv;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);

    std::vector<cv::Mat> ch;
    cv::split(hsv, ch);
    const cv::Mat& s = ch[1];
    const cv::Mat& v = ch[2];

    cv::Mat sMask, vMask, mask;
    cv::threshold(s, sMask, sMax, 255, cv::THRESH_BINARY_INV);  // s <= sMax
    cv::threshold(v, vMask, vMin, 255, cv::THRESH_BINARY);      // v >= vMin
    cv::bitwise_and(sMask, vMask, mask);

    return mask;
}

static cv::Mat whiteMaskLAB(const cv::Mat& bgr, int Lmin = 190, int abMaxDev = 22) {
    cv::Mat lab;
    cv::cvtColor(bgr, lab, cv::COLOR_BGR2Lab);

    std::vector<cv::Mat> ch;
    cv::split(lab, ch);
    const cv::Mat& L = ch[0];
    const cv::Mat& A = ch[1];
    const cv::Mat& B = ch[2];

    cv::Mat Lmask;
    cv::threshold(L, Lmask, Lmin, 255, cv::THRESH_BINARY);  // L >= Lmin

    cv::Mat aDev, bDev, aMask, bMask, abMask;
    cv::absdiff(A, cv::Scalar(128), aDev);
    cv::absdiff(B, cv::Scalar(128), bDev);

    cv::threshold(aDev, aMask, abMaxDev, 255, cv::THRESH_BINARY_INV);
    cv::threshold(bDev, bMask, abMaxDev, 255, cv::THRESH_BINARY_INV);
    cv::bitwise_and(aMask, bMask, abMask);

    cv::Mat mask;
    cv::bitwise_and(Lmask, abMask, mask);
    return mask;
}

static cv::Mat postProcessMask(const cv::Mat& maskIn, bool connectDashes = true) {
    cv::Mat mask = maskIn.clone();

    // Safer than MORPH_OPEN for thin lines:
    cv::medianBlur(mask, mask, 3);

    if (connectDashes) {
        cv::Mat k9 = cv::getStructuringElement(cv::MORPH_ELLIPSE, {9, 9});
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, k9, cv::Point(-1, -1), 2);
    } else {
        cv::Mat k5 = cv::getStructuringElement(cv::MORPH_ELLIPSE, {5, 5});
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, k5, cv::Point(-1, -1), 1);
    }

    return mask;
}

static std::vector<Segment> maskToLineSegments(
    const cv::Mat& mask,
    int canny1 = 30,
    int canny2 = 120,
    int houghThresh = 25,
    int minLen = 20,
    int maxGap = 35
) {
    cv::Mat edges;
    cv::Canny(mask, edges, canny1, canny2);

    // Connect small breaks in edges
    cv::Mat k3 = cv::getStructuringElement(cv::MORPH_RECT, {3, 3});
    cv::morphologyEx(edges, edges, cv::MORPH_CLOSE, k3, cv::Point(-1, -1), 1);

    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(edges, lines, 1.0, CV_PI / 180.0, houghThresh, minLen, maxGap);

    std::vector<Segment> segs;
    segs.reserve(lines.size());
    for (const auto& l : lines) segs.push_back({l[0], l[1], l[2], l[3]});

    // Save for debugging (caller can also save)
    cv::imwrite("debug_edges.png", edges);

    return segs;
}

struct SegFeat {
    double theta;   // [0, pi)
    double a, b, c; // normalized ax + by + c = 0
};

static SegFeat segmentFeat(const Segment& s) {
    const double x1 = s.x1, y1 = s.y1, x2 = s.x2, y2 = s.y2;
    const double dx = x2 - x1;
    const double dy = y2 - y1;

    double theta = std::atan2(dy, dx);
    if (theta < 0.0) theta += CV_PI;

    // ax + by + c = 0
    double a = dy;
    double b = -dx;
    double c = -(a * x1 + b * y1);

    const double n = std::hypot(a, b);
    if (n > 1e-6) {
        a /= n;
        b /= n;
        c /= n;
    }

    return {theta, a, b, c};
}

static std::vector<Segment> mergeSimilarSegments(
    const std::vector<Segment>& segments,
    double angleThreshDeg = 12.0,
    double distThreshPx = 35.0
) {
    if (segments.empty()) return {};

    const double angThr = angleThreshDeg * CV_PI / 180.0;

    std::vector<SegFeat> feats;
    feats.reserve(segments.size());
    for (const auto& s : segments) feats.push_back(segmentFeat(s));

    std::vector<bool> used(segments.size(), false);
    std::vector<Segment> merged;

    for (size_t i = 0; i < segments.size(); ++i) {
        if (used[i]) continue;

        const auto& fi = feats[i];
        std::vector<cv::Point2f> pts;
        pts.reserve(64);

        pts.emplace_back((float)segments[i].x1, (float)segments[i].y1);
        pts.emplace_back((float)segments[i].x2, (float)segments[i].y2);
        used[i] = true;

        for (size_t j = i + 1; j < segments.size(); ++j) {
            if (used[j]) continue;

            const auto& fj = feats[j];

            double dtheta = std::abs(fi.theta - fj.theta);
            dtheta = std::min(dtheta, CV_PI - dtheta);
            if (dtheta > angThr) continue;

            const double x3 = segments[j].x1, y3 = segments[j].y1;
            const double x4 = segments[j].x2, y4 = segments[j].y2;

            const double d1 = std::abs(fi.a * x3 + fi.b * y3 + fi.c);
            const double d2 = std::abs(fi.a * x4 + fi.b * y4 + fi.c);
            if (std::min(d1, d2) > distThreshPx) continue;

            pts.emplace_back((float)x3, (float)y3);
            pts.emplace_back((float)x4, (float)y4);
            used[j] = true;
        }

        if (pts.size() < 2) continue;

        cv::Vec4f line;
        cv::fitLine(pts, line, cv::DIST_L2, 0, 0.01, 0.01);
        const float vx = line[0], vy = line[1], x0 = line[2], y0 = line[3];

        cv::Point2f d(vx, vy);
        cv::Point2f p0(x0, y0);

        float tMin = std::numeric_limits<float>::infinity();
        float tMax = -std::numeric_limits<float>::infinity();

        for (const auto& p : pts) {
            const cv::Point2f diff = p - p0;
            const float t = diff.x * d.x + diff.y * d.y; // projection scalar
            tMin = std::min(tMin, t);
            tMax = std::max(tMax, t);
        }

        const cv::Point2f p1 = p0 + d * tMin;
        const cv::Point2f p2 = p0 + d * tMax;

        merged.push_back({
            (int)std::lround(p1.x), (int)std::lround(p1.y),
            (int)std::lround(p2.x), (int)std::lround(p2.y)
        });
    }

    return merged;
}

static cv::Mat drawSegments(const cv::Mat& bgr, const std::vector<Segment>& segs, const cv::Scalar& color,
                            int thickness = 3) {
    cv::Mat out = bgr.clone();
    for (const auto& s : segs) {
        cv::line(out, {s.x1, s.y1}, {s.x2, s.y2}, color, thickness, cv::LINE_AA);
    }
    return out;
}

static std::string segmentsToGeoJsonPixel(const std::vector<Segment>& segs, int imageHeight, bool flipY = true) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(6);
    os << "{\n  \"type\": \"FeatureCollection\",\n  \"features\": [\n";

    for (size_t i = 0; i < segs.size(); ++i) {
        const auto& s = segs[i];

        const int y1g = flipY ? (imageHeight - 1 - s.y1) : s.y1;
        const int y2g = flipY ? (imageHeight - 1 - s.y2) : s.y2;
        const double len = std::hypot((double)s.x2 - s.x1, (double)s.y2 - s.y1);

        os << "    {\n"
           << "      \"type\": \"Feature\",\n"
           << "      \"properties\": {\"id\": " << i << ", \"length_px\": " << len << "},\n"
           << "      \"geometry\": {\n"
           << "        \"type\": \"LineString\",\n"
           << "        \"coordinates\": [["
           << (double)s.x1 << ", " << (double)y1g << "], ["
           << (double)s.x2 << ", " << (double)y2g << "]]\n"
           << "      }\n"
           << "    }" << (i + 1 < segs.size() ? "," : "") << "\n";
    }

    os << "  ]\n}\n";
    return os.str();
}

int main(int argc, char** argv) {
    try {
        const std::string inputPath = (argc >= 2) ? argv[1] : "road.png";

        cv::Mat img = cv::imread(inputPath);
        if (img.empty()) {
            std::cerr << "Could not read image: " << inputPath << "\n";
            return 1;
        }

        // 1) White pixel classification (two masks) + union
        cv::Mat maskHSV = whiteMaskHSV(img, 90, 170);
        cv::Mat maskLAB = whiteMaskLAB(img, 190, 22);

        cv::imwrite("debug_mask_hsv_2.png", maskHSV);
        cv::imwrite("debug_mask_lab_2.png", maskLAB);

        cv::Mat maskUnion;
        cv::bitwise_or(maskHSV, maskLAB, maskUnion);

        // 2) Post-process to keep thin lines + connect dashes
        cv::Mat mask = postProcessMask(maskUnion, true);
        cv::imwrite("debug_mask_combined_2.png", mask);

        // 3) Extract line segments (Hough)
        std::vector<Segment> segments = maskToLineSegments(mask);

        // 4) Smooth/merge segments into longer lines
        std::vector<Segment> smooth = mergeSimilarSegments(segments, 12.0, 35.0);

        std::cout << "Detected segments: " << segments.size() << " | Smoothed: " << smooth.size() << "\n";

        // Debug drawings
        cv::imwrite("white_segments_raw.png", drawSegments(img, segments, cv::Scalar(0, 255, 255), 2));
        cv::imwrite("white_segments_smooth.png", drawSegments(img, smooth, cv::Scalar(0, 0, 255), 4));

        // 5) GeoJSON export in pixel coordinates
        const std::string geojson = segmentsToGeoJsonPixel(smooth, img.rows, true);
        saveTextFile("white_lines.geojson", geojson);

        std::cout << "Saved: white_lines.geojson\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 2;
    }
}
