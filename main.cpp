#include <iostream>
#include <algorithm>
#include <vector>
#include <random>
#include <chrono>
#include <cstdint>
#include <cassert>
#include <iomanip>
#include <fstream>

#include "EasyBMP/EasyBMP.h"
#include <sycl/sycl.hpp>
#include "medianFilter.h"
#include "medianFilterGPU.h"

bool compare_images(const uint8_t* left, const uint8_t* right, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        if (left[i] != right[i]) return false;
    }
    return true;
}

void warmupGPU(sycl::queue q) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(1), [=](sycl::id<1> idx) {});
        });
    q.wait();
}

void create_BMP(BMP& inputBMP, BMP& resultBMP, const uint8_t* resultImage) {
    const int width = inputBMP.TellWidth();
    const int height = inputBMP.TellHeight();

    resultBMP.SetSize(width, height);
    resultBMP.SetBitDepth(inputBMP.TellBitDepth());

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            size_t idx = (y * width + x) * 3;
            resultBMP(x, y)->Red = resultImage[idx + 0];
            resultBMP(x, y)->Green = resultImage[idx + 1];
            resultBMP(x, y)->Blue = resultImage[idx + 2];
        }
    }
}

int main() {
    BMP inputBMP;
    std::string filename = "test.bmp";
    inputBMP.ReadFromFile(filename.c_str());
    const int width = inputBMP.TellWidth();
    const int height = inputBMP.TellHeight();

    const size_t imageSize = width * height * 3;
    uint8_t* inputImage = new uint8_t[imageSize];

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            size_t idx = (y * width + x) * 3;
            inputImage[idx + 0] = inputBMP.GetPixel(x, y).Red;
            inputImage[idx + 1] = inputBMP.GetPixel(x, y).Green;
            inputImage[idx + 2] = inputBMP.GetPixel(x, y).Blue;
        }
    }

    const size_t stride = width * 3;

    uint8_t* resultCPU = new uint8_t[imageSize];
    auto start1 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < 100; ++i) {
        MedianFilter::median_filter_3x3(inputImage, resultCPU, width, height, stride);
    }
    auto end1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);
    std::cout << "Single thread version: " << duration1.count() << " ms" << std::endl;

    sycl::queue q;
    warmupGPU(q);

    uint8_t* resultGPU_naive = new uint8_t[imageSize];
    auto start2 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < 100; ++i) {
        MedianFilterGPU::median_filter_3x3_naive(inputImage, resultGPU_naive, width, height, stride, q);
    }
    auto end2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2);
    std::cout << "GPU version (naive): " << duration2.count() << " ms" << std::endl;

    uint8_t* resultGPU_not_naive = new uint8_t[imageSize];
    auto start3 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < 100; ++i) {
        MedianFilterGPU::median_filter_3x3_not_naive(inputImage, resultGPU_not_naive, width, height, stride, q);
    }
    auto end3 = std::chrono::high_resolution_clock::now();
    auto duration3 = std::chrono::duration_cast<std::chrono::milliseconds>(end3 - start3);
    std::cout << "GPU version (not naive): " << duration3.count() << " ms" << std::endl;

    assert(compare_images(resultCPU, resultGPU_naive, imageSize));
    assert(compare_images(resultCPU, resultGPU_not_naive, imageSize));

    BMP resultBMP;
    create_BMP(inputBMP, resultBMP, resultGPU_naive);
    resultBMP.WriteToFile("result.bmp");

    delete[] inputImage;
    delete[] resultCPU;
    delete[] resultGPU_naive;
    delete[] resultGPU_not_naive;

    return 0;
}