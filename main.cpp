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

void create_BMP(BMP& inputBMP, BMP& resultBMP, uint8_t* resultPixelsRed, uint8_t* resultPixelsGreen, uint8_t* resultPixelsBlue) {
    const int width = inputBMP.TellWidth();
    const int height = inputBMP.TellHeight();

    resultBMP.SetSize(width, height);
    resultBMP.SetBitDepth(inputBMP.TellBitDepth());

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            resultBMP(x, y)->Red = resultPixelsRed[y * width + x];
            resultBMP(x, y)->Green = resultPixelsGreen[y * width + x];
            resultBMP(x, y)->Blue = resultPixelsBlue[y * width + x];
        }
    }
}

int main() {

    BMP inputBMP;
    std::string filename = "test.bmp";
    inputBMP.ReadFromFile((filename).c_str());
    const int width = inputBMP.TellWidth();
    const int height = inputBMP.TellHeight();
    uint8_t* inputPixelsRed = new uint8_t[width * height];
    uint8_t* inputPixelsGreen = new uint8_t[width * height];
    uint8_t* inputPixelsBlue = new uint8_t[width * height];

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            ebmpBYTE pixelRed = inputBMP.GetPixel(x, y).Red;
            ebmpBYTE pixelGreen = inputBMP.GetPixel(x, y).Green;
            ebmpBYTE pixelBlue = inputBMP.GetPixel(x, y).Blue;
            inputPixelsRed[y * width + x] = static_cast<uint8_t>(pixelRed);
            inputPixelsGreen[y * width + x] = static_cast<uint8_t>(pixelGreen);
            inputPixelsBlue[y * width + x] = static_cast<uint8_t>(pixelBlue);
        }
    }

    //ÎÄÍÎÏÎÒÎ×ÍÀß ÂÅÐÑÈß

    uint8_t* resultPixelsRed = new uint8_t[width * height];
    uint8_t* resultPixelsGreen = new uint8_t[width * height];
    uint8_t* resultPixelsBlue = new uint8_t[width * height];

    auto start1 = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < 10; ++i) {
        MedianFilter::median_filter_3x3(inputPixelsRed, resultPixelsRed, width, height, width);
        MedianFilter::median_filter_3x3(inputPixelsGreen, resultPixelsGreen, width, height, width);
        MedianFilter::median_filter_3x3(inputPixelsBlue, resultPixelsBlue, width, height, width);
    }

    auto end1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);
    std::cout << "Single thread version: " << duration1.count() << " ms" << std::endl;

    //GPU ÂÅÐÑÈß ÍÀÈÂÍÀß

    sycl::queue q;
    warmupGPU(q);

    uint8_t* resultPixelsRed_naive = new uint8_t[width * height];
    uint8_t* resultPixelsGreen_naive = new uint8_t[width * height];
    uint8_t* resultPixelsBlue_naive = new uint8_t[width * height];

    for (size_t i = 0; i < 10; ++i) {
        MedianFilterGPU::median_filter_3x3_naive(inputPixelsRed, resultPixelsRed_naive, width, height, width, q);
        MedianFilterGPU::median_filter_3x3_naive(inputPixelsGreen, resultPixelsGreen_naive, width, height, width, q);
        MedianFilterGPU::median_filter_3x3_naive(inputPixelsBlue, resultPixelsBlue_naive, width, height, width, q);
    }

    //GPU ÂÅÐÑÈß ÍÅ ÍÀÈÂÍÀß

    uint8_t* resultPixelsRed_not_naive = new uint8_t[width * height];
    uint8_t* resultPixelsGreen_not_naive = new uint8_t[width * height];
    uint8_t* resultPixelsBlue_not_naive = new uint8_t[width * height];

    for (size_t i = 0; i < 10; ++i) {
        MedianFilterGPU::median_filter_3x3_not_naive(inputPixelsRed, resultPixelsRed_not_naive, width, height, width, q);
        MedianFilterGPU::median_filter_3x3_not_naive(inputPixelsGreen, resultPixelsGreen_not_naive, width, height, width, q);
        MedianFilterGPU::median_filter_3x3_not_naive(inputPixelsBlue, resultPixelsBlue_not_naive, width, height, width, q);
    }

    assert(compare_images(resultPixelsRed, resultPixelsRed_naive, width * height));
    assert(compare_images(resultPixelsGreen, resultPixelsGreen_naive, width * height));
    assert(compare_images(resultPixelsBlue, resultPixelsBlue_naive, width * height));
    
    assert(compare_images(resultPixelsRed, resultPixelsRed_not_naive, width * height));
    assert(compare_images(resultPixelsGreen, resultPixelsGreen_not_naive, width * height));
    assert(compare_images(resultPixelsBlue, resultPixelsBlue_not_naive, width * height));

    BMP resultBMP;
    create_BMP(inputBMP, resultBMP, resultPixelsRed_naive, resultPixelsGreen_naive, resultPixelsBlue_naive);
    resultBMP.WriteToFile("result.bmp");

    return 0;
}
