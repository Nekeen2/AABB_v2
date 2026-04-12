#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <sycl/sycl.hpp>
#include "utils.h"



class MedianFilterGPU {
private:
    static float median_7(float arr[7]);
    static uint8_t median_9(uint8_t window[9]);

public:
    static void median_filter_3x3_naive(const uint8_t* input, uint8_t* result, size_t width, size_t height, size_t stride, sycl::queue& q);
    static void median_filter_3x3_not_naive(const uint8_t* input, uint8_t* result, size_t width, size_t height, size_t stride, sycl::queue& q);
};

float MedianFilterGPU::median_7(float arr[7]) {
    cond_swap(arr[0], arr[6]);
    cond_swap(arr[2], arr[3]);
    cond_swap(arr[4], arr[5]);

    cond_swap(arr[0], arr[2]);
    cond_swap(arr[1], arr[4]);
    cond_swap(arr[3], arr[6]);

    arr[1] = get_max(arr[0], arr[1]);
    cond_swap(arr[2], arr[5]);
    cond_swap(arr[3], arr[4]);

    arr[2] = get_max(arr[1], arr[2]);
    arr[4] = get_min(arr[4], arr[6]);

    arr[3] = get_max(arr[2], arr[3]);
    arr[4] = get_min(arr[4], arr[5]);

    arr[3] = get_min(arr[3], arr[4]);

    return arr[3];
}

uint8_t MedianFilterGPU::median_9(uint8_t window[9]) {
    cond_swap(window[0], window[3]);
    cond_swap(window[1], window[7]);
    cond_swap(window[2], window[5]);
    cond_swap(window[4], window[8]);

    cond_swap(window[0], window[7]);
    cond_swap(window[2], window[4]);
    cond_swap(window[3], window[8]);
    cond_swap(window[5], window[6]);

    window[2] = get_max(window[0], window[2]);
    cond_swap(window[1], window[3]);
    cond_swap(window[4], window[5]);
    window[7] = get_min(window[7], window[8]);

    window[4] = get_max(window[1], window[4]);
    window[3] = get_min(window[3], window[6]);
    window[5] = get_min(window[5], window[7]);

    cond_swap(window[2], window[4]);
    cond_swap(window[3], window[5]);

    window[3] = get_max(window[2], window[3]);
    window[4] = get_min(window[4], window[5]);

    window[4] = get_max(window[3], window[4]);

    return window[4];
}

void MedianFilterGPU::median_filter_3x3_naive(const uint8_t* input, uint8_t* result, size_t width, size_t height, size_t stride, sycl::queue& q) {
    uint8_t* d_input = sycl::malloc_shared<uint8_t>(height * stride, q);
    uint8_t* d_result = sycl::malloc_shared<uint8_t>(height * stride, q);

    q.memcpy(d_input, input, height * stride * sizeof(uint8_t)).wait();

    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>(height, width), [=](sycl::id<2> idx) {
            size_t y = idx[0];
            size_t x = idx[1];

            size_t y0 = (y > 0) ? y - 1 : 0;
            size_t y1 = y;
            size_t y2 = (y < height - 1) ? y + 1 : y;

            size_t x0 = (x > 0) ? x - 1 : 0;
            size_t x1 = x;
            size_t x2 = (x < width - 1) ? x + 1 : x;

            uint8_t window[9];

            window[0] = d_input[y0 * stride + x0];
            window[1] = d_input[y0 * stride + x1];
            window[2] = d_input[y0 * stride + x2];

            window[3] = d_input[y1 * stride + x0];
            window[4] = d_input[y1 * stride + x1];
            window[5] = d_input[y1 * stride + x2];

            window[6] = d_input[y2 * stride + x0];
            window[7] = d_input[y2 * stride + x1];
            window[8] = d_input[y2 * stride + x2];

            uint8_t median = median_9(window);

            d_result[y * stride + x] = median;
            });
        });
    q.wait();

    q.memcpy(result, d_result, height * stride * sizeof(uint8_t)).wait();

    sycl::free(d_input, q);
    sycl::free(d_result, q);
}

void MedianFilterGPU::median_filter_3x3_not_naive(const uint8_t* input, uint8_t* result, size_t width, size_t height, size_t stride, sycl::queue& q) {
    uint8_t* d_input = sycl::malloc_shared<uint8_t>(height * stride, q);
    uint8_t* d_result = sycl::malloc_shared<uint8_t>(height * stride, q);

    q.memcpy(d_input, input, height * stride * sizeof(uint8_t)).wait();

    const int BLOCK_SIZE = 16;
    const int SHARED_SIZE = BLOCK_SIZE + 2;

    size_t blocks_y = (height + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t blocks_x = (width + BLOCK_SIZE - 1) / BLOCK_SIZE;

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint8_t, 2> shared(sycl::range<2>(SHARED_SIZE, SHARED_SIZE), h);

        h.parallel_for(
            sycl::nd_range<2>(
                sycl::range<2>(blocks_y * BLOCK_SIZE, blocks_x * BLOCK_SIZE),
                sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE)
            ),
            [=](sycl::nd_item<2> item) {
                size_t global_y = item.get_global_id(0);
                size_t global_x = item.get_global_id(1);
                size_t local_y = item.get_local_id(0);
                size_t local_x = item.get_local_id(1);
                size_t block_y = item.get_group(0);
                size_t block_x = item.get_group(1);

                int block_start_y = block_y * BLOCK_SIZE;
                int block_start_x = block_x * BLOCK_SIZE;

                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        int src_y = block_start_y + local_y + dy;
                        int src_x = block_start_x + local_x + dx;
                        int dst_y = local_y + 1 + dy;
                        int dst_x = local_x + 1 + dx;

                        src_y = std::clamp(src_y, 0, (int)height - 1);
                        src_x = std::clamp(src_x, 0, (int)width - 1);

                        shared[dst_y][dst_x] = d_input[src_y * stride + src_x];
                    }
                }

                item.barrier();

                if (global_y < height && global_x < width) {
                    uint8_t window[9] = {
                        shared[local_y][local_x],     shared[local_y][local_x + 1],     shared[local_y][local_x + 2],
                        shared[local_y + 1][local_x], shared[local_y + 1][local_x + 1], shared[local_y + 1][local_x + 2],
                        shared[local_y + 2][local_x], shared[local_y + 2][local_x + 1], shared[local_y + 2][local_x + 2]
                    };
                    d_result[global_y * stride + global_x] = median_9(window);
                }
            }
        );
    });
    q.wait();

    q.memcpy(result, d_result, height * stride * sizeof(uint8_t)).wait();
    sycl::free(d_input, q);
    sycl::free(d_result, q);
}