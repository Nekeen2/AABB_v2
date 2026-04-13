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
    static void median_filter_3x3_naive(const uint8_t* input, uint8_t* result,
        size_t width, size_t height, size_t stride,
        sycl::queue& q);
    static void median_filter_3x3_not_naive(const uint8_t* input, uint8_t* result,
        size_t width, size_t height, size_t stride,
        sycl::queue& q);
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

inline void MedianFilterGPU::median_filter_3x3_naive(const uint8_t* input, uint8_t* result,
    size_t width, size_t height, size_t stride, sycl::queue& q) {
    size_t totalBytes = height * stride;
    uint8_t* d_input = sycl::malloc_shared<uint8_t>(totalBytes, q);
    uint8_t* d_result = sycl::malloc_shared<uint8_t>(totalBytes, q);
    q.memcpy(d_input, input, totalBytes).wait();

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

            size_t off00 = y0 * stride + x0 * 3;
            size_t off01 = y0 * stride + x1 * 3;
            size_t off02 = y0 * stride + x2 * 3;
            size_t off10 = y1 * stride + x0 * 3;
            size_t off11 = y1 * stride + x1 * 3;
            size_t off12 = y1 * stride + x2 * 3;
            size_t off20 = y2 * stride + x0 * 3;
            size_t off21 = y2 * stride + x1 * 3;
            size_t off22 = y2 * stride + x2 * 3;

            uint8_t winR[9] = {
                d_input[off00], d_input[off01], d_input[off02],
                d_input[off10], d_input[off11], d_input[off12],
                d_input[off20], d_input[off21], d_input[off22]
            };
            uint8_t winG[9] = {
                d_input[off00 + 1], d_input[off01 + 1], d_input[off02 + 1],
                d_input[off10 + 1], d_input[off11 + 1], d_input[off12 + 1],
                d_input[off20 + 1], d_input[off21 + 1], d_input[off22 + 1]
            };
            uint8_t winB[9] = {
                d_input[off00 + 2], d_input[off01 + 2], d_input[off02 + 2],
                d_input[off10 + 2], d_input[off11 + 2], d_input[off12 + 2],
                d_input[off20 + 2], d_input[off21 + 2], d_input[off22 + 2]
            };

            size_t outOff = y * stride + x * 3;
            d_result[outOff] = median_9(winR);
            d_result[outOff + 1] = median_9(winG);
            d_result[outOff + 2] = median_9(winB);
            });
        });
    q.wait();

    q.memcpy(result, d_result, totalBytes).wait();
    sycl::free(d_input, q);
    sycl::free(d_result, q);
}

inline void MedianFilterGPU::median_filter_3x3_not_naive(const uint8_t* input, uint8_t* result,
    size_t width, size_t height, size_t stride, sycl::queue& q) {

    size_t totalBytes = height * stride;
    uint8_t* d_input = sycl::malloc_shared<uint8_t>(totalBytes, q);
    uint8_t* d_result = sycl::malloc_shared<uint8_t>(totalBytes, q);
    q.memcpy(d_input, input, totalBytes).wait();

    const int BLOCK_SIZE = 16;
    const int SHARED_SIZE = BLOCK_SIZE + 2;

    size_t blocks_y = (height + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t blocks_x = (width + BLOCK_SIZE - 1) / BLOCK_SIZE;

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint8_t, 2> sharedR(sycl::range<2>(SHARED_SIZE, SHARED_SIZE), h);
        sycl::local_accessor<uint8_t, 2> sharedG(sycl::range<2>(SHARED_SIZE, SHARED_SIZE), h);
        sycl::local_accessor<uint8_t, 2> sharedB(sycl::range<2>(SHARED_SIZE, SHARED_SIZE), h);

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

                int src_y = block_start_y + local_y;
                int src_x = block_start_x + local_x;

                src_y = sycl::clamp(src_y, 0, (int)height - 1);
                src_x = sycl::clamp(src_x, 0, (int)width - 1);
                size_t srcIdx = src_y * stride + src_x * 3;

                sharedR[local_y + 1][local_x + 1] = d_input[srcIdx + 0];
                sharedG[local_y + 1][local_x + 1] = d_input[srcIdx + 1];
                sharedB[local_y + 1][local_x + 1] = d_input[srcIdx + 2];

                if (local_y == 0) {
                    int top_y = block_start_y - 1;
                    top_y = sycl::clamp(top_y, 0, (int)height - 1);
                    size_t topIdx = top_y * stride + src_x * 3;
                    sharedR[0][local_x + 1] = d_input[topIdx + 0];
                    sharedG[0][local_x + 1] = d_input[topIdx + 1];
                    sharedB[0][local_x + 1] = d_input[topIdx + 2];
                }

                if (local_y == BLOCK_SIZE - 1) {
                    int bottom_y = block_start_y + BLOCK_SIZE;
                    bottom_y = sycl::clamp(bottom_y, 0, (int)height - 1);
                    size_t bottomIdx = bottom_y * stride + src_x * 3;
                    sharedR[SHARED_SIZE - 1][local_x + 1] = d_input[bottomIdx + 0];
                    sharedG[SHARED_SIZE - 1][local_x + 1] = d_input[bottomIdx + 1];
                    sharedB[SHARED_SIZE - 1][local_x + 1] = d_input[bottomIdx + 2];
                }

                if (local_x == 0) {
                    int left_x = block_start_x - 1;
                    left_x = sycl::clamp(left_x, 0, (int)width - 1);
                    size_t leftIdx = src_y * stride + left_x * 3;
                    sharedR[local_y + 1][0] = d_input[leftIdx + 0];
                    sharedG[local_y + 1][0] = d_input[leftIdx + 1];
                    sharedB[local_y + 1][0] = d_input[leftIdx + 2];
                }

                if (local_x == BLOCK_SIZE - 1) {
                    int right_x = block_start_x + BLOCK_SIZE;
                    right_x = sycl::clamp(right_x, 0, (int)width - 1);
                    size_t rightIdx = src_y * stride + right_x * 3;
                    sharedR[local_y + 1][SHARED_SIZE - 1] = d_input[rightIdx + 0];
                    sharedG[local_y + 1][SHARED_SIZE - 1] = d_input[rightIdx + 1];
                    sharedB[local_y + 1][SHARED_SIZE - 1] = d_input[rightIdx + 2];
                }

                if (local_y == 0 && local_x == 0) {
                    int corner_y = sycl::clamp(block_start_y - 1, 0, (int)height - 1);
                    int corner_x = sycl::clamp(block_start_x - 1, 0, (int)width - 1);
                    size_t cornerIdx = corner_y * stride + corner_x * 3;
                    sharedR[0][0] = d_input[cornerIdx + 0];
                    sharedG[0][0] = d_input[cornerIdx + 1];
                    sharedB[0][0] = d_input[cornerIdx + 2];
                }
                if (local_y == 0 && local_x == BLOCK_SIZE - 1) {
                    int corner_y = sycl::clamp(block_start_y - 1, 0, (int)height - 1);
                    int corner_x = sycl::clamp(block_start_x + BLOCK_SIZE, 0, (int)width - 1);
                    size_t cornerIdx = corner_y * stride + corner_x * 3;
                    sharedR[0][SHARED_SIZE - 1] = d_input[cornerIdx + 0];
                    sharedG[0][SHARED_SIZE - 1] = d_input[cornerIdx + 1];
                    sharedB[0][SHARED_SIZE - 1] = d_input[cornerIdx + 2];
                }
                if (local_y == BLOCK_SIZE - 1 && local_x == 0) {
                    int corner_y = sycl::clamp(block_start_y + BLOCK_SIZE, 0, (int)height - 1);
                    int corner_x = sycl::clamp(block_start_x - 1, 0, (int)width - 1);
                    size_t cornerIdx = corner_y * stride + corner_x * 3;
                    sharedR[SHARED_SIZE - 1][0] = d_input[cornerIdx + 0];
                    sharedG[SHARED_SIZE - 1][0] = d_input[cornerIdx + 1];
                    sharedB[SHARED_SIZE - 1][0] = d_input[cornerIdx + 2];
                }
                if (local_y == BLOCK_SIZE - 1 && local_x == BLOCK_SIZE - 1) {
                    int corner_y = sycl::clamp(block_start_y + BLOCK_SIZE, 0, (int)height - 1);
                    int corner_x = sycl::clamp(block_start_x + BLOCK_SIZE, 0, (int)width - 1);
                    size_t cornerIdx = corner_y * stride + corner_x * 3;
                    sharedR[SHARED_SIZE - 1][SHARED_SIZE - 1] = d_input[cornerIdx + 0];
                    sharedG[SHARED_SIZE - 1][SHARED_SIZE - 1] = d_input[cornerIdx + 1];
                    sharedB[SHARED_SIZE - 1][SHARED_SIZE - 1] = d_input[cornerIdx + 2];
                }

                item.barrier();

                if (global_y < height && global_x < width) {
                    uint8_t winR[9] = {
                        sharedR[local_y][local_x],     sharedR[local_y][local_x + 1],     sharedR[local_y][local_x + 2],
                        sharedR[local_y + 1][local_x], sharedR[local_y + 1][local_x + 1], sharedR[local_y + 1][local_x + 2],
                        sharedR[local_y + 2][local_x], sharedR[local_y + 2][local_x + 1], sharedR[local_y + 2][local_x + 2]
                    };
                    uint8_t winG[9] = {
                        sharedG[local_y][local_x],     sharedG[local_y][local_x + 1],     sharedG[local_y][local_x + 2],
                        sharedG[local_y + 1][local_x], sharedG[local_y + 1][local_x + 1], sharedG[local_y + 1][local_x + 2],
                        sharedG[local_y + 2][local_x], sharedG[local_y + 2][local_x + 1], sharedG[local_y + 2][local_x + 2]
                    };
                    uint8_t winB[9] = {
                        sharedB[local_y][local_x],     sharedB[local_y][local_x + 1],     sharedB[local_y][local_x + 2],
                        sharedB[local_y + 1][local_x], sharedB[local_y + 1][local_x + 1], sharedB[local_y + 1][local_x + 2],
                        sharedB[local_y + 2][local_x], sharedB[local_y + 2][local_x + 1], sharedB[local_y + 2][local_x + 2]
                    };

                    size_t outIdx = global_y * stride + global_x * 3;
                    d_result[outIdx + 0] = median_9(winR);
                    d_result[outIdx + 1] = median_9(winG);
                    d_result[outIdx + 2] = median_9(winB);
                }
            }
        );
        });
    q.wait();

    q.memcpy(result, d_result, totalBytes).wait();
    sycl::free(d_input, q);
    sycl::free(d_result, q);
}