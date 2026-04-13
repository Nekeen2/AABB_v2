#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstddef>
#include "utils.h"

class MedianFilter {
private:
    static float median_7(float arr[7]);
    static uint8_t median_9(uint8_t window[9]);
public:
    static void median_filter_7(const float* input, float* output, size_t length);
    static void median_filter_3x3(const uint8_t* input, uint8_t* output,
        size_t width, size_t height, size_t stride);
};

float MedianFilter::median_7(float arr[7]) {
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

void MedianFilter::median_filter_7(const float* input, float* output, size_t length) {
    float window[7];

    for (size_t i = 0; i < 3; ++i) {
        for (int j = -3; j <= 3; ++j) {
            int idx = i + j;
            if (idx < 0) window[j + 3] = input[0];
            else if (idx >= length) window[j + 3] = input[length - 1];
            else window[j + 3] = input[idx];
        }

        output[i] = median_7(window);
    }

    for (size_t i = 3; i < length - 3; ++i) {
        for (int j = -3; j <= 3; ++j) window[j + 3] = input[i + j];

        output[i] = median_7(window);
    }

    for (size_t i = length - 3; i < length; ++i) {
        for (int j = -3; j <= 3; ++j) {
            int idx = i + j;
            if (idx < 0) window[j + 3] = input[0];
            else if (idx >= length) window[j + 3] = input[length - 1];
            else window[j + 3] = input[idx];
        }

        output[i] = median_7(window);
    }
}

uint8_t MedianFilter::median_9(uint8_t window[9]) {
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

inline void MedianFilter::median_filter_3x3(const uint8_t* input, uint8_t* output,
    size_t width, size_t height, size_t stride) {
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {

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
                input[off00 + 0], input[off01 + 0], input[off02 + 0],
                input[off10 + 0], input[off11 + 0], input[off12 + 0],
                input[off20 + 0], input[off21 + 0], input[off22 + 0]
            };
            uint8_t winG[9] = {
                input[off00 + 1], input[off01 + 1], input[off02 + 1],
                input[off10 + 1], input[off11 + 1], input[off12 + 1],
                input[off20 + 1], input[off21 + 1], input[off22 + 1]
            };
            uint8_t winB[9] = {
                input[off00 + 2], input[off01 + 2], input[off02 + 2],
                input[off10 + 2], input[off11 + 2], input[off12 + 2],
                input[off20 + 2], input[off21 + 2], input[off22 + 2]
            };

            size_t outIdx = y * stride + x * 3;
            output[outIdx + 0] = median_9(winR);
            output[outIdx + 1] = median_9(winG);
            output[outIdx + 2] = median_9(winB);
        }
    }
}