// Native host primitives for the existing AXI-Lite projection register ABI.
#include <algorithm>
#include <chrono>
#include <cfloat>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <time.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace {
constexpr std::uint32_t busy = 1;
constexpr std::uint32_t done_or_error = 6;

#if defined(__ARM_NEON)
void pack_eight_rows(std::int8_t *out, const std::int8_t *in,
                     std::size_t depth, std::ptrdiff_t stride)
{
    for (std::size_t column = 0; column < depth; column += 16) {
        int8x16x4_t lower, upper;
        for (int row = 0; row < 4; ++row) {
            const auto *first = in + row * stride + column;
            const auto pair = vzipq_s8(vld1q_s8(first), vld1q_s8(first + 4 * stride));
            lower.val[row] = pair.val[0];
            upper.val[row] = pair.val[1];
        }
        vst4q_s8(out + 8 * column, lower);
        vst4q_s8(out + 8 * column + 64, upper);
    }
}
#endif

void mmio_barrier()
{
#if defined(__arm__) || defined(__aarch64__)
    __asm__ __volatile__("dmb sy" ::: "memory");
#elif defined(__GNUC__) || defined(__clang__)
    __sync_synchronize();
#endif
}
}

extern "C" {

// Return 1 for inputs or floating-point environments requiring the NumPy path.
int pp_projection_quantize(std::int8_t *output, float *scales, const float *source,
                           std::size_t rows, std::size_t depth)
{
#if defined(__FAST_MATH__)
    return 1;
#endif
    if (FLT_EVAL_METHOD != 0 || std::fegetround() != FE_TONEAREST) {
        return 1;
    }
    for (std::size_t row = 0; row < rows; ++row) {
        float maximum = 0.0f;
        for (std::size_t k = 0; k < depth; ++k) {
            if (!std::isfinite(source[k])) {
                return 1;
            }
            maximum = std::max(maximum, std::fabs(source[k]));
        }
        const float scale = std::max(maximum / 127.0f, 1e-30f);
        scales[row] = scale;
        for (std::size_t k = 0; k < depth; ++k) {
            const float ratio = source[k] / scale;
            const float magnitude = std::fabs(ratio);
            int rounded = static_cast<int>(magnitude);
            const float fraction = magnitude - static_cast<float>(rounded);
            if (fraction > 0.5f || (fraction == 0.5f && (rounded & 1))) {
                ++rounded;
            }
            rounded = std::min(rounded, 127);
            output[k] = static_cast<std::int8_t>(ratio < 0.0f ? -rounded : rounded);
        }
        source += depth;
        output += depth;
    }
    return 0;
}

void pp_projection_pack(std::int8_t *destination, const std::int8_t *source,
                        std::size_t rows, std::size_t depth, std::size_t lanes,
                        std::ptrdiff_t row_stride, std::ptrdiff_t col_stride)
{
    for (std::size_t tile = 0; tile < rows; tile += lanes) {
        const auto live = std::min(lanes, rows - tile);
#if defined(__ARM_NEON)
        if (lanes == 8 && live == 8 && col_stride == 1 && depth % 16 == 0) {
            pack_eight_rows(destination, source + static_cast<std::ptrdiff_t>(tile) * row_stride,
                            depth, row_stride);
            destination += depth * lanes;
            continue;
        }
#endif
        if (live < lanes) {
            std::memset(destination, 0, depth * lanes);
        }
        // Keep the output lines hot across all lanes of the 64-wide decoder tile.
        for (std::size_t block = 0; block < depth; block += 32) {
            const auto end = std::min(block + 32, depth);
            for (std::size_t lane = 0; lane < live; ++lane) {
                const auto *input = source + static_cast<std::ptrdiff_t>(tile + lane) * row_stride;
                for (std::size_t k = block; k < end; ++k) {
                    destination[k * lanes + lane] = input[static_cast<std::ptrdiff_t>(k) * col_stride];
                }
            }
        }
        destination += depth * lanes;
    }
}

void pp_projection_unpack(void *destination, const std::int32_t *source,
                          std::size_t batch, std::size_t rows,
                          std::size_t position_lanes, std::size_t output_lanes,
                          std::ptrdiff_t row_stride, std::ptrdiff_t col_stride)
{
    const auto tiles = (rows + output_lanes - 1) / output_lanes;
    auto *output = static_cast<char *>(destination);
    for (std::size_t pos = 0; pos < batch; ++pos) {
        auto *dest_row = output + static_cast<std::ptrdiff_t>(pos) * row_stride;
        const auto *tile = source + (pos / position_lanes * tiles) * 64 +
                           pos % position_lanes * output_lanes;
        for (std::size_t row = 0; row < rows; row += output_lanes, tile += 64) {
            auto live = std::min(output_lanes, rows - row);
            auto *dest = dest_row + static_cast<std::ptrdiff_t>(row) * col_stride;
            if (col_stride == sizeof(std::int32_t)) {
                std::memcpy(dest, tile, live * sizeof(std::int32_t));
            } else {
                for (std::size_t lane = 0; lane < live; ++lane) {
                    std::memcpy(dest + static_cast<std::ptrdiff_t>(lane) * col_stride,
                                tile + lane, sizeof(std::int32_t));
                }
            }
        }
    }
}

// Return 0 with a terminal status, 1 if already busy, or 2 on timeout.
int pp_projection_dispatch(volatile std::uint32_t *mmio, std::uint32_t mode,
                           std::uint32_t depth, std::uint32_t positions, std::uint32_t outputs,
                           std::uint32_t a, std::uint32_t w, std::uint32_t y,
                           std::uint64_t timeout_ns, std::uint32_t *status)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::nanoseconds(timeout_ns);
    mmio_barrier();
    if (mmio[0] & busy) {
        return 1;
    }
    mmio[1] = mode;
    mmio[2] = depth;
    mmio[3] = positions;
    mmio[4] = outputs;
    mmio[5] = a;
    mmio[6] = w;
    mmio[7] = y;
    mmio_barrier();
    mmio[0] = 1;
    mmio_barrier();

    const timespec pause = {0, 25000};
    while (true) {
        const auto current = mmio[0];
        if (!(current & busy) && (current & done_or_error)) {
            mmio_barrier();
            *status = current;
            return 0;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            return 2;
        }
        nanosleep(&pause, nullptr);
    }
}

}
