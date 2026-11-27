/**
 * Per-camera Brown–Conrady frustum-clamp limits in undistorted normalised units: the pinhole definition, |p| up to
 * the image border plus 0.3 of the half extent, with the border undistorted. Launched by each fused EWA launcher,
 * so the forward and backward clamp at the same limits.
 */

#pragma once

#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "camera_ops.cuh"
#include "common.h"

// Samples per image edge; odd, so the corners and the edge midpoints are among them.
constexpr int EWA_LIMIT_EDGE_SAMPLES = 257;
constexpr int EWA_LIMIT_THREADS = 256;

/** True when Brown–Conrady is monotone on all of [0, r], r = |(x, y)|: the inner branch, not one past a fold. */
static __device__ __forceinline__ bool ewa_bc_inner_branch(float x, float y, const float* d)
{
    const float s_end = x * x + y * y;
    for (int k = 1; k <= 32; ++k) {
        const float s = s_end * (float)(k * k) / 1024.f;
        const float n = 1.f + s * (d[0] + s * (d[1] + s * d[4]));
        const float m = 1.f + s * (d[5] + s * (d[6] + s * d[7]));
        const float n_s = d[0] + s * (2.f * d[1] + s * 3.f * d[4]);
        const float m_s = d[5] + s * (2.f * d[6] + s * 3.f * d[7]);
        if (m <= 1e-8f || n * m + 2.f * s * (n_s * m - n * m_s) <= 0.f)
            return false;
    }
    return true;
}

/** lims[c] = [x_pos, x_neg, y_pos, y_neg]: the clamp keeps −x_neg ≤ p_x ≤ x_pos and likewise for y. */
static __global__ void ewa_lens_clamp_limits_kernel(
    const float* __restrict__ Ks,   // [C, 3, 3]
    const float* __restrict__ dist, // [C, 12]
    int image_width,
    int image_height,
    float* __restrict__ lims // [C, 4]
)
{
    const int c = blockIdx.x;
    const float* K = Ks + c * 9;
    const float fx = K[0], fy = K[4], cx = K[2], cy = K[5];
    const float* d = dist + c * CameraOps<PINHOLE_FAMILY>::n_slots;
    const float w = (float)image_width, h = (float)image_height;

    float ext[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    for (int i = threadIdx.x; i < 4 * EWA_LIMIT_EDGE_SAMPLES; i += blockDim.x) {
        const int edge = i / EWA_LIMIT_EDGE_SAMPLES;
        const float t = (float)(i % EWA_LIMIT_EDGE_SAMPLES) / (float)(EWA_LIMIT_EDGE_SAMPLES - 1);
        const float u = edge < 2 ? t * w : (edge == 2 ? 0.f : w);
        const float v = edge < 2 ? (edge == 0 ? 0.f : h) : t * h;
        const float un = (u - cx) / fx, vn = (v - cy) / fy;
        float x, y, z, rx, ry;
        // A border point past the lens's largest distorted radius has no inner-branch preimage; the solver then
        // returns an unconverged iterate, which the reprojection residual rejects.
        if (!CameraOps<PINHOLE_FAMILY>::unproject(un, vn, d, x, y, z) || !ewa_bc_inner_branch(x, y, d) ||
            !CameraOps<PINHOLE_FAMILY>::project(x, y, 1.f, 1.f, 1.f, 0.f, 0.f, d, rx, ry) ||
            fabsf(rx - un) + fabsf(ry - vn) > 1e-4f)
            continue; // such centres are culled at the fold, they need no limit
        ext[0] = fmaxf(ext[0], x);
        ext[1] = fmaxf(ext[1], -x);
        ext[2] = fmaxf(ext[2], y);
        ext[3] = fmaxf(ext[3], -y);
    }

    __shared__ float red[4][EWA_LIMIT_THREADS];
    for (int k = 0; k < 4; ++k)
        red[k][threadIdx.x] = ext[k];
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            for (int k = 0; k < 4; ++k)
                red[k][threadIdx.x] = fmaxf(red[k][threadIdx.x], red[k][threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        // No border point on the inner branch: every centre inside the fold is on screen, so nothing is clamped.
        float e[4];
        for (int k = 0; k < 4; ++k)
            e[k] = red[k][0] > -INFINITY ? red[k][0] : INFINITY;
        const float mx = 0.3f * 0.5f * (e[0] + e[1]), my = 0.3f * 0.5f * (e[2] + e[3]);
        lims[c * 4 + 0] = e[0] + mx;
        lims[c * 4 + 1] = e[1] + mx;
        lims[c * 4 + 2] = e[2] + my;
        lims[c * 4 + 3] = e[3] + my;
    }
}

/** [C, 4] clamp limits for a Brown–Conrady batch; dist is contiguous [C, 12]. */
static inline at::Tensor
ewa_lens_clamp_limits(const at::Tensor& Ks, const at::Tensor& dist, int image_width, int image_height)
{
    const int C = (int)Ks.size(0);
    at::Tensor lims = at::empty({C, 4}, Ks.options());
    const at::Tensor Ks_c = Ks.contiguous();
    ewa_lens_clamp_limits_kernel<<<C, EWA_LIMIT_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        Ks_c.data_ptr<float>(),
        dist.data_ptr<float>(),
        image_width,
        image_height,
        lims.data_ptr<float>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return lims;
}

/** A launcher's lens inputs: validated distortion and, for Brown–Conrady, its clamp limits; null pointers when empty.
 */
struct EwaLensArgs {
    bool has_lens;
    at::Tensor dist, lims;
    const float* dist_ptr;
    const float* lims_ptr;
};

static inline EwaLensArgs
ewa_lens_args(const at::Tensor& Ks, const at::Tensor& dist, int family, int image_width, int image_height)
{
    EwaLensArgs a{dist.numel() > 0, at::Tensor(), at::Tensor(), nullptr, nullptr};
    if (!a.has_lens)
        return a;
    const int n_slots =
        family == FISHEYE_FAMILY ? CameraOps<FISHEYE_FAMILY>::n_slots : CameraOps<PINHOLE_FAMILY>::n_slots;
    TORCH_CHECK(
        dist.is_cuda() && dist.scalar_type() == at::kFloat && dist.dim() == 2 && dist.size(0) == Ks.size(0) &&
            dist.size(1) == n_slots,
        "dist must be a float32 CUDA tensor [C, n_slots] for the projection family"
    );
    a.dist = dist.contiguous();
    a.dist_ptr = a.dist.data_ptr<float>();
    if (family == PINHOLE_FAMILY) {
        a.lims = ewa_lens_clamp_limits(Ks, a.dist, image_width, image_height);
        a.lims_ptr = a.lims.data_ptr<float>();
    }
    return a;
}
