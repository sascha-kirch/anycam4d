/**
 * Fused selective Adam without bias correction, in place; float32, contiguous.
 *
 * One thread per element; primitive g_idx = p_idx / M. Inactive primitives return at once, keeping
 * value and moments bit-exact.
 *
 * Selective Adam pattern from gsplat's AdamCUDA.cu (after Taming-3DGS, as ported in NVIDIA 3DGRUT):
 * Copyright the Regents of the University of California, Nerfstudio Team and contributors; Apache License 2.0.
 */

#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>

using at::Tensor;

/**
 * Fused sparse Adam element kernel. Grid over n_elements = N*M, block 256.
 */
__global__ void stso_adam_update_kernel(
    float* __restrict__ param,            // [N, ...]
    const float* __restrict__ param_grad, // [N, ...]
    float* __restrict__ exp_avg,          // [N, ...]
    float* __restrict__ exp_avg_sq,       // [N, ...]
    const bool* __restrict__ visibility,  // [N]
    const float lr,
    const float b1,
    const float b2,
    const float eps,
    const uint32_t N,
    const uint32_t M
)
{
    const uint32_t p_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t g_idx = p_idx / M;

    if (g_idx >= N)
        return;
    if (!visibility[g_idx])
        return;

    const float g = param_grad[p_idx];

    const float m = b1 * exp_avg[p_idx] + (1.0f - b1) * g;
    const float v = b2 * exp_avg_sq[p_idx] + (1.0f - b2) * g * g;

    param[p_idx] += -lr * m / (sqrtf(v) + eps);
    exp_avg[p_idx] = m;
    exp_avg_sq[p_idx] = v;
}

/**
 * Host launcher: mutates param / exp_avg / exp_avg_sq in place on the current CUDA stream.
 */
void stso_adam_update(
    Tensor& param,            // [N, ...]
    const Tensor& param_grad, // [N, ...]
    Tensor& exp_avg,          // [N, ...]
    Tensor& exp_avg_sq,       // [N, ...]
    const Tensor& visibility, // [N] bool
    const double lr,
    const double b1,
    const double b2,
    const double eps
)
{
    // fp32-only, CUDA, contiguous: the p_idx/M row indexing assumes a contiguous [N, ...] layout.
    TORCH_CHECK(param.is_cuda(), "param must be a CUDA tensor");
    TORCH_CHECK(param.scalar_type() == at::kFloat, "param must be float32");
    TORCH_CHECK(param_grad.scalar_type() == at::kFloat, "param_grad must be float32");
    TORCH_CHECK(exp_avg.scalar_type() == at::kFloat, "exp_avg must be float32");
    TORCH_CHECK(exp_avg_sq.scalar_type() == at::kFloat, "exp_avg_sq must be float32");
    TORCH_CHECK(param.is_contiguous(), "param must be contiguous");
    TORCH_CHECK(param_grad.is_contiguous(), "param_grad must be contiguous");
    TORCH_CHECK(exp_avg.is_contiguous(), "exp_avg must be contiguous");
    TORCH_CHECK(exp_avg_sq.is_contiguous(), "exp_avg_sq must be contiguous");
    TORCH_CHECK(visibility.scalar_type() == at::kBool, "visibility must be bool");

    const uint32_t N = param.size(0);
    const uint32_t M = param.numel() / N;
    const uint32_t n_elements = N * M;

    // Nothing to do for an empty parameter (matches gsplat's selective_adam skip).
    if (n_elements == 0)
        return;

    const int threads = 256;
    const int blocks = (n_elements + threads - 1) / threads;
    auto stream = at::cuda::getCurrentCUDAStream();

    stso_adam_update_kernel<<<blocks, threads, 0, stream>>>(
        param.data_ptr<float>(),
        param_grad.data_ptr<float>(),
        exp_avg.data_ptr<float>(),
        exp_avg_sq.data_ptr<float>(),
        visibility.data_ptr<bool>(),
        (float)lr,
        (float)b1,
        (float)b2,
        (float)eps,
        N,
        M
    );
}
