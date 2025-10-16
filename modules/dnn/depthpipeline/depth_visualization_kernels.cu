// depth_visualization_kernels.cu
#include <cuda_runtime.h>
#include <cstdint>

namespace depth_pipeline {

// Color scheme enumeration (must match DepthRenderer::ColorScheme)
enum ColorScheme {
    GRAYSCALE = 0,
    JET_COLORMAP = 1,
    TURBO_COLORMAP = 2
};

__device__ inline float3 jetColormap(float t)
{
    float3 color;
    
    if (t < 0.25f) {
        color.x = 0.0f;
        color.y = t * 4.0f;
        color.z = 1.0f;
    } else if (t < 0.5f) {
        color.x = 0.0f;
        color.y = 1.0f;
        color.z = 1.0f - (t - 0.25f) * 4.0f;
    } else if (t < 0.75f) {
        color.x = (t - 0.5f) * 4.0f;
        color.y = 1.0f;
        color.z = 0.0f;
    } else {
        color.x = 1.0f;
        color.y = 1.0f - (t - 0.75f) * 4.0f;
        color.z = 0.0f;
    }
    
    return color;
}

__device__ inline float3 turboColormap(float t)
{
    // Turbo colormap approximation (simplified)
    const float r = fminf(fmaxf(2.0f * (t - 0.5f), 0.0f), 1.0f);
    const float g = fminf(1.0f - fabsf(2.0f * t - 1.0f), 1.0f);
    const float b = fminf(fmaxf(2.0f * (0.5f - t), 0.0f), 1.0f);
    
    return make_float3(r, g, b);
}

__global__ void depthColorizationKernel(
    const float* __restrict__ depthMap,
    uint8_t* __restrict__ outputRGBA,
    uint32_t width,
    uint32_t height,
    size_t pitch,
    float minDepth,
    float maxDepth,
    int colorScheme)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Read depth value
    uint32_t depthIdx = y * width + x;
    float depth = depthMap[depthIdx];
    
    // Normalize depth to [0, 1] and invert (closer = brighter)
    float normalized = (depth - minDepth) / (maxDepth - minDepth + 1e-6f);
    normalized = 1.0f - fminf(fmaxf(normalized, 0.0f), 1.0f);
    
    // Apply color mapping
    float3 color;
    switch (colorScheme) {
        case GRAYSCALE:
            color = make_float3(normalized, normalized, normalized);
            break;
        case JET_COLORMAP:
            color = jetColormap(normalized);
            break;
        case TURBO_COLORMAP:
            color = turboColormap(normalized);
            break;
        default:
            color = jetColormap(normalized);
            break;
    }
    
    // Write RGBA output
    uint8_t* pixel = outputRGBA + y * pitch + x * 4;
    pixel[0] = static_cast<uint8_t>(color.x * 255.0f);
    pixel[1] = static_cast<uint8_t>(color.y * 255.0f);
    pixel[2] = static_cast<uint8_t>(color.z * 255.0f);
    pixel[3] = 255; // Alpha
}

void launchDepthColorizationKernel(
    const float* depthMap,
    uint8_t* outputRGBA,
    uint32_t width,
    uint32_t height,
    size_t pitch,
    float minDepth,
    float maxDepth,
    int colorScheme,
    cudaStream_t stream)
{
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );
    
    depthColorizationKernel<<<gridSize, blockSize, 0, stream>>>(
        depthMap, outputRGBA, width, height, pitch,
        minDepth, maxDepth, colorScheme
    );
}

} // namespace depth_pipeline