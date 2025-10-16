// depth_kernels.cu
#include <cuda_runtime.h>
#include <cstdint>

namespace depth_pipeline {

__global__ void minMaxReductionKernel(const float* input, uint32_t n,
                                     float* minResult, float* maxResult)
{
    __shared__ float sharedMin[256];
    __shared__ float sharedMax[256];
    
    uint32_t tid = threadIdx.x;
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float localMin = INFINITY;
    float localMax = -INFINITY;
    
    while (idx < n) {
        float val = input[idx];
        
        if (!isnan(val) && !isinf(val)) {
            localMin = fminf(localMin, val);
            localMax = fmaxf(localMax, val);
        }
        
        idx += blockDim.x * gridDim.x;
    }
    
    sharedMin[tid] = localMin;
    sharedMax[tid] = localMax;
    __syncthreads();
    
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedMin[tid] = fminf(sharedMin[tid], sharedMin[tid + s]);
            sharedMax[tid] = fmaxf(sharedMax[tid], sharedMax[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicMin((int*)minResult, __float_as_int(sharedMin[0]));
        atomicMax((int*)maxResult, __float_as_int(sharedMax[0]));
    }
}

void launchMinMaxReductionKernel(const float* input, uint32_t numElements,
                                           float* minResult, float* maxResult,
                                           cudaStream_t stream)
{
    float initMin = INFINITY;
    float initMax = -INFINITY;
    cudaMemcpyAsync(minResult, &initMin, sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(maxResult, &initMax, sizeof(float), cudaMemcpyHostToDevice, stream);
    
    const uint32_t blockSize = 256;
    const uint32_t numBlocks = (numElements + blockSize - 1) / blockSize;
    const uint32_t maxBlocks = 1024;
    
    minMaxReductionKernel<<<numBlocks < maxBlocks ? numBlocks : maxBlocks, blockSize, 0, stream>>>(
        input, numElements, minResult, maxResult
    );
}

} // namespace depth_pipeline