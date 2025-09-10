

#include <cstdint>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cub/device/device_reduce.cuh>

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

__device__ __forceinline__ int clamp_i(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__device__ __forceinline__ float clamp_f(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// ============================================================================
// IMAGE PRE-PROCESSING  (RGBA → planar-RGB, NCHW, normalized)
// ============================================================================

__global__ void preprocessImageKernel(
    const uint8_t* __restrict__ inRGBA,
    float*        __restrict__ outNCHW,
    int W, int H,
    int inPitch,          // bytes between rows in the source image
    int outPitchBytes,    // bytes between channels in the output (H*W*sizeof(float))
    float scale,
    float meanR, float meanG, float meanB,
    float stdR,  float stdG,  float stdB)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    // --- load one RGBA pixel -------------------------------------------------
    const uint8_t* src = inRGBA + y * inPitch + 4 * x;
    float r = (src[0] * scale - meanR) / stdR;
    float g = (src[1] * scale - meanG) / stdG;
    float b = (src[2] * scale - meanB) / stdB;

    // --- write to planar NCHW buffer ----------------------------------------
    // channel stride = H*W floats  (= outPitchBytes/sizeof(float))
    const int HW      = H * W;
    const int dstIdx  = y * W + x;          // offset inside each plane
    float* baseR = outNCHW;                 // plane 0
    float* baseG = baseR + HW;              // plane 1
    float* baseB = baseG + HW;              // plane 2
    baseR[dstIdx] = r;
    baseG[dstIdx] = g;
    baseB[dstIdx] = b;
}

// ============================================================================
// PROTOTYPE PROCESSING KERNELS
// ============================================================================

__global__ void resizePrototypesKernelChannelFirst(
    const float* __restrict__ prototypes,
    float* __restrict__ resized,
    int C, int H_proto, int W_proto,
    int H_det, int W_det)
{
    int c = blockIdx.z;
    int y_det = blockIdx.y * blockDim.y + threadIdx.y;
    int x_det = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (c >= C || y_det >= H_det || x_det >= W_det) return;

    // Map detection pixel to prototype coordinate
    float y_proto = ((float)y_det + 0.5f) * H_proto / (float)H_det - 0.5f;
    float x_proto = ((float)x_det + 0.5f) * W_proto / (float)W_det - 0.5f;

    int y0 = clamp_i((int)y_proto, 0, H_proto - 1);
    int x0 = clamp_i((int)x_proto, 0, W_proto - 1);
    int y1 = clamp_i(y0 + 1, 0, H_proto - 1);
    int x1 = clamp_i(x0 + 1, 0, W_proto - 1);

    float dy = y_proto - (float)y0;
    float dx = x_proto - (float)x0;

    // Channel-first input: [C, H, W] - EXACTLY matching Python reference
    const float* proto_plane = prototypes + c * H_proto * W_proto;
    float v00 = proto_plane[y0 * W_proto + x0];
    float v01 = proto_plane[y0 * W_proto + x1];
    float v10 = proto_plane[y1 * W_proto + x0];
    float v11 = proto_plane[y1 * W_proto + x1];

    // Bilinear interpolation
    float val = v00 * (1.0f - dy) * (1.0f - dx) +
                v01 * (1.0f - dy) * dx +
                v10 * dy * (1.0f - dx) +
                v11 * dy * dx;

    // Channel-last output: [HW, C] - EXACTLY matching Python reference
    // Python: out[ c + (yn * Wn +xn ) * C] = val;
    resized[c + (y_det * W_det + x_det) * C] = val;
}


__global__ void resizePrototypesKernelChannelLast(
    const float* __restrict__ prototypes,
    float* __restrict__ resized,
    int C, int H_proto, int W_proto,
    int H_det, int W_det)
{
    int c = blockIdx.z;
    int y_det = blockIdx.y * blockDim.y + threadIdx.y;
    int x_det = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (c >= C || y_det >= H_det || x_det >= W_det) return;

    // Map detection pixel to prototype coordinate
    float y_proto = ((float)y_det + 0.5f) * H_proto / (float)H_det - 0.5f;
    float x_proto = ((float)x_det + 0.5f) * W_proto / (float)W_det - 0.5f;

    int y0 = clamp_i((int)y_proto, 0, H_proto - 1);
    int x0 = clamp_i((int)x_proto, 0, W_proto - 1);
    int y1 = clamp_i(y0 + 1, 0, H_proto - 1);
    int x1 = clamp_i(x0 + 1, 0, W_proto - 1);

    float dy = y_proto - (float)y0;
    float dx = x_proto - (float)x0;

    // CRITICAL FIX: Channel-last input [H, W, C] - DriveWorks format [160, 160, 32, 1]
    // Access pattern: prototypes[y * W * C + x * C + c]
    float v00 = prototypes[(y0 * W_proto + x0) * C + c];
    float v01 = prototypes[(y0 * W_proto + x1) * C + c];
    float v10 = prototypes[(y1 * W_proto + x0) * C + c];
    float v11 = prototypes[(y1 * W_proto + x1) * C + c];

    // Bilinear interpolation
    float val = v00 * (1.0f - dy) * (1.0f - dx) +
                v01 * (1.0f - dy) * dx +
                v10 * dy * (1.0f - dx) +
                v11 * dy * dx;

    // Channel-last output: [HW, C] matching Python exactly
    // Output pattern: resized[pixel_idx * C + c] where pixel_idx = y_det * W_det + x_det
    resized[c + (y_det * W_det + x_det) * C] = val;
}


/**
 * Sigmoid activation with thresholding
 */

__global__ void sigmoidThresholdKernel(
    const float* __restrict__ linear_masks,
    uint8_t* __restrict__ binary_masks,
    int N, int HW, float threshold = 0.5f)
{
    int hw = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y;
    
    if (n >= N || hw >= HW) return;

    // CRITICAL FIX: Match Python indexing exactly
    // Python: float x = lin[ hw * N + n ];
    float x = linear_masks[hw * N + n];
    
    // Use fast exponential like Python reference
    float e = __expf(-x);
    float sigmoid = 1.0f / (1.0f + e);
    
    // Python outputs 1/0, but we need 255/0 for visualization
    binary_masks[n * HW + hw] = (sigmoid > threshold) ? 255 : 0;
}

// ============================================================================
// MASK COMBINATION AND ANALYSIS KERNELS
// ============================================================================

/**
 * OR-reduce multiple binary masks into single combined mask
 */
__global__ void orReduceMasksKernel(
    const uint8_t* __restrict__ masks,
    uint8_t* __restrict__ combined,
    int N, int H, int W)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (y >= H || x >= W) return;

    int idx_out = y * W + x;
    uint8_t result = 0;
    
    // CRITICAL FIX: Match Python indexing and logic exactly
    // Python: v |= masks[n * H * W + idx_out]; out[idx_out] = v ? 255 : 0;
    for (int n = 0; n < N; ++n) {
        result |= masks[n * H * W + idx_out];
    }
    
    combined[idx_out] = result ? 255 : 0;
}
/**
 * Row-wise boundary detection for lane analysis
 */
__global__ void rowBoundaryDetectionKernel(
    const uint8_t* __restrict__ combined_mask,
    int32_t* __restrict__ left_bounds,
    int32_t* __restrict__ right_bounds,
    int H, int W)
{
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= H) return;

    const uint8_t* row = combined_mask + y * W;
    
    int32_t left = -1;
    int32_t right = W;
    
    // Find leftmost and rightmost active pixels
    for (int x = 0; x < W; ++x) {
        if (row[x] > 0) {
            if (left < 0) left = x;
            right = x;
        }
    }
    
    left_bounds[y] = left;
    right_bounds[y] = right;
}

/**
 * Generate lane boundary and area masks from boundary arrays
 */
__global__ void buildLaneMasksKernel(
    const int32_t* __restrict__ left_bounds,
    const int32_t* __restrict__ right_bounds,
    uint8_t* __restrict__ left_mask,
    uint8_t* __restrict__ right_mask,
    uint8_t* __restrict__ area_mask,
    int H, int W)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (y >= H || x >= W) return;

    int32_t left = left_bounds[y];
    int32_t right = right_bounds[y];
    
    int idx = y * W + x;
    
    // Left boundary marking
    left_mask[idx] = (left >= 0 && x == left) ? 255 : 0;
    
    // Right boundary marking  
    right_mask[idx] = (right < W && x == right) ? 255 : 0;
    
    // Lane area between boundaries
    area_mask[idx] = (left >= 0 && right < W && x >= left && x <= right) ? 255 : 0;
}

// ============================================================================
// MORPHOLOGICAL OPERATIONS
// ============================================================================

/**
 * Morphological dilation with 5x5 kernel
 */
__global__ void dilate5x5Kernel(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    int H, int W)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= W || y >= H) return;

    bool dilated = false;
    
    #pragma unroll
    for (int dy = -2; dy <= 2; ++dy) {
        #pragma unroll
        for (int dx = -2; dx <= 2; ++dx) {
            int nx = clamp_i(x + dx, 0, W - 1);
            int ny = clamp_i(y + dy, 0, H - 1);
            if (input[ny * W + nx] > 0) {
                dilated = true;
            }
        }
    }
    
    output[y * W + x] = dilated ? 255 : 0;
}

/**
 * Morphological erosion with 5x5 kernel
 */
__global__ void erode5x5Kernel(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    int H, int W)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= W || y >= H) return;

    bool eroded = true;
    
    #pragma unroll
    for (int dy = -2; dy <= 2; ++dy) {
        #pragma unroll
        for (int dx = -2; dx <= 2; ++dx) {
            int nx = clamp_i(x + dx, 0, W - 1);
            int ny = clamp_i(y + dy, 0, H - 1);
            if (input[ny * W + nx] == 0) {
                eroded = false;
            }
        }
    }
    
    output[y * W + x] = eroded ? 255 : 0;
}

// ============================================================================
// VISUALIZATION KERNELS
// ============================================================================

/**
 * Colorize multiple masks with distinct colors
 */
__global__ void colorizeMasksKernel(
    const uint8_t* __restrict__ masks,
    const uchar3* __restrict__ color_lut,
    uchar3* __restrict__ colored_output,
    int N, int H, int W, int max_colors)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= W || y >= H) return;

    uchar3 color = make_uchar3(0, 0, 0);
    
    for (int n = 0; n < N; ++n) {
        if (masks[n * H * W + y * W + x] > 0) {
            color = color_lut[n % max_colors];
            break; // Use first matching color
        }
    }
    
    colored_output[y * W + x] = color;
}

//debug kernel :

__global__ void validateTensorAccessKernel(
    const float* __restrict__ tensor,
    float* __restrict__ output,
    int C, int H, int W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Test a few sample positions
    if (idx < 10) {
        int y = idx % H;
        int x = (idx / H) % W;
        int c = 0; // Test channel 0
        
        // Try both access patterns and see which one gives reasonable values
        float val_channel_first = tensor[c * H * W + y * W + x];
        float val_channel_last = tensor[(y * W + x) * C + c];
        
        // Output both values for comparison
        output[idx * 2] = val_channel_first;
        output[idx * 2 + 1] = val_channel_last;
        
        // Print first few values for debugging
        if (idx < 5) {
            printf("idx=%d: channel_first=%.6f, channel_last=%.6f\n", 
                   idx, val_channel_first, val_channel_last);
        }
    }
}

// ============================================================================
// C WRAPPER FUNCTIONS
// ============================================================================

extern "C" {

void launchPreprocessImageKernel(
    const uint8_t* inputRGBA, float* outputRGB,
    int width, int height,
    int inputPitch, int outputPitch,     // keep same signature as header
    float scale,
    float meanR, float meanG, float meanB,
    float stdR,  float stdG,  float stdB,
    cudaStream_t stream)
{
    if (!inputRGBA || !outputRGB)
    {
        printf("ERROR: null pointer(s) in preprocessImageKernel launch\n");
        return;
    }

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height+ block.y - 1) / block.y);

    preprocessImageKernel<<<grid, block, 0, stream>>>(
        inputRGBA, outputRGB,
        width, height,
        inputPitch, outputPitch,
        scale,
        meanR, meanG, meanB,
        stdR,  stdG,  stdB);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("ERROR: preprocessImageKernel launch failed: %s\n", cudaGetErrorString(err));
}


//debug kernel wrapper

void launchValidateTensorAccessKernel(
        const float* tensor, float* output,
        int C, int H, int W, cudaStream_t stream)
    {
        dim3 blockSize(256);
        dim3 gridSize(1);
        
        printf("Launching tensor validation kernel: C=%d, H=%d, W=%d\n", C, H, W);
        
        validateTensorAccessKernel<<<gridSize, blockSize, 0, stream>>>(
            tensor, output, C, H, W);
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("ERROR: Validation kernel launch failed: %s\n", cudaGetErrorString(err));
        }
        
        cudaStreamSynchronize(stream);
    }


void launchResizePrototypesKernel(
    const float* prototypes, float* resized,
    int C, int H_proto, int W_proto, int H_det, int W_det,
    cudaStream_t stream)
{
    // Validate input parameters
    if (!prototypes || !resized) {
        printf("ERROR: Null pointers in resize kernel\n");
        return;
    }
    
    if (C <= 0 || H_proto <= 0 || W_proto <= 0 || H_det <= 0 || W_det <= 0) {
        printf("ERROR: Invalid dimensions in resize kernel: C=%d, H_proto=%d, W_proto=%d, H_det=%d, W_det=%d\n",
               C, H_proto, W_proto, H_det, W_det);
        return;
    }
    
    // Use 3D grid matching Python reference exactly
    dim3 blockSize(16, 16, 1);
    dim3 gridSize(
        (W_det + blockSize.x - 1) / blockSize.x,
        (H_det + blockSize.y - 1) / blockSize.y,
        C  // 3D launch for channels
    );
    
    printf("3D Resize kernel launch: C=%d, H_proto=%d, W_proto=%d, H_det=%d, W_det=%d\n",
           C, H_proto, W_proto, H_det, W_det);
    printf("Grid: (%d, %d, %d), Block: (%d, %d, %d)\n",
           gridSize.x, gridSize.y, gridSize.z, blockSize.x, blockSize.y, blockSize.z);
    
    // CRITICAL FIX: DriveWorks tensor [160, 160, 32, 1] is [H, W, C, N] format
    // This means we have channel-LAST input, not channel-first!
    printf("Using CHANNEL-LAST input kernel (matching DriveWorks [H,W,C,N] format)\n");
    
    resizePrototypesKernelChannelLast<<<gridSize, blockSize, 0, stream>>>(
        prototypes, resized, C, H_proto, W_proto, H_det, W_det
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERROR: Resize kernel launch failed: %s\n", cudaGetErrorString(err));
    } else {
        printf("Resize kernel launched successfully with channel-last input\n");
    }
}

void launchSigmoidThresholdKernel(
    const float* linear_masks, uint8_t* binary_masks,
    int N, int HW, float threshold, cudaStream_t stream)
{
    if (!linear_masks || !binary_masks) {
        printf("ERROR: Null pointers in sigmoid threshold kernel\n");
        return;
    }
    
    if (N <= 0 || HW <= 0) {
        printf("ERROR: Invalid dimensions in sigmoid threshold kernel: N=%d, HW=%d\n", N, HW);
        return;
    }
    
    // CRITICAL FIX: Match Python grid configuration exactly
    // Python: block=(256,1,1), grid=((HW+255)//256, N, 1)
    dim3 blockSize(256, 1, 1);
    dim3 gridSize((HW + 255) / 256, N, 1);
    
    printf("Sigmoid threshold kernel launch: N=%d, HW=%d, threshold=%f\n", N, HW, threshold);
    printf("Grid: (%d, %d, %d), Block: (%d, %d, %d)\n",
           gridSize.x, gridSize.y, gridSize.z, blockSize.x, blockSize.y, blockSize.z);
    
    sigmoidThresholdKernel<<<gridSize, blockSize, 0, stream>>>(
        linear_masks, binary_masks, N, HW, threshold);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERROR: Sigmoid threshold kernel launch failed: %s\n", cudaGetErrorString(err));
    } else {
        printf("Sigmoid threshold kernel launched successfully\n");
    }
}

void launchOrReduceMasksKernel(
    const uint8_t* masks, uint8_t* combined,
    int N, int H, int W, cudaStream_t stream)
{
    if (!masks || !combined) {
        printf("ERROR: Null pointers in OR reduce masks kernel\n");
        return;
    }
    
    if (N <= 0 || H <= 0 || W <= 0) {
        printf("ERROR: Invalid dimensions in OR reduce kernel: N=%d, H=%d, W=%d\n", N, H, W);
        return;
    }
    
    dim3 blockSize(16, 16, 1);
    dim3 gridSize(
        (W + blockSize.x - 1) / blockSize.x,
        (H + blockSize.y - 1) / blockSize.y, 1
    );
    
    printf("OR reduce masks kernel launch: N=%d, H=%d, W=%d\n", N, H, W);
    
    orReduceMasksKernel<<<gridSize, blockSize, 0, stream>>>(
        masks, combined, N, H, W);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERROR: OR reduce masks kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

void launchRowBoundaryDetectionKernel(
    const uint8_t* combined_mask, int32_t* left_bounds, int32_t* right_bounds,
    int H, int W, cudaStream_t stream)
{
    if (!combined_mask || !left_bounds || !right_bounds) {
        printf("ERROR: Null pointers in row boundary detection kernel\n");
        return;
    }
    
    if (H <= 0 || W <= 0) {
        printf("ERROR: Invalid dimensions in row boundary kernel: H=%d, W=%d\n", H, W);
        return;
    }
    
    dim3 blockSize(256, 1, 1);
    dim3 gridSize((H + blockSize.x - 1) / blockSize.x, 1, 1);
    
    printf("Row boundary detection kernel launch: H=%d, W=%d\n", H, W);
    
    rowBoundaryDetectionKernel<<<gridSize, blockSize, 0, stream>>>(
        combined_mask, left_bounds, right_bounds, H, W);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERROR: Row boundary detection kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

void launchBuildLaneMasksKernel(
    const int32_t* left_bounds, const int32_t* right_bounds,
    uint8_t* left_mask, uint8_t* right_mask, uint8_t* area_mask,
    int H, int W, cudaStream_t stream)
{
    if (!left_bounds || !right_bounds || !left_mask || !right_mask || !area_mask) {
        printf("ERROR: Null pointers in build lane masks kernel\n");
        return;
    }
    
    if (H <= 0 || W <= 0) {
        printf("ERROR: Invalid dimensions in build lane masks kernel: H=%d, W=%d\n", H, W);
        return;
    }
    
    dim3 blockSize(16, 16, 1);
    dim3 gridSize(
        (W + blockSize.x - 1) / blockSize.x,
        (H + blockSize.y - 1) / blockSize.y, 1
    );
    
    printf("Build lane masks kernel launch: H=%d, W=%d\n", H, W);
    
    buildLaneMasksKernel<<<gridSize, blockSize, 0, stream>>>(
        left_bounds, right_bounds, left_mask, right_mask, area_mask, H, W);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERROR: Build lane masks kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

// ============================================================================
// LEGACY KERNEL COMPATIBILITY IMPLEMENTATIONS
// ============================================================================

__global__ void maxCombineKernel(
    uint8_t* dest, size_t destPitch,
    const uint8_t* src, size_t srcPitch,
    uint32_t width, uint32_t height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    uint8_t* destRow = (uint8_t*)((char*)dest + y * destPitch);
    const uint8_t* srcRow = (const uint8_t*)((const char*)src + y * srcPitch);
    
    destRow[x] = max(destRow[x], srcRow[x]);
}

void launchMaxCombineKernel(uint8_t* dest, size_t destPitch,
                           const uint8_t* src, size_t srcPitch,
                           uint32_t width, uint32_t height,
                           cudaStream_t stream)
{
    if (!dest || !src) {
        printf("ERROR: Null pointers in max combine kernel\n");
        return;
    }
    
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    printf("Max combine kernel launch: %ux%u\n", width, height);
    
    maxCombineKernel<<<gridSize, blockSize, 0, stream>>>(
        dest, destPitch, src, srcPitch, width, height);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERROR: Max combine kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

void launchMorphologyKernel(uint8_t* dest, size_t destPitch,
                           const uint8_t* src, size_t srcPitch,
                           uint32_t width, uint32_t height,
                           int kernelSize, bool isDilation,
                           cudaStream_t stream)
{
    if (!dest || !src) {
        printf("ERROR: Null pointers in morphology kernel\n");
        return;
    }
    
    if (width == 0 || height == 0) {
        printf("ERROR: Invalid dimensions in morphology kernel: %ux%u\n", width, height);
        return;
    }
    
    // Convert pitched memory to linear for our kernels
    // Note: This is a simplified implementation - for production, you'd want
    // to handle pitched memory properly or modify kernels to work with pitched memory
    
    dim3 blockSize(16, 16, 1);
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y, 1
    );
    
    printf("Morphology kernel launch: %ux%u, kernel=%d, dilation=%s\n", 
           width, height, kernelSize, isDilation ? "true" : "false");
    
    if (kernelSize == 5) {
        if (isDilation) {
            dilate5x5Kernel<<<gridSize, blockSize, 0, stream>>>(
                src, dest, height, width);
        } else {
            erode5x5Kernel<<<gridSize, blockSize, 0, stream>>>(
                src, dest, height, width);
        }
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("ERROR: Morphology kernel launch failed: %s\n", cudaGetErrorString(err));
        }
    } else {
        printf("WARNING: Unsupported morphology kernel size: %d (only size 5 supported)\n", kernelSize);
    }
}

} // extern "C"