// postprocess_kernels.cu

// Custom CUDA kernels for post-processing in TensorRT pipeline.
// No external includes to avoid conflicting CUDA headers.

// Clamp integer between min and max
__device__ __forceinline__ int clamp_i(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// --------------------------------------------------------
// Bilinear resize of prototype feature maps
// Input:  in  [C, H, W]
// Output: out [C, Hn, Wn]
extern "C" __global__ void resizePrototypesKernel(
    const float* __restrict__ in,
    float* __restrict__ out,
    int C, int H, int W,
    int Hn, int Wn)
{
    int c  = blockIdx.z;
    int yn = blockIdx.y * blockDim.y + threadIdx.y;
    int xn = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C || yn >= Hn || xn >= Wn) return;

    // Map output pixel to input coordinate
    float fy = ( (float)yn + 0.5f ) * H / (float)Hn - 0.5f;
    float fx = ( (float)xn + 0.5f ) * W / (float)Wn - 0.5f;

    int y0 = clamp_i((int)fy, 0, H-1);
    int x0 = clamp_i((int)fx, 0, W-1);
    int y1 = clamp_i(y0 + 1, 0, H-1);
    int x1 = clamp_i(x0 + 1, 0, W-1);

    float dy = fy - (float)y0;
    float dx = fx - (float)x0;

    const float* in_plane = in + c * H * W;
    float v00 = in_plane[y0 * W + x0];
    float v01 = in_plane[y0 * W + x1];
    float v10 = in_plane[y1 * W + x0];
    float v11 = in_plane[y1 * W + x1];

    // bilinear interpolation
    float val = v00 * (1.0f - dy) * (1.0f - dx)
              + v01 * (1.0f - dy) * dx
              + v10 * dy * (1.0f - dx)
              + v11 * dy * dx;

    out[ c + (yn * Wn +xn ) * C] = val;
}

// --------------------------------------------------------
// Sigmoid activation + threshold
// lin: [N, HW]
// bin: [N, HW] as 0 or 1
extern "C" __global__ void sigmoidThresholdKernel(
    const float* __restrict__ lin,
    unsigned char* __restrict__ bin,
    int N, int HW)
{
    int hw = blockIdx.x * blockDim.x + threadIdx.x;
    int n  = blockIdx.y;
    if (n >= N || hw >= HW) return;

    float x = lin[ hw * N + n ];
    // use fast exp
    float e = __expf(-x);
    float s = 1.0f / (1.0f + e);
    bin[ n * HW + hw ] = (s > 0.5f) ? 1 : 0;
}

// --------------------------------------------------------
// Fill lane area between left and right boundary masks
// Inputs:
//   leftB:  [H, W] binary mask (0 or 255)
//   rightB: [H, W] binary mask (0 or 255)
// Output:
//   area:   [H, W] binary mask (0 or 255)
extern "C" __global__ void laneFillKernel(
    const unsigned char* __restrict__ leftB,
    const unsigned char* __restrict__ rightB,
    unsigned char* __restrict__ area,
    int H, int W)
{
    int y = blockIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= H || x >= W) return;

    // compute row boundaries once per row
    __shared__ int leftMost;
    __shared__ int rightMost;
    if (threadIdx.x == 0) {
        int lm = -1;
        for (int i = 0; i < W; ++i) {
            if (leftB[ y * W + i ] > 0) lm = i;
        }
        int rm = W;
        for (int i = 0; i < W; ++i) {
            if (rightB[ y * W + i ] > 0) { rm = i; break; }
        }
        leftMost  = lm;
        rightMost = rm;
    }
    __syncthreads();

    unsigned char v = 0;
    if ( leftMost >= 0 && rightMost < W && x >= leftMost && x <= rightMost ) {
        v = 255;
    }
    area[ y * W + x ] = v;
}

//  ---------------------------------------------------------------------
//  Bitwise OR reduction of N binary masks (0/1) into a single [H,W] mask
//  Inputs : masks [N, H, W]   (unsigned char, 0/1)
//  Output : out   [H, W]      (unsigned char, 0/255)
//  ---------------------------------------------------------------------
extern "C" __global__ void orReduceMasks(
    const unsigned char* __restrict__ masks,
    unsigned char*       __restrict__ out,
    int N, int H, int W)
{
    int y  = blockIdx.y * blockDim.y + threadIdx.y;
    int x  = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= H || x >= W) return;

    int idx_out = y * W + x;
    unsigned char v = 0;
    for (int n = 0; n < N; ++n)
        v |= masks[n * H * W + idx_out];

    out[idx_out] = v ? 255 : 0;
}


// ---------------------------------------------------------------------
// For each row find leftmost / rightmost pixel that is 1.
//  inMask  : [H, W]  (uchar; 0/1)
//  leftX   : [H]     (int32)
//  rightX  : [H]     (int32,  = W if none found)
// ---------------------------------------------------------------------
extern "C" __global__ void rowMinMaxKernel(
        const unsigned char* __restrict__ inMask,
        int*  __restrict__ leftX,
        int*  __restrict__ rightX,
        int   H, int W)
{
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= H) return;

    const unsigned char* row = inMask + y * W;

    int l = -1;
    int r = W;              // sentinel

    // unroll by 4 for speed (optional)
    for (int x = 0; x < W; ++x) {
        unsigned char v = row[x];
        if (v && l < 0) l = x;
        if (v)          r = x;
    }
    leftX [y] = l;
    rightX[y] = r;
}

// ---------------------------------------------------------------------
// Build three masks in one pass
//  leftMask, rightMask, laneArea  ←  leftX / rightX
//  Every thread writes one pixel; only global memory writes, no atomics
// ---------------------------------------------------------------------
extern "C" __global__ void buildLaneMasksKernel(
        const int*  __restrict__ leftX,
        const int*  __restrict__ rightX,
        unsigned char* __restrict__ leftMask,
        unsigned char* __restrict__ rightMask,
        unsigned char* __restrict__ area,
        int H, int W)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= H || x >= W) return;

    int l = leftX[y];
    int r = rightX[y];

    unsigned char vL = 0, vR = 0, vA = 0;
    if (l >= 0      && x == l) vL = 255;
    if (r <  W      && x == r) vR = 255;
    if (l >= 0 && r < W && x >= l && x <= r) vA = 255;

    int idx = y * W + x;
    leftMask [idx] = vL;
    rightMask[idx] = vR;
    area     [idx] = vA;
}

// --------------------------------------------------------------------
// colourizeMasksKernel
// Input : masks  [N,H,W]  (0/1)
// Output: rgbOut [H,W,3]  (uint8 0-255)
// Each mask[i] gets a colour from a lookup table LUT[i]
// --------------------------------------------------------------------
extern "C" __global__ void colourizeMasksKernel(
        const unsigned char* __restrict__ masks,
        const uchar3*       __restrict__ lut,   // N x (r,g,b)
        uchar3*             __restrict__ rgbOut,
        int N, int H, int W)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    uchar3 c = {0,0,0};
    for (int n = 0; n < N; ++n)
    {
        if (masks[n*H*W + y*W + x])
        {
            c = lut[n];             
        }
    }
    rgbOut[y*W + x] = c;
}


extern "C" __global__ void dilate5x5(
        const unsigned char* in,
        unsigned char* out,
        int H,int W)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x>=W||y>=H) return;

    bool set = false;
    #pragma unroll
    for (int dy=-2; dy<=2; ++dy)
        #pragma unroll
        for (int dx=-2; dx<=2; ++dx)
            if (in[ clamp_i(y+dy,0,H-1)*W +
                    clamp_i(x+dx,0,W-1) ])
                set = true;
    out[y*W+x] = set?255:0;
}

extern "C" __global__ void erode5x5(
        const unsigned char* in,
        unsigned char* out,
        int H,int W)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x>=W||y>=H) return;

    bool set = true;
    #pragma unroll
    for (int dy=-2; dy<=2; ++dy)
        #pragma unroll
        for (int dx=-2; dx<=2; ++dx)
            if (!in[ clamp_i(y+dy,0,H-1)*W +
                     clamp_i(x+dx,0,W-1) ])
                set = false;
    out[y*W+x] = set?255:0;
}
