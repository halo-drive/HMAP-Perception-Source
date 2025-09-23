import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import DynamicModule
import argparse
import time
import ctypes
from collections import deque
fps_hist = deque(maxlen=120)
log_tick = time.perf_counter()

# kernel compiling command ::
# nvcc -O3 -std=c++14 -lineinfo      -gencode arch=compute_86,code=sm_86      -ptx postprocess_kernels.cu -o postprocess_kernels.ptx
#
#
# -----------------------------------------------------------------------------
# Load cuBLAS via ctypes
# -----------------------------------------------------------------------------
_libcublas = ctypes.cdll.LoadLibrary("libcublas.so")
# cublasCreate_v2(handle*)
_libcublas.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
_libcublas.cublasCreate_v2.restype  = ctypes.c_int
# cublasSgemm_v2(handle, opA, opB,
#                m, n, k,
#                *alpha, A, lda, B, ldb, *beta, C, ldc)
_libcublas.cublasSgemm_v2.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_void_p, ctypes.c_int,
    ctypes.c_void_p, ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_void_p, ctypes.c_int
]
_libcublas.cublasSgemm_v2.restype = ctypes.c_int

#cuBLAS operation constants
CUBLAS_OP_N = 0
CUBLAS_OP_T = 1


# -----------------------------------------------------------------------------
# Load CUDA-accelerated postprocessing kernels
# -----------------------------------------------------------------------------
postproc_mod = cuda.module_from_file("postprocess_kernels.ptx")
resizeK  = postproc_mod.get_function("resizePrototypesKernel")
sigmK    = postproc_mod.get_function("sigmoidThresholdKernel")
laneK    = postproc_mod.get_function("laneFillKernel")
orReduceK = postproc_mod.get_function("orReduceMasks")
rowMinMaxK  = postproc_mod.get_function("rowMinMaxKernel")
buildLaneK  = postproc_mod.get_function("buildLaneMasksKernel")
colourizeK = postproc_mod.get_function("colourizeMasksKernel")


# --- LUT with visually distinct colours (BGR because we'll show in OpenCV)
_colour_lut = np.array([
    (  0,255,255),  # yellow
    (255,  0,255),  # magenta
    (  0,255,  0),  # green
    (255,255,  0),  # cyan
    (255,128,  0),  # orange
    (  0,128,255),  # amber-blue, … add more if you need
], dtype=np.uint8)

lut_dev = cuda.mem_alloc(_colour_lut.nbytes)
cuda.memcpy_htod(lut_dev, _colour_lut)
max_colours = _colour_lut.shape[0]


class TensorRTInference:
    def __init__(self, engine_path):
        self.logger  = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.cublas_handle = ctypes.c_void_p()
        status = _libcublas.cublasCreate_v2(ctypes.byref(self.cublas_handle))
        if status != 0:
        	raise RuntimeError(f"cublasCreate failed with {status}")
        # Load engine
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()

        # Allocate I/O
        self.inputs, self.outputs, self.bindings = [], [], []
        for i in range(self.engine.num_io_tensors):
            name  = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size  = trt.volume(shape)

            host_mem = cuda.pagelocked_empty(size, dtype)
            dev_mem  = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(dev_mem))

            io = {"name": name, "host": host_mem, "device": dev_mem}
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(io)
            else:
                self.outputs.append(io)

    def colourise_masks(self, masks_bin, lut_dev, alpha=0.4):
        N, H, W = masks_bin.shape

        if N == 0:                         # nothing to draw
            return np.zeros((H, W, 3), np.uint8), 0.0

        N_used  = min(N, max_colours)
        needed  = N_used * H * W           # bytes we’ll actually upload

        masks_dev = cuda.mem_alloc(needed)
        cuda.memcpy_htod(masks_dev,
                        masks_bin[:N_used].ravel())   # contiguous slice

        rgb_dev  = cuda.mem_alloc(H * W * 3)           # uchar3 buffer
        colourizeK(masks_dev, lut_dev, rgb_dev,
                np.int32(N_used), np.int32(H), np.int32(W),
                block=(16,16,1), grid=((W+15)//16, (H+15)//16, 1))

        rgb_host = np.empty((H * W, 3), np.uint8)
        cuda.memcpy_dtoh(rgb_host, rgb_dev)

        overlay  = rgb_host.reshape(H, W, 3)

        masks_dev.free(); rgb_dev.free()
        return overlay, alpha



    
    def preprocess_frame(self, frame, input_size=640):
        orig = frame.copy()
        h0, w0 = frame.shape[:2]
        img = cv2.resize(frame, (input_size, input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.transpose(img, (2,0,1))[None, ...]
        return img, orig, (w0, h0)

    def infer(self, img):
        cuda.memcpy_htod(self.inputs[0]["device"], img.ravel())
        for io in self.inputs:  self.context.set_tensor_address(io["name"], io["device"])
        for io in self.outputs: self.context.set_tensor_address(io["name"], io["device"])
        self.context.execute_async_v3(0)
        outs = []
        for out in self.outputs:
            cuda.memcpy_dtoh(out["host"], out["device"])
            outs.append(out["host"].copy())
        return outs

    def postprocess_masks_only(self, outputs, conf_threshold=0.80, input_size=640):
        # CPU: filter confidences & gather mask coeffs
        det = outputs[1].reshape(37, 8400)
        scores = det[4]
        keep = scores > conf_threshold
        
        if not np.any(keep):
            empty_masks = np.empty((0, input_size, input_size), np.uint8)
            empty_scores = np.empty(0, dtype=scores.dtype)
            return empty_masks, empty_scores
        
        
        mc = det[5:37, keep].astype(np.float32)
        N = mc.shape[1]

        # Push mask coeffs to GPU
        mc_dev = cuda.mem_alloc(mc.nbytes)
        cuda.memcpy_htod(mc_dev, mc)

        # Resize prototypes on GPU
        seg = outputs[0].astype(np.float32)
        proto_dev = cuda.mem_alloc(seg.nbytes)
        cuda.memcpy_htod(proto_dev, seg)
        C, H, W = 32, 160, 160
        Hn, Wn = input_size, input_size
        proto_res_dev = cuda.mem_alloc(C*Hn*Wn*4)
        block = (16,16,1)
        grid  = ((Wn+15)//16, (Hn+15)//16, C)
        resizeK(proto_dev, proto_res_dev,
                np.int32(C), np.int32(H), np.int32(W),
                np.int32(Hn), np.int32(Wn),
                block=block, grid=grid)

        # GEMM: mask coeffsᵀ [N×32] × proto_res [32×HW] → lin [N×HW]
        HW = Hn*Wn
        lin_dev = cuda.mem_alloc(N*HW*4)
        alpha = np.float32(1.0); beta = np.float32(0.0)
        alpha_ct = ctypes.c_float(1.0)
        beta_ct  = ctypes.c_float(0.0)
        mc_ptr   = ctypes.c_void_p(int(mc_dev))
        proto_ptr= ctypes.c_void_p(int(proto_res_dev))
        lin_ptr  = ctypes.c_void_p(int(lin_dev))
        status = _libcublas.cublasSgemm_v2(
            self.cublas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            N, HW, 32,
            ctypes.byref(alpha_ct),
            mc_ptr, 32,
            proto_ptr, 32,
            ctypes.byref(beta_ct),
            lin_ptr, N
        )
        if status != 0:
            raise RuntimeError(f"cublasSgemm_v2 failed with {status}")

        # Sigmoid + threshold
        bin_dev = cuda.mem_alloc(N*HW)
        block = (256,1,1)
        grid  = ((HW+255)//256, N, 1)
        sigmK(lin_dev, bin_dev,
              np.int32(N), np.int32(HW),
              block=block, grid=grid)

        masks = np.empty((N, HW), dtype=np.uint8)
        cuda.memcpy_dtoh(masks, bin_dev)
        masks = masks.reshape(N, input_size, input_size)

        # cleanup
        for d in (mc_dev, proto_dev, proto_res_dev, lin_dev, bin_dev): d.free()
        return masks, scores[keep]

    def analyze_lanes(self, masks, input_size=640):
        if masks.size == 0:                         # no detections
            return [], [], 0

        H = W = input_size
        N = masks.shape[0]

        # 1) OR-reduce N binary masks → one mask
        masks_dev = cuda.mem_alloc(masks.nbytes)
        cuda.memcpy_htod(masks_dev, masks)
        comb_dev  = cuda.mem_alloc(H * W)
        orReduceK(masks_dev, comb_dev,
                  np.int32(N), np.int32(H), np.int32(W),
                  block=(16,16,1),
                  grid=((W+15)//16, (H+15)//16, 1))

        # 2) for every row, get leftmost / rightmost ‘1’
        left_dev  = cuda.mem_alloc(H * 4)   # int32
        right_dev = cuda.mem_alloc(H * 4)
        rowMinMaxK(comb_dev, left_dev, right_dev,
                   np.int32(H), np.int32(W),
                   block=(256,1,1),
                   grid=((H+255)//256,1,1))

        # bring back the two small vectors
        leftX  = np.empty(H,  dtype=np.int32);  cuda.memcpy_dtoh(leftX,  left_dev)
        rightX = np.empty(H,  dtype=np.int32);  cuda.memcpy_dtoh(rightX, right_dev)

        if not np.any(rightX < W):        # nothing found
            for d in (masks_dev, comb_dev, left_dev, right_dev): d.free()
            return [], [], 0

        # 3) build three masks on GPU in one go
        leftB_dev  = cuda.mem_alloc(H * W)
        rightB_dev = cuda.mem_alloc(H * W)
        area_dev   = cuda.mem_alloc(H * W)
        buildLaneK(left_dev, right_dev,
                   leftB_dev, rightB_dev, area_dev,
                   np.int32(H), np.int32(W),
                   block=(16,16,1),
                   grid=((W+15)//16, (H+15)//16, 1))

        # copy the masks we need for drawing
        leftB  = np.empty((H,W), np.uint8); cuda.memcpy_dtoh(leftB,  leftB_dev)
        rightB = np.empty((H,W), np.uint8); cuda.memcpy_dtoh(rightB, rightB_dev)
        area   = np.empty((H,W), np.uint8); cuda.memcpy_dtoh(area,   area_dev)

        # free GPU buffers
        for d in (masks_dev, comb_dev, left_dev, right_dev,
                  leftB_dev, rightB_dev, area_dev): d.free()

        boundary_masks = [(leftB>0), (rightB>0)]
        lane_areas     = [(area>0)]
        return boundary_masks, lane_areas, 1
    
    def visualize_lanes_and_areas(self, original_frame, boundary_masks, lane_areas, num_lanes, scores, input_size=640, alpha=0.6):
        """Create visualization with lane markings and lane areas"""
        
        # Resize original frame to input size for consistent visualization
        vis_frame = cv2.resize(original_frame, (input_size, input_size))
        
        # Create overlays
        marking_overlay = np.zeros_like(vis_frame, dtype=np.uint8)
        area_overlay = np.zeros_like(vis_frame, dtype=np.uint8)
        
        # Colors for lane markings (bright colors)
        marking_colors = [
            (0, 255, 255),    # Yellow (left marking)
            (255, 0, 255),    # Magenta (right marking)
            (0, 255, 0),      # Green
            (255, 255, 0),    # Cyan
        ]
        
        # Colors for lane areas (semi-transparent)
        area_colors = [
            (0, 255, 0),      # Green for drivable lane area
            (0, 200, 255),    # Orange for additional lanes
            (255, 200, 0),    # Light blue
        ]
        
        # Draw lane markings (boundaries)
        for i, mask in enumerate(boundary_masks):
            color = marking_colors[i % len(marking_colors)]
            marking_overlay[mask > 0] = color
        
        # Draw lane areas (drivable areas)
        for i, mask in enumerate(lane_areas):
            color = area_colors[i % len(area_colors)]
            area_overlay[mask > 0] = color
        
        # Combine overlays with original frame
        result_frame = vis_frame.copy()
        
        # Apply lane areas first (lower opacity)
        if len(lane_areas) > 0:
            result_frame = cv2.addWeighted(result_frame, 1.0, area_overlay, alpha * 0.4, 0)
        
        # Apply lane markings on top (higher opacity)
        if len(boundary_masks) > 0:
            result_frame = cv2.addWeighted(result_frame, 1.0, marking_overlay, alpha, 0)
        
        # Add text information
        lane_text = f"Lanes: {num_lanes}" if num_lanes > 0 else "No lanes detected"
        cv2.putText(result_frame, lane_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        marking_text = f"Lane markings: {len(boundary_masks)}"
        cv2.putText(result_frame, marking_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if len(scores) > 0:
            conf_text = f"Avg confidence: {np.mean(scores):.3f}"
            cv2.putText(result_frame, conf_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add legend
        legend_y = input_size - 80
        cv2.putText(result_frame, "Legend:", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.rectangle(result_frame, (10, legend_y + 10), (30, legend_y + 25), (0, 255, 255), -1)  # Yellow
        cv2.putText(result_frame, "Lane markings", (35, legend_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(result_frame, (10, legend_y + 30), (30, legend_y + 45), (0, 255, 0), -1)  # Green
        cv2.putText(result_frame, "Drivable lane", (35, legend_y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result_frame

# Camera setup function (unchanged)
def setup_camera(source):
    is_video_file = False
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    elif source.startswith(('http://', 'https://', 'rtsp://')):
        cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source)
        is_video_file = True
    if not cap.isOpened(): raise ValueError(f"Cannot open source: {source}")
    return cap, is_video_file, None

# Main loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', required=True)
    parser.add_argument('--source', default='0')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--display-size', type=int, default=640)
    args = parser.parse_args()

    trt_inf = TensorRTInference(args.engine)
    cap, is_video, _ = setup_camera(args.source)

    while True:
        loop_start = time.perf_counter()
        ret, frame = cap.read()
        
        if not ret: break
        img, orig, _ = trt_inf.preprocess_frame(frame, args.display_size)
        
        t0 = time.perf_counter()
        outputs = trt_inf.infer(img)
        t1 = time.perf_counter()
        masks, scores = trt_inf.postprocess_masks_only(outputs, args.conf, args.display_size)
        t2 = time.perf_counter()
        boundary_masks, lane_areas, num_lanes = trt_inf.analyze_lanes(masks, args.display_size)
        t3 = time.perf_counter()
        
        colour_overlay, col_alpha = trt_inf.colourise_masks(masks, lut_dev)
        # resize overlay back to the display size just like the frame
        colour_overlay = cv2.resize(colour_overlay, (args.display_size, args.display_size),
                                    interpolation=cv2.INTER_NEAREST)

        vis = orig.copy()
        vis = cv2.resize(vis, (args.display_size, args.display_size))
        cv2.addWeighted(colour_overlay, col_alpha, vis, 1.0, 0, vis)

        # still draw lane outlines & legend on top
        vis = trt_inf.visualize_lanes_and_areas(
                vis, boundary_masks, lane_areas, num_lanes,
                scores, args.display_size)

        cv2.imshow('Result', vis)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        frame_time = time.perf_counter() - loop_start
        fps_hist.append(frame_time)
        if (time.perf_counter() - log_tick) >= 1.0:
            avg_fps = len(fps_hist) / sum(fps_hist)
            inf_ms  = (t1 - t0)*1e3
            post_ms = (t2 - t1)*1e3
            seg_ms  = (t3 - t2)*1e3
            print(f"[{avg_fps:5.1f} FPS] infer {inf_ms:5.1f} ms | "
                  f"mask {post_ms:5.1f} ms | seg {seg_ms:5.1f} ms | "
                  f"lanes {num_lanes}")
            fps_hist.clear()
            log_tick = time.perf_counter()
    cap.release()
    cv2.destroyAllWindows()

