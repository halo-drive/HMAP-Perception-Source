#!/usr/bin/env python3
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import argparse
import time
import ctypes
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Tuple
import logging
import json
import os
from filterpy.kalman import KalmanFilter

# ========================= Logging =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dual_engine_lane_fusion")

# ========================= cuBLAS ==========================
try:
    _libcublas = ctypes.cdll.LoadLibrary("libcublas.so")
    _libcublas.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    _libcublas.cublasCreate_v2.restype = ctypes.c_int
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
except OSError as e:
    logger.error(f"Failed to load libcublas.so: {e}")
    raise

CUBLAS_OP_N = 0
CUBLAS_OP_T = 1

# ===================== CUDA kernels (PTX) ==================
PTX_FILE = "postprocess_kernels.ptx"
if not os.path.exists(PTX_FILE):
    raise FileNotFoundError(
        f"Missing {PTX_FILE}. Compile postprocess_kernels.cu with nvcc to generate it."
    )
try:
    postproc_mod = cuda.module_from_file(PTX_FILE)
    resizeK   = postproc_mod.get_function("resizePrototypesKernel")
    sigmK     = postproc_mod.get_function("sigmoidThresholdKernel")
    dilateK   = postproc_mod.get_function("dilate5x5")
    erodeK    = postproc_mod.get_function("erode5x5")
    maxMinK   = postproc_mod.get_function("multiLaneMaxMinKernel")
    multiFillK= postproc_mod.get_function("multiLaneFillKernel")
except Exception as e:
    logger.error(f"Failed to load CUDA kernels from {PTX_FILE}: {e}")
    raise

# ===================== Color LUT for drawing ===============
_COLOUR_LUT = np.array([
    (0, 255, 255),  # yellow
    (255, 0, 255),  # magenta
    (0, 255, 0),    # green
    (255, 255, 0),  # cyan
    (255, 128, 0),  # orange
    (0, 128, 255),  # amber-blue
], dtype=np.uint8)

# ========================= Data classes ====================
@dataclass
class FusionResult:
    fused_lanes: np.ndarray                      # bool [H,W]
    boundary_masks: List[np.ndarray]             # list of bool [H,W]
    semantic_masks: List[np.ndarray]             # list of bool [H,W]
    lane_areas: List[np.ndarray]                 # list of bool [H,W] (between boundaries)
    confidence_map: np.ndarray                   # float32 [H,W]
    boundary_confidence: float
    semantic_confidence: float
    fusion_method: str
    processing_times: Dict[str, float]
    lane_centroids: List[Tuple[float, float]]

# ========================== Core class ======================
class DualEngineTensorRT:
    """Two-TRT-engine inference + fusion pipeline."""

    def __init__(self, boundary_engine_path, semantic_engine_path, config):
        # TRT runtime
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.trt_logger)

        # cuBLAS handle
        self.cublas_handle = ctypes.c_void_p()
        status = _libcublas.cublasCreate_v2(ctypes.byref(self.cublas_handle))
        if status != 0:
            raise RuntimeError(f"cublasCreate failed with status {status}")

        # Load engines & contexts
        self.boundary_engine = self._load_engine(boundary_engine_path)
        self.semantic_engine = self._load_engine(semantic_engine_path)
        self.boundary_context = self.boundary_engine.create_execution_context()
        self.semantic_context = self.semantic_engine.create_execution_context()

        # Config
        self.input_size = int(config.get("input_size", 640))
        self.target_fps = float(config.get("target_fps", 18))
        self.frame_skip_ratio = max(1.0, 30.0 / max(self.target_fps, 1e-3))
        self.confidence_params = config.get("confidence_params", {
            "boundary": {"edge_strength_threshold": 50, "continuity_threshold": 0.7, "parallelism_tolerance": 15},
            "semantic": {"coverage_min": 0.05, "coverage_max": 0.4, "consistency_threshold": 0.8}
        })
        self.visualization_config = config.get("visualization", {
            "show_boundary": True, "show_semantic": True, "show_fused": True, "show_constraint": True, "base_alpha": 0.4
        })

        # Thresholds for detections (these are reasonable starting points)
        self.BOUNDARY_THRESHOLD = 0.25
        self.SEMANTIC_THRESHOLD = 0.25

        # Allocate I/O
        self.boundary_io = self._allocate_io(self.boundary_engine, "boundary")
        self.semantic_io = self._allocate_io(self.semantic_engine, "semantic")

        # CUDA streams
        self.boundary_stream = cuda.Stream()
        self.semantic_stream = cuda.Stream()

        # Temporal & stats
        self.frame_counter = 0
        self.previous_boundary_masks: List[np.ndarray] = []
        self.previous_semantic_masks: List[np.ndarray] = []
        self.previous_lane_areas: List[np.ndarray] = []
        self.ema_alpha = 0.7  # EMA for masks

        self.kalman_filters: Dict[str, KalmanFilter] = {}  # per lane id

        self.timing_stats = {
            "boundary_inference": deque(maxlen=60),
            "semantic_inference": deque(maxlen=60),
            "fusion_processing": deque(maxlen=60),
            "total_processing": deque(maxlen=60),
            "cuda_kernels": deque(maxlen=60),
        }

        # Reusable device buffer for small ops
        self._scratch_dev_u8 = cuda.mem_alloc(self.input_size * self.input_size)

        logger.info("Allocated I/O for boundary engine: %d inputs, %d outputs",
                    len(self.boundary_io["inputs"]), len(self.boundary_io["outputs"]))
        logger.info("Allocated I/O for semantic engine: %d inputs, %d outputs",
                    len(self.semantic_io["inputs"]), len(self.semantic_io["outputs"]))

    # -------------------- Engine helpers --------------------
    def _load_engine(self, engine_path: str):
        with open(engine_path, "rb") as f:
            engine = self.runtime.deserialize_cuda_engine(f.read())
            if engine is None:
                raise RuntimeError(f"Failed to deserialize engine: {engine_path}")
            return engine

    def _allocate_io(self, engine, tag):
        io = {"inputs": [], "outputs": [], "bindings": []}
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            shape = engine.get_tensor_shape(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            size = trt.volume(shape)
            host = cuda.pagelocked_empty(size, dtype)
            dev = cuda.mem_alloc(host.nbytes)
            io["bindings"].append(int(dev))
            rec = {"name": name, "host": host, "device": dev, "shape": shape, "dtype": dtype}
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                io["inputs"].append(rec)
            else:
                io["outputs"].append(rec)
        return io

    # -------------------- Frame control ---------------------
    def should_process_frame(self) -> bool:
        self.frame_counter += 1
        # e.g. ratio=1.2 -> process every 1 or 2 frames; keep it simple (round)
        stride = max(1, int(round(self.frame_skip_ratio)))
        return (self.frame_counter % stride) == 0

    def preprocess_frame(self, frame):
        h0, w0 = frame.shape[:2]
        img = cv2.resize(frame, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[None, ...]
        return img, frame.copy(), (w0, h0)

    # -------------------- Inference -------------------------
    def _infer(self, img, context, io, stream):
        cuda.memcpy_htod_async(io["inputs"][0]["device"], img.ravel(), stream)
        for rec in io["inputs"]:
            context.set_tensor_address(rec["name"], rec["device"])
        for rec in io["outputs"]:
            context.set_tensor_address(rec["name"], rec["device"])
        context.execute_async_v3(stream.handle)
        for rec in io["outputs"]:
            cuda.memcpy_dtoh_async(rec["host"], rec["device"], stream)
        stream.synchronize()
        return [rec["host"].copy() for rec in io["outputs"]]

    def infer_boundary_engine(self, img):
        return self._infer(img, self.boundary_context, self.boundary_io, self.boundary_stream)

    def infer_semantic_engine(self, img):
        return self._infer(img, self.semantic_context, self.semantic_io, self.semantic_stream)

    # ---------------- YOLOv8-Seg parser ---------------------
    @staticmethod
    def _parse_yolov8_seg(outputs):
        """
        Returns: (proto[C,H,W], det[37,K], C,H,W)
        Format assumptions:
          outputs[0]: proto (C,H,W) or (1,C,H,W)
          outputs[1]: det (37,K) or (K,37)
        """
        proto = outputs[0].astype(np.float32)
        if proto.ndim == 4:
            proto = proto[0]
        C, H, W = proto.shape
        det = outputs[1].astype(np.float32)
        if det.shape[0] != 37 and det.shape[1] == 37:
            det = det.T
        return proto, det, C, H, W

    # --------------- Post-process: boundary -----------------
    def postprocess_boundary_masks(self, outputs, conf_threshold=0.25):
        if not outputs:
            return np.array([]), np.array([]), ([], [], 0, [])
        try:
            proto, det, C, H, W = self._parse_yolov8_seg(outputs)
            scores = det[4]
            keep = scores > conf_threshold
            if not np.any(keep):
                return np.array([]), np.array([]), ([], [], 0, [])

            mc = det[5:5+C, keep].astype(np.float32)         # (C, N)
            N = mc.shape[1]
            if N == 0:
                return np.array([]), np.array([]), ([], [], 0, [])

            # Upload proto & resize to input_size
            proto_dev = cuda.mem_alloc(proto.nbytes)
            cuda.memcpy_htod(proto_dev, proto)

            Cn, Hn, Wn = C, self.input_size, self.input_size
            proto_res_dev = cuda.mem_alloc(Cn * Hn * Wn * 4)
            grid = ((Wn + 15) // 16, (Hn + 15) // 16, Cn)
            resizeK(proto_dev, proto_res_dev,
                    np.int32(C), np.int32(H), np.int32(W),
                    np.int32(Hn), np.int32(Wn),
                    block=(16, 16, 1), grid=grid)
            cuda.Context.synchronize()

            # GEMM: (C,N)^T x (C,HW) -> (N,HW)
            mc_dev = cuda.mem_alloc(mc.nbytes)
            cuda.memcpy_htod(mc_dev, mc)
            HW = Hn * Wn
            lin_dev = cuda.mem_alloc(N * HW * 4)
            alpha, beta = ctypes.c_float(1.0), ctypes.c_float(0.0)
            stat = _libcublas.cublasSgemm_v2(
                self.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                N, HW, C, ctypes.byref(alpha),
                ctypes.c_void_p(int(mc_dev)), C,
                ctypes.c_void_p(int(proto_res_dev)), C,
                ctypes.byref(beta),
                ctypes.c_void_p(int(lin_dev)), N
            )
            if stat != 0:
                raise RuntimeError(f"cublasSgemm_v2 failed: {stat}")

            # Sigmoid + threshold on GPU
            bin_dev = cuda.mem_alloc(N * HW)
            sigmK(lin_dev, bin_dev, np.int32(N), np.int32(HW),
                  block=(256, 1, 1), grid=((HW + 255) // 256, N, 1))
            cuda.Context.synchronize()

            masks = np.empty((N, HW), dtype=np.uint8)
            cuda.memcpy_dtoh(masks, bin_dev)
            masks = masks.reshape(N, Hn, Wn)

            for d in (proto_dev, proto_res_dev, mc_dev, lin_dev, bin_dev):
                d.free()

            # Analyze boundaries -> lane areas, centroids
            boundary_masks, lane_areas, num_lanes, centroids = self._analyze_boundary_lanes(masks)
            return masks, scores[keep], (boundary_masks, lane_areas, num_lanes, centroids)
        except Exception as e:
            logger.error(f"Boundary postprocess failed: {e}")
            return np.array([]), np.array([]), ([], [], 0, [])

    # --------------- Post-process: semantic -----------------
    def postprocess_semantic_masks(self, outputs, conf_threshold=0.25):
        if not outputs:
            return [], []
        try:
            proto, det, C, H, W = self._parse_yolov8_seg(outputs)
            scores = det[4]
            keep = scores > conf_threshold
            if not np.any(keep):
                return [], []

            mc = det[5:5+C, keep].astype(np.float32)  # (C, N)
            N = mc.shape[1]
            if N == 0:
                return [], []

            # Upload proto & resize
            proto_dev = cuda.mem_alloc(proto.nbytes)
            cuda.memcpy_htod(proto_dev, proto)
            Hn, Wn = self.input_size, self.input_size
            proto_res_dev = cuda.mem_alloc(C * Hn * Wn * 4)
            grid = ((Wn + 15) // 16, (Hn + 15) // 16, C)
            resizeK(proto_dev, proto_res_dev, np.int32(C), np.int32(H), np.int32(W),
                    np.int32(Hn), np.int32(Wn), block=(16, 16, 1), grid=grid)
            cuda.Context.synchronize()

            # GEMM -> (N, HW)
            mc_dev = cuda.mem_alloc(mc.nbytes)
            cuda.memcpy_htod(mc_dev, mc)
            HW = Hn * Wn
            lin_dev = cuda.mem_alloc(N * HW * 4)
            alpha, beta = ctypes.c_float(1.0), ctypes.c_float(0.0)
            stat = _libcublas.cublasSgemm_v2(
                self.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                N, HW, C, ctypes.byref(alpha),
                ctypes.c_void_p(int(mc_dev)), C,
                ctypes.c_void_p(int(proto_res_dev)), C,
                ctypes.byref(beta),
                ctypes.c_void_p(int(lin_dev)), N
            )
            if stat != 0:
                raise RuntimeError(f"cublasSgemm_v2 failed: {stat}")

            bin_dev = cuda.mem_alloc(N * HW)
            sigmK(lin_dev, bin_dev, np.int32(N), np.int32(HW),
                  block=(256, 1, 1), grid=((HW + 255) // 256, N, 1))
            cuda.Context.synchronize()

            masks_u8 = np.empty((N, HW), dtype=np.uint8)
            cuda.memcpy_dtoh(masks_u8, bin_dev)
            masks_u8 = masks_u8.reshape(N, Hn, Wn)

            for d in (proto_dev, proto_res_dev, mc_dev, lin_dev, bin_dev):
                d.free()

            # Optional small opening to clean noise
            kernel = np.ones((3, 3), np.uint8)
            masks = []
            for i in range(N):
                m = cv2.erode(masks_u8[i], kernel, iterations=1)
                m = cv2.dilate(m, kernel, iterations=1)
                masks.append(m > 0)
            return masks, scores[keep]
        except Exception as e:
            logger.error(f"Semantic postprocess failed: {e}")
            return [], []

    # --------------- Analyze boundaries -> lanes ------------
    def _analyze_boundary_lanes(self, masks_u8: np.ndarray):
        """
        masks_u8: (N, H, W) uint8 boundary masks (0/1)
        Returns boundary_masks[bool], lane_areas[bool], num_lanes, centroids
        """
        if masks_u8.size == 0:
            return [], [], 0, []

        N, H, W = masks_u8.shape
        masks_dev = cuda.mem_alloc(masks_u8.nbytes)
        cuda.memcpy_htod(masks_dev, masks_u8)

        # Compute per-row maxX and minX for each mask
        maxX_dev = cuda.mem_alloc(N * H * 4)
        minX_dev = cuda.mem_alloc(N * H * 4)
        maxMinK(masks_dev, maxX_dev, minX_dev,
                np.int32(N), np.int32(H), np.int32(W),
                block=(256, 1, 1), grid=((H + 255) // 256, 1, N))
        cuda.Context.synchronize()

        maxX = np.empty((N, H), dtype=np.int32)
        minX = np.empty((N, H), dtype=np.int32)
        cuda.memcpy_dtoh(maxX, maxX_dev)
        cuda.memcpy_dtoh(minX, minX_dev)

        # Filter boundaries by mean x position
        boundary_info = []
        for n in range(N):
            valid = maxX[n] >= 0
            if np.any(valid):
                mean_x = np.mean(maxX[n][valid])
                boundary_info.append((float(mean_x), n))

        # Need at least two boundaries to form a lane
        if len(boundary_info) < 2:
            for d in (masks_dev, maxX_dev, minX_dev):
                d.free()
            return [], [], 0, []

        # NMS on mean_x (suppresses close-by boundaries)
        boundary_info.sort(key=lambda p: p[0])
        kept = []
        i = 0
        nms_thresh = 20  # pixels
        while i < len(boundary_info):
            kept.append(boundary_info[i])
            j = i + 1
            while j < len(boundary_info) and (boundary_info[j][0] - boundary_info[i][0]) < nms_thresh:
                j += 1
            i = j

        sorted_idx = [idx for _, idx in kept]
        num_boundaries = len(sorted_idx)
        num_lanes = max(0, num_boundaries - 1)
        if num_lanes == 0:
            for d in (masks_dev, maxX_dev, minX_dev):
                d.free()
            return [], [], 0, []

        # Prepare tensors to fill lanes between consecutive boundaries
        maxLeft = maxX[sorted_idx[:-1]]
        minRight = minX[sorted_idx[1:]]
        maxLeft_dev = cuda.mem_alloc(maxLeft.nbytes)
        minRight_dev = cuda.mem_alloc(minRight.nbytes)
        cuda.memcpy_htod(maxLeft_dev, maxLeft)
        cuda.memcpy_htod(minRight_dev, minRight)

        areas_dev = cuda.mem_alloc(num_lanes * H * W)  # uint8
        multiFillK(maxLeft_dev, minRight_dev, areas_dev,
                   np.int32(num_lanes), np.int32(H), np.int32(W),
                   block=(16, 16, 1), grid=((W + 15) // 16, (H + 15) // 16, num_lanes))
        cuda.Context.synchronize()

        areas = np.empty((num_lanes, H, W), dtype=np.uint8)
        cuda.memcpy_dtoh(areas, areas_dev)

        boundary_masks = [masks_u8[idx] > 0 for idx in sorted_idx]
        lane_areas = [(areas[i] > 0) for i in range(num_lanes)]

        # Centroids for each lane area
        centroids = []
        for area in lane_areas:
            if np.any(area):
                ys, xs = np.nonzero(area)
                centroids.append((float(np.mean(xs)), float(np.mean(ys))))
            else:
                centroids.append((W / 2.0, H / 2.0))

        for d in (masks_dev, maxX_dev, minX_dev, maxLeft_dev, minRight_dev, areas_dev):
            d.free()

        return boundary_masks, lane_areas, num_lanes, centroids

    # ---------------- Confidence analysis -------------------
    def analyze_confidence(self, boundary_result, semantic_masks, semantic_scores):
        boundary_masks, _, _, _ = boundary_result if len(boundary_result) == 4 else ([], [], 0, [])
        # Boundary confidence: combine edge continuity + simple gradient magnitude proxy
        if boundary_masks:
            cont_scores = []
            edge_scores = []
            for m in boundary_masks:
                mu8 = m.astype(np.uint8)
                contours, _ = cv2.findContours(mu8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    main = max(contours, key=cv2.contourArea)
                    hull = cv2.convexHull(main)
                    hull_len = max(cv2.arcLength(hull, False), 1e-6)
                    cont_scores.append(min(cv2.arcLength(main, False) / hull_len, 1.0))
                else:
                    cont_scores.append(0.0)
                gx = cv2.Sobel(mu8, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(mu8, cv2.CV_32F, 0, 1, ksize=3)
                mag = np.sqrt(gx * gx + gy * gy)
                edge_scores.append(float(np.mean(mag[mu8 > 0])) / 100.0)
            boundary_conf = float(np.clip(np.mean([np.mean(cont_scores), np.mean(edge_scores)]), 0, 1))
        else:
            boundary_conf = 0.5

        # Semantic confidence: coverage in a reasonable band + mean score
        if semantic_masks:
            total = self.input_size * self.input_size
            coverage = sum(int(m.sum()) for m in semantic_masks) / float(total)
            p = self.confidence_params["semantic"]
            if p["coverage_min"] <= coverage <= p["coverage_max"]:
                cov_score = 1.0
            elif coverage < p["coverage_min"]:
                cov_score = coverage / max(p["coverage_min"], 1e-6)
            else:
                cov_score = max(0.0, 1.0 - (coverage - p["coverage_max"]) / max(1.0 - p["coverage_max"], 1e-6))
            s_mean = float(np.mean(semantic_scores)) if len(semantic_scores) else 0.5
            semantic_conf = float(np.clip(0.7 * cov_score + 0.3 * s_mean, 0, 1))
        else:
            semantic_conf = 0.5

        return boundary_conf, semantic_conf

    # --------------- Temporal smoothing (EMA) ----------------
    def _smooth_masks(self, current_list: List[np.ndarray], previous_list: List[np.ndarray]) -> List[np.ndarray]:
        if not previous_list or len(previous_list) != len(current_list):
            return [m.copy() for m in current_list]
        smoothed = []
        for curr, prev in zip(current_list, previous_list):
            s = (self.ema_alpha * curr.astype(np.float32) + (1 - self.ema_alpha) * prev.astype(np.float32)) > 0.5
            smoothed.append(s)
        return smoothed

    # ---------------- Constraint mask -----------------------
    def create_boundary_constraint_mask(self, boundary_masks, lane_areas):
        mask = np.zeros((self.input_size, self.input_size), dtype=np.bool_)
        if boundary_masks:
            for m in boundary_masks:
                mu8 = m.astype(np.uint8)
                cuda.memcpy_htod(self._scratch_dev_u8, mu8)
                dilateK(self._scratch_dev_u8, self._scratch_dev_u8,
                        np.int32(self.input_size), np.int32(self.input_size),
                        block=(16, 16, 1),
                        grid=((self.input_size + 15) // 16, (self.input_size + 15) // 16, 1))
                tmp = np.empty_like(mu8)
                cuda.memcpy_dtoh(tmp, self._scratch_dev_u8)
                mask |= (tmp > 0)
        elif lane_areas:
            for a in lane_areas:
                au8 = a.astype(np.uint8)
                cuda.memcpy_htod(self._scratch_dev_u8, au8)
                dilateK(self._scratch_dev_u8, self._scratch_dev_u8,
                        np.int32(self.input_size), np.int32(self.input_size),
                        block=(16, 16, 1),
                        grid=((self.input_size + 15) // 16, (self.input_size + 15) // 16, 1))
                tmp = np.empty_like(au8)
                cuda.memcpy_dtoh(tmp, self._scratch_dev_u8)
                mask |= (tmp > 0)
        else:
            # fallback: drivable lower 60%
            mask[int(self.input_size * 0.4):, :] = True
        return mask

    # ---------------- Fusion algorithms ---------------------
    def _boundary_dominant_fusion(self, boundary_masks, lane_areas, semantic_masks):
        if not lane_areas:
            return np.zeros((self.input_size, self.input_size), dtype=bool)
        primary = np.logical_or.reduce(lane_areas)
        if semantic_masks:
            sem = np.logical_or.reduce(semantic_masks)
            return np.logical_or(primary, np.logical_and(primary, sem))
        return primary

    def _semantic_dominant_fusion(self, semantic_masks, boundary_masks):
        if not semantic_masks:
            return np.zeros((self.input_size, self.input_size), dtype=bool)
        sem = np.logical_or.reduce(semantic_masks)
        if boundary_masks:
            corridor = self.create_boundary_constraint_mask(boundary_masks, [])
            return np.logical_and(sem, corridor)
        return sem

    def _balanced_fusion(self, boundary_masks, lane_areas, semantic_masks, b_conf, s_conf):
        b = np.logical_or.reduce(lane_areas) if lane_areas else np.zeros((self.input_size, self.input_size), bool)
        s = np.logical_or.reduce(semantic_masks) if semantic_masks else np.zeros((self.input_size, self.input_size), bool)
        total = b_conf + s_conf + 1e-6
        bf = b.astype(np.float32) * (b_conf / total)
        sf = s.astype(np.float32) * (s_conf / total)
        return (bf + sf) > 0.4

    def fusion_algorithm(self, boundary_result, semantic_masks, semantic_scores, boundary_conf, semantic_conf):
        boundary_masks, lane_areas, num_lanes, centroids = boundary_result if len(boundary_result) == 4 else ([], [], 0, [])
        if not lane_areas and not semantic_masks:
            logger.warning("No valid detections for fusion")
            return np.zeros((self.input_size, self.input_size), dtype=bool), "none"

        if abs(boundary_conf - semantic_conf) > 0.3:
            if boundary_conf > semantic_conf:
                return self._boundary_dominant_fusion(boundary_masks, lane_areas, semantic_masks), "boundary_dominant"
            else:
                return self._semantic_dominant_fusion(semantic_masks, boundary_masks), "semantic_dominant"
        return self._balanced_fusion(boundary_masks, lane_areas, semantic_masks, boundary_conf, semantic_conf), "balanced"

    # ----------------- Kalman filters -----------------------
    def _kf(self, lane_id, centroid):
        if lane_id not in self.kalman_filters:
            kf = KalmanFilter(dim_x=4, dim_z=2)
            kf.x = np.array([centroid[0], centroid[1], 0, 0], dtype=np.float32)
            kf.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], dtype=np.float32)
            kf.H = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float32)
            kf.P *= 1000.0
            kf.R = np.array([[5,0],[0,5]], dtype=np.float32)
            kf.Q = np.eye(4, dtype=np.float32) * 0.1
            self.kalman_filters[lane_id] = kf
        return self.kalman_filters[lane_id]

    def update_kalman_filters(self, centroids):
        out = []
        for i, c in enumerate(centroids):
            kf = self._kf(f"lane_{i}", c)
            kf.update(np.array(c, dtype=np.float32))
            kf.predict()
            out.append((float(kf.x[0]), float(kf.x[1])))
        return out

    # --------------------- Main step ------------------------
    def process_frame(self, frame):
        if not self.should_process_frame():
            return None

        total_t0 = time.perf_counter()
        img, orig, _ = self.preprocess_frame(frame)

        # Boundary path
        t1 = time.perf_counter()
        b_out = self.infer_boundary_engine(img)
        t2 = time.perf_counter()
        b_masks_raw, b_scores, b_result = self.postprocess_boundary_masks(b_out, self.BOUNDARY_THRESHOLD)
        boundary_masks, lane_areas, num_lanes, centroids = b_result if len(b_result) == 4 else ([], [], 0, [])
        constraint_mask = self.create_boundary_constraint_mask(boundary_masks, lane_areas)

        # Semantic path
        t3 = time.perf_counter()
        s_out = self.infer_semantic_engine(img)
        t4 = time.perf_counter()
        semantic_masks_raw, semantic_scores = self.postprocess_semantic_masks(s_out, self.SEMANTIC_THRESHOLD)
        # Apply constraint
        semantic_masks = [np.logical_and(m, constraint_mask) for m in semantic_masks_raw]

        # Confidence
        t5 = time.perf_counter()
        b_conf, s_conf = self.analyze_confidence((boundary_masks, lane_areas, num_lanes, centroids),
                                                 semantic_masks, semantic_scores)

        # Temporal smoothing (EMA)
        boundary_masks = self._smooth_masks(boundary_masks, self.previous_boundary_masks)
        semantic_masks = self._smooth_masks(semantic_masks, self.previous_semantic_masks)
        lane_areas     = self._smooth_masks(lane_areas, self.previous_lane_areas)
        self.previous_boundary_masks = boundary_masks
        self.previous_semantic_masks = semantic_masks
        self.previous_lane_areas     = lane_areas

        # Kalman centroids
        centroids = self.update_kalman_filters(centroids)

        # Fusion
        fused, method = self.fusion_algorithm((boundary_masks, lane_areas, num_lanes, centroids),
                                              semantic_masks, semantic_scores, b_conf, s_conf)
        t6 = time.perf_counter()

        # Stats
        self.timing_stats["boundary_inference"].append((t2 - t1) * 1000)
        self.timing_stats["semantic_inference"].append((t4 - t3) * 1000)
        self.timing_stats["fusion_processing"].append((t6 - t5) * 1000)
        self.timing_stats["total_processing"].append((t6 - total_t0) * 1000)
        self.timing_stats["cuda_kernels"].append((t3 - t2 + t5 - t4) * 1000)

        return FusionResult(
            fused_lanes=fused,
            boundary_masks=boundary_masks,
            semantic_masks=semantic_masks,
            lane_areas=lane_areas,
            confidence_map=constraint_mask.astype(np.float32) * max(b_conf, s_conf),
            boundary_confidence=b_conf,
            semantic_confidence=s_conf,
            fusion_method=method,
            processing_times={
                "boundary_inference": (t2 - t1) * 1000,
                "semantic_inference": (t4 - t3) * 1000,
                "fusion_processing": (t6 - t5) * 1000,
                "total": (t6 - total_t0) * 1000,
            },
            lane_centroids=centroids,
        )

    # -------------------- Visualizer ------------------------
    def visualize_fusion_result(self, original_frame, fusion_result: FusionResult):
        vis = cv2.resize(original_frame, (self.input_size, self.input_size))
        boundary_overlay  = np.zeros_like(vis)
        semantic_overlay  = np.zeros_like(vis)
        lanes_overlay     = np.zeros_like(vis)
        constraint_overlay= np.zeros_like(vis)

        # Constraint
        if self.visualization_config["show_constraint"]:
            cmask = fusion_result.confidence_map > 0
            constraint_overlay[cmask] = (100, 50, 0)
            vis = cv2.addWeighted(vis, 1.0, constraint_overlay,
                                  self.visualization_config["base_alpha"] * float(fusion_result.confidence_map.max()),
                                  0)

        # Semantic (cyan)
        if self.visualization_config["show_semantic"]:
            for m in fusion_result.semantic_masks:
                semantic_overlay[m] = (255, 255, 0)
            vis = cv2.addWeighted(vis, 1.0, semantic_overlay,
                                  self.visualization_config["base_alpha"] * fusion_result.semantic_confidence, 0)

        # Fused lanes (colored per-lane area)
        if self.visualization_config["show_fused"] and fusion_result.lane_areas:
            for i, area in enumerate(fusion_result.lane_areas):
                color = tuple(int(c) for c in _COLOUR_LUT[i % len(_COLOUR_LUT)])
                lanes_overlay[area] = color
            vis = cv2.addWeighted(vis, 1.0, lanes_overlay,
                                  self.visualization_config["base_alpha"] *
                                  max(fusion_result.boundary_confidence, fusion_result.semantic_confidence), 0)

        # Boundaries (white/yellow/magenta/amber)
        if self.visualization_config["show_boundary"]:
            bcolors = [(255, 255, 255), (0, 255, 255), (255, 0, 255), (0, 128, 255)]
            for i, m in enumerate(fusion_result.boundary_masks):
                boundary_overlay[m] = bcolors[i % len(bcolors)]
            vis = cv2.addWeighted(vis, 1.0, boundary_overlay,
                                  self.visualization_config["base_alpha"] * fusion_result.boundary_confidence, 0)

        # HUD
        constraint_area = float(np.sum(fusion_result.confidence_map > 0)) / (self.input_size * self.input_size) * 100.0
        hud = [
            f"Fusion Method: {fusion_result.fusion_method}",
            f"Boundary Conf: {fusion_result.boundary_confidence:.3f}",
            f"Semantic Conf: {fusion_result.semantic_confidence:.3f}",
            f"Constraint Area: {constraint_area:.1f}%",
            f"Processing: {fusion_result.processing_times['total']:.1f}ms",
            f"Lanes Detected: {len(fusion_result.lane_centroids)}",
        ]
        for i, line in enumerate(hud):
            cv2.putText(vis, line, (10, 28 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Legend
        y0 = self.input_size - 100
        def leg(y, color, text):
            cv2.rectangle(vis, (10, y-12), (28, y+6), color, -1)
            cv2.putText(vis, text, (35, y+4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        if self.visualization_config["show_constraint"]: leg(y0, (100, 50, 0), "Constraint (c)")
        if self.visualization_config["show_boundary"]:   leg(y0+22, (255,255,255), "Boundary (b)")
        if self.visualization_config["show_semantic"]:   leg(y0+44, (255,255,0), "Semantic (s)")
        if self.visualization_config["show_fused"]:      leg(y0+66, (0,255,0), "Fused (f)")

        return vis

    # ---------------- Key toggles & stats -------------------
    def toggle_visualization(self, key):
        kmap = {ord('c'): "show_constraint", ord('b'): "show_boundary", ord('s'): "show_semantic", ord('f'): "show_fused"}
        if key in kmap:
            k = kmap[key]
            self.visualization_config[k] = not self.visualization_config[k]
            logger.info("Toggled %s: %s", k, self.visualization_config[k])

    def get_performance_stats(self):
        stats = {}
        for k, v in self.timing_stats.items():
            if len(v):
                stats[k] = {"mean": float(np.mean(v)), "std": float(np.std(v)),
                            "min": float(np.min(v)), "max": float(np.max(v)),
                            "samples": len(v)}
        free, total = cuda.mem_get_info()
        stats["memory"] = {"free": free / (1024**2), "total": total / (1024**2),
                           "used": (total - free) / (1024**2)}
        return stats


# ====================== Utilities ===========================
def setup_camera(source: str):
    try:
        is_video_file = False
        if source.isdigit():
            cap = cv2.VideoCapture(int(source))
        else:
            cap = cv2.VideoCapture(source)
            is_video_file = not source.startswith(("rtsp://", "http://", "https://"))
        if not cap.isOpened():
            raise ValueError(f"Cannot open source: {source}")
        return cap, is_video_file, None
    except Exception as e:
        logger.error(f"Camera setup failed: {e}")
        raise

def load_config(config_path: str):
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load config {config_path}: {e}. Using default config.")
        return {
            "input_size": 640,
            "target_fps": 18,
            "confidence_params": {
                "boundary": {"edge_strength_threshold": 50, "continuity_threshold": 0.7, "parallelism_tolerance": 15},
                "semantic": {"coverage_min": 0.05, "coverage_max": 0.4, "consistency_threshold": 0.8},
            },
            "visualization": {
                "show_boundary": True, "show_semantic": True, "show_fused": True, "show_constraint": True, "base_alpha": 0.4,
            },
        }

# ========================= Main =============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dual-Engine Lane Fusion")
    parser.add_argument("--boundary-engine", required=True, help="Path to boundary TensorRT engine")
    parser.add_argument("--semantic-engine", required=True, help="Path to semantic TensorRT engine")
    parser.add_argument("--source", default="0", help="Video source (file path, camera index, or URL)")
    parser.add_argument("--input-size", type=int, default=640, help="Processing input size")
    parser.add_argument("--display-size", type=int, default=640, help="Display size (unused; using input-size for now)")
    parser.add_argument("--target-fps", type=float, default=18, help="Target processing FPS")
    parser.add_argument("--config", default="config.json", help="Path to configuration JSON file")
    args = parser.parse_args()

    config = load_config(args.config)
    config["input_size"] = int(args.input_size)
    config["target_fps"] = float(args.target_fps)

    print("Initializing dual-engine fusion system...")
    fusion = DualEngineTensorRT(args.boundary_engine, args.semantic_engine, config)
    cap, is_video, _ = setup_camera(args.source)

    print("Starting fusion processing...")
    print(f"Target FPS: {args.target_fps} (processing every {fusion.frame_skip_ratio:.1f} frames)")

    frame_count = 0
    processed_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if is_video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            frame_count += 1
            result = fusion.process_frame(frame)
            if result is not None:
                processed_count += 1
                vis = fusion.visualize_fusion_result(frame, result)
                cv2.imshow("Dual-Engine Lane Fusion", vis)

                if processed_count % 30 == 0:
                    stats = fusion.get_performance_stats()
                    print(f"\n=== Performance Stats (Frame {frame_count}, Processed {processed_count}) ===")
                    for k, s in stats.items():
                        if k == "memory":
                            continue
                        print(f"{k}: {s['mean']:.1f}±{s['std']:.1f}ms [{s['min']:.1f}-{s['max']:.1f}] ({s['samples']} samples)")
                    print(f"Memory: {stats['memory']['used']:.1f}/{stats['memory']['total']:.1f} MB")
                    print(f"Latest: Method={result.fusion_method}, "
                          f"B_conf={result.boundary_confidence:.3f}, "
                          f"S_conf={result.semantic_confidence:.3f}, "
                          f"Lanes={len(result.lane_centroids)}")

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            fusion.toggle_visualization(key)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
    finally:
        print("\n=== Final Statistics ===")
        print(f"Total frames: {frame_count}")
        print(f"Processed frames: {processed_count}")
        print(f"Processing ratio: {processed_count / frame_count:.2f}" if frame_count > 0 else "Processing ratio: N/A")
        stats = fusion.get_performance_stats()
        for k, s in stats.items():
            if k == "memory":
                continue
            print(f"{k}: {s['mean']:.1f}±{s['std']:.1f}ms")
        print(f"Memory: {stats['memory']['used']:.1f}/{stats['memory']['total']:.1f} MB")
        cap.release()
        cv2.destroyAllWindows()
