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
from scipy.optimize import linear_sum_assignment

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load cuBLAS
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

# Load CUDA kernels
ptx_file = "postprocess_kernels.ptx"
if not os.path.exists(ptx_file):
    raise FileNotFoundError(f"CUDA PTX file {ptx_file} not found. Please compile postprocess_kernels.cu with nvcc.")
try:
    postproc_mod = cuda.module_from_file(ptx_file)
    resizeK = postproc_mod.get_function("resizePrototypesKernel")
    sigmK = postproc_mod.get_function("sigmoidThresholdKernel")
    dilateK = postproc_mod.get_function("dilate5x5")
    erodeK = postproc_mod.get_function("erode5x5")
    minMaxK = postproc_mod.get_function("multiLaneMinMaxKernel")
    fillK = postproc_mod.get_function("multiLaneFillKernel")
except Exception as e:
    logger.error(f"Failed to load CUDA kernels from {ptx_file}: {e}")
    raise

# Color LUT for multi-lane coloring
_colour_lut = np.array([
    (0, 255, 255),  # yellow
    (255, 0, 255),  # magenta
    (0, 255, 0),    # green
    (255, 255, 0),  # cyan
    (255, 128, 0),  # orange
    (0, 128, 255),  # amber-blue
], dtype=np.uint8)

lut_dev = cuda.mem_alloc(_colour_lut.nbytes)
cuda.memcpy_htod(lut_dev, _colour_lut)


@dataclass
class LaneTrack:
    """Persistent lane track"""
    track_id: int
    color: Tuple[int, int, int]
    mask: np.ndarray
    mask_history: deque  # Last N masks for temporal smoothing
    centroid: Tuple[float, float]
    kalman_filter: KalmanFilter
    confidence: float
    age: int
    last_seen: int

class LaneTrackManager:
    """Manages persistent lane tracks"""
    
    def __init__(self, max_tracks=6, history_length=5):
        self.tracks: Dict[int, LaneTrack] = {}
        self.next_track_id = 0
        self.color_palette = deque(_colour_lut.tolist())
        self.used_colors: Dict[int, Tuple] = {}
        self.max_tracks = max_tracks
        self.history_length = history_length
    def _greedy_matching(self, track_ids, lane_areas, centroids, cost_matrix):
        """Fallback greedy matching when Hungarian fails"""
        matched_pairs = []
        matched_tracks = set()
        matched_detections = set()
        
        # Flatten and sort by cost
        candidates = []
        for i in range(len(track_ids)):
            for j in range(len(lane_areas)):
                if cost_matrix[i, j] < 0.8:
                    candidates.append((cost_matrix[i, j], i, j))
        
        candidates.sort()
        
        for cost, i, j in candidates:
            if i not in matched_tracks and j not in matched_detections:
                matched_pairs.append((track_ids[i], lane_areas[j], centroids[j]))
                matched_tracks.add(i)
                matched_detections.add(j)
        
        unmatched_tracks = [track_ids[i] for i in range(len(track_ids)) if i not in matched_tracks]
        unmatched_detections = [(lane_areas[j], centroids[j]) for j in range(len(lane_areas)) if j not in matched_detections]
        
        return matched_pairs, unmatched_tracks, unmatched_detections
    
    def associate_lanes(self, lane_areas, centroids, current_frame):
        """Hungarian algorithm matching with safety checks"""
        if not self.tracks:
            return [], [], list(zip(lane_areas, centroids))
        
        if not lane_areas or not centroids:
            return [], list(self.tracks.keys()), []
        
        # Safety: Ensure same length
        if len(lane_areas) != len(centroids):
            logger.warning(f"Mismatch: {len(lane_areas)} lanes vs {len(centroids)} centroids")
            min_len = min(len(lane_areas), len(centroids))
            lane_areas = lane_areas[:min_len]
            centroids = centroids[:min_len]
        
        track_ids = list(self.tracks.keys())
        cost_matrix = np.zeros((len(track_ids), len(lane_areas)))
        
        for i, track_id in enumerate(track_ids):
            track = self.tracks[track_id]
            for j, centroid in enumerate(centroids):
                dist = np.linalg.norm(np.array(track.centroid) - np.array(centroid))
                cost_matrix[i, j] = dist / 640.0
        
        # Handle edge cases for Hungarian algorithm
        if cost_matrix.size == 0:
            return [], list(self.tracks.keys()), list(zip(lane_areas, centroids))
        
        # If only one track or one detection, use simple matching
        if len(track_ids) == 1 and len(lane_areas) == 1:
            if cost_matrix[0, 0] < 0.5:
                return [(track_ids[0], lane_areas[0], centroids[0])], [], []
            else:
                return [], [track_ids[0]], [(lane_areas[0], centroids[0])]
        
        try:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        except ValueError as e:
            logger.error(f"Hungarian algorithm failed: {e}, cost_matrix shape: {cost_matrix.shape}")
            # Fallback: greedy matching
            return self._greedy_matching(track_ids, lane_areas, centroids, cost_matrix)
        
        matched_pairs = []
        matched_tracks = []
        matched_detections = []
        
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < 0.5:
                matched_pairs.append((track_ids[i], lane_areas[j], centroids[j]))
                matched_tracks.append(i)
                matched_detections.append(j)
        
        unmatched_tracks = [track_ids[i] for i in range(len(track_ids)) if i not in matched_tracks]
        unmatched_detections = [(lane_areas[j], centroids[j]) for j in range(len(lane_areas)) if j not in matched_detections]
        
        return matched_pairs, unmatched_tracks, unmatched_detections

    
    def create_track(self, lane_mask, centroid, current_frame):
        track_id = self.next_track_id
        self.next_track_id += 1
        
        if self.color_palette:
            color_list = self.color_palette.popleft()
            color = tuple(int(c) for c in color_list)
        else:
            idx = track_id % len(_colour_lut)
            # Ensure proper tuple conversion from numpy array
            color_array = _colour_lut[idx]
            color = (int(color_array[0]), int(color_array[1]), int(color_array[2]))
        
        
        self.used_colors[track_id] = color
        
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.x = np.array([centroid[0], centroid[1], 0, 0], dtype=np.float32)
        kf.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        kf.P *= 1000.0
        kf.R = np.array([[5, 0], [0, 5]], dtype=np.float32)
        kf.Q = np.eye(4) * 0.1
        
        mask_history = deque(maxlen=self.history_length)
        mask_history.append(lane_mask.copy())
        
        track = LaneTrack(
            track_id=track_id,
            color=color,
            mask=lane_mask.copy(),
            mask_history=mask_history,
            centroid=centroid,
            kalman_filter=kf,
            confidence=1.0,
            age=0,
            last_seen=current_frame
        )
        
        self.tracks[track_id] = track
        return track
    
    def update_track(self, track_id, lane_mask, centroid, current_frame):
        if track_id not in self.tracks:
            return
        
        track = self.tracks[track_id]
        track.kalman_filter.update(np.array(centroid, dtype=np.float32))
        track.kalman_filter.predict()
        
        # Temporal mask averaging
        track.mask_history.append(lane_mask.copy())
        if len(track.mask_history) >= 3:
            weights = np.exp(np.linspace(-1, 0, len(track.mask_history)))
            weights /= weights.sum()
            
            mask_float = np.zeros_like(lane_mask, dtype=np.float32)
            for mask, weight in zip(track.mask_history, weights):
                mask_float += mask.astype(np.float32) * weight
            
            track.mask = mask_float > 0.4
        else:
            track.mask = lane_mask.copy()
        
        track.centroid = (track.kalman_filter.x[0], track.kalman_filter.x[1])
        track.age += 1
        track.last_seen = current_frame
    
    def prune_dead_tracks(self, current_frame, max_age=8):
        dead = [tid for tid, t in self.tracks.items() if current_frame - t.last_seen > max_age]
        for tid in dead:
            track = self.tracks[tid]
            if track.color in self.used_colors.values():
                self.color_palette.append(track.color)
            del self.tracks[tid]
            del self.used_colors[tid]
    
    def get_sorted_tracks(self):
        tracks = list(self.tracks.values())
        tracks.sort(key=lambda t: t.centroid[0])
        return tracks
    
    
@dataclass
class FusionResult:
    fused_lanes: np.ndarray
    boundary_masks: List[np.ndarray]
    semantic_masks: List[np.ndarray]
    confidence_map: np.ndarray
    boundary_confidence: float
    semantic_confidence: float
    fusion_method: str
    processing_times: Dict[str, float]
    lane_centroids: List[Tuple[float, float]]

class DualEngineTensorRT:
    """Improved TensorRT inference class for dual-engine operation with enhancements"""

    def __init__(self, boundary_engine_path, semantic_engine_path, config):
        # ---- Logging / bookkeeping
        self.logger_py = logging.getLogger(__name__)
        self.track_manager = LaneTrackManager(max_tracks=6, history_length=5)
        self.current_frame_number = 0

        # ---- Thresholds
        self.BOUNDARY_THRESHOLD = 0.20
        self.SEMANTIC_THRESHOLD = 0.25

        # ---- Sizes & config FIRST (so ROI can use input_size)
        self.input_size = config.get('input_size', 640)
        self.confidence_params = config.get('confidence_params', {
            'boundary': {'edge_strength_threshold': 50, 'continuity_threshold': 0.7, 'parallelism_tolerance': 15},
            'semantic': {'coverage_min': 0.05, 'coverage_max': 0.4, 'consistency_threshold': 0.8}
        })
        self.dilation_kernel_size = config.get('dilation_kernel_size', 15)
        self.target_fps = config.get('target_fps', 18)
        self.frame_skip_ratio = 30 / self.target_fps
        self.visualization_config = config.get('visualization', {
            'show_boundary': True,
            'show_semantic': True,
            'show_fused': True,
            'show_constraint': True,
            'base_alpha': 0.4
        })

        # ---- ROI (safe now that input_size is known)
        self.roi_mask = None
        self.roi_config = None
        roi_path = config.get('roi_config_path', 'pent_roi_config.json')
        if os.path.exists(roi_path):
            self._load_roi_mask(roi_path)
            self.logger_py.info(f"Loaded ROI mask from {roi_path}")
        else:
            self.logger_py.warning(f"No ROI config found at {roi_path}, processing full frame")

        # ---- TensorRT runtime / cuBLAS
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        self.cublas_handle = ctypes.c_void_p()
        status = _libcublas.cublasCreate_v2(ctypes.byref(self.cublas_handle))
        if status != 0:
            raise RuntimeError(f"cublasCreate failed with {status}")

        # ---- Load engines & create contexts
        self.boundary_engine = self._load_engine(boundary_engine_path)
        self.semantic_engine = self._load_engine(semantic_engine_path)
        self.boundary_context = self.boundary_engine.create_execution_context()
        self.semantic_context = self.semantic_engine.create_execution_context()
        if self.boundary_context is None or self.semantic_context is None:
            raise RuntimeError("Failed to create execution contexts")

        # ---- Allocate I/O
        self.boundary_io = self._allocate_io(self.boundary_engine, "boundary")
        self.semantic_io = self._allocate_io(self.semantic_engine, "semantic")

        # ---- CUDA streams for async infer
        self.boundary_stream = cuda.Stream()
        self.semantic_stream = cuda.Stream()

        # ---- Temporal state
        self.kalman_filters: Dict[str, KalmanFilter] = {}
        self.frame_counter = 0
        self.previous_result = None
        self.previous_boundary_masks = None
        self.previous_semantic_masks = None
        self.previous_lane_areas = None
        self.ema_alpha = 0.7

        # ---- Perf stats
        from collections import deque
        self.timing_stats = {
            'boundary_inference': deque(maxlen=30),
            'semantic_inference': deque(maxlen=30),
            'fusion_processing': deque(maxlen=30),
            'total_processing': deque(maxlen=30),
            'cuda_kernels': deque(maxlen=30),
        }

        # ---- Basic engine I/O sanity logs (optional but helpful)
        try:
            b_in_names = [t['name'] for t in self.boundary_io['inputs']]
            b_out_names = [t['name'] for t in self.boundary_io['outputs']]
            s_in_names = [t['name'] for t in self.semantic_io['inputs']]
            s_out_names = [t['name'] for t in self.semantic_io['outputs']]
            self.logger_py.info(f"Boundary I/O -> inputs: {b_in_names}, outputs: {b_out_names}")
            self.logger_py.info(f"Semantic I/O -> inputs: {s_in_names}, outputs: {s_out_names}")
        except Exception:
            pass

    
    def _load_roi_mask(self, roi_path):
        """Load ROI polygon and create binary mask"""
        try:
            with open(roi_path, 'r') as f:
                self.roi_config = json.load(f)

            # Denormalize points to current processing resolution
            points_norm = self.roi_config['roi_points_normalized']
            pts = np.array(
                [[int(p['x'] * self.input_size), int(p['y'] * self.input_size)] for p in points_norm],
                dtype=np.int32
            )

            # OpenCV expects uint8/UMat, not bool
            roi_u8 = np.zeros((self.input_size, self.input_size), dtype=np.uint8)
            cv2.fillPoly(roi_u8, [pts], 1)          # fill with 1s
            self.roi_mask = roi_u8.astype(bool)     # convert to boolean for the rest of the pipeline

            roi_coverage = float(roi_u8.sum()) / (self.input_size ** 2)
            self.logger_py.info(f"ROI covers {roi_coverage*100:.1f}% of frame")

        except Exception as e:
            self.logger_py.error(f"Failed to load ROI: {e}")
            self.roi_mask = None

    def _apply_roi_mask(self, masks_list):
        """Apply ROI mask to list of detection masks"""
        if self.roi_mask is None:
            return masks_list
        
        masked_list = []
        for mask in masks_list:
            masked = np.logical_and(mask, self.roi_mask)
            # Only keep if >10% of original detection remains within ROI
            if np.sum(masked) > 0.1 * np.sum(mask):
                masked_list.append(masked)
        
        return masked_list
    
    def _load_engine(self, engine_path):
        """Load TensorRT engine from file"""
        try:
            with open(engine_path, "rb") as f:
                engine = self.runtime.deserialize_cuda_engine(f.read())
                if engine is None:
                    raise RuntimeError(f"Failed to deserialize engine {engine_path}")
                return engine
        except Exception as e:
            self.logger_py.error(f"Failed to load engine {engine_path}: {e}")
            raise

    def _allocate_io(self, engine, engine_name):
        """Allocate I/O tensors for an engine"""
        io_data = {'inputs': [], 'outputs': [], 'bindings': []}
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            shape = engine.get_tensor_shape(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            size = trt.volume(shape)

            try:
                host_mem = cuda.pagelocked_empty(size, dtype)
                dev_mem = cuda.mem_alloc(host_mem.nbytes)
                io_data['bindings'].append(int(dev_mem))
                tensor_info = {
                    "name": name,
                    "host": host_mem,
                    "device": dev_mem,
                    "shape": shape,
                    "dtype": dtype
                }
                if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    io_data['inputs'].append(tensor_info)
                else:
                    io_data['outputs'].append(tensor_info)
            except Exception as e:
                self.logger_py.error(f"Failed to allocate I/O for {engine_name} tensor {name}: {e}")
                raise

        self.logger_py.info(f"Allocated I/O for {engine_name} engine: "
                           f"{len(io_data['inputs'])} inputs, {len(io_data['outputs'])} outputs")
        return io_data

    def should_process_frame(self) -> bool:
        """Frame skip logic for target FPS"""
        self.frame_counter += 1
        should_process = (self.frame_counter % max(1, int(self.frame_skip_ratio))) == 0
        if not should_process:
            self.logger_py.debug(f"Frame {self.frame_counter} skipped to meet {self.target_fps} FPS")
        return should_process

    def preprocess_frame(self, frame):
        """Preprocess frame for both engines"""
        try:
            orig = frame.copy()
            h0, w0 = frame.shape[:2]
            img = cv2.resize(frame, (self.input_size, self.input_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))[None, ...]
            return img, orig, (w0, h0)
        except Exception as e:
            self.logger_py.error(f"Preprocessing failed: {e}")
            return None, frame, (frame.shape[1], frame.shape[0])

    def infer_boundary_engine(self, img):
        """Inference on boundary detection engine with async stream"""
        try:
            cuda.memcpy_htod_async(self.boundary_io['inputs'][0]["device"], img.ravel(), self.boundary_stream)
            for io in self.boundary_io['inputs']:
                self.boundary_context.set_tensor_address(io["name"], io["device"])
            for io in self.boundary_io['outputs']:
                self.boundary_context.set_tensor_address(io["name"], io["device"])
            self.boundary_context.execute_async_v3(self.boundary_stream.handle)
            outputs = []
            for out in self.boundary_io['outputs']:
                cuda.memcpy_dtoh_async(out["host"], out["device"], self.boundary_stream)
            self.boundary_stream.synchronize()
            outputs = [out["host"].copy() for out in self.boundary_io['outputs']]
            return outputs
        except Exception as e:
            self.logger_py.error(f"Boundary inference failed: {e}")
            return []

    def infer_semantic_engine(self, img):
        """Inference on semantic segmentation engine with async stream"""
        try:
            cuda.memcpy_htod_async(self.semantic_io['inputs'][0]["device"], img.ravel(), self.semantic_stream)
            for io in self.semantic_io['inputs']:
                self.semantic_context.set_tensor_address(io["name"], io["device"])
            for io in self.semantic_io['outputs']:
                self.semantic_context.set_tensor_address(io["name"], io["device"])
            self.semantic_context.execute_async_v3(self.semantic_stream.handle)
            outputs = []
            for out in self.semantic_io['outputs']:
                cuda.memcpy_dtoh_async(out["host"], out["device"], self.semantic_stream)
            self.semantic_stream.synchronize()
            outputs = [out["host"].copy() for out in self.semantic_io['outputs']]
            return outputs
        except Exception as e:
            self.logger_py.error(f"Semantic inference failed: {e}")
            return []

    def postprocess_boundary_masks(self, outputs, conf_threshold=0.25):
        """Postprocess boundary engine outputs"""
        if not outputs or len(outputs) < 2:
            self.logger_py.warning("No boundary outputs received")
            return np.array([]), np.array([]), ([], [], 0, [])
         # === DEBUG: Print actual output structure ===
        print(f"\n[BOUNDARY DEBUG]")
        print(f"  Number of outputs: {len(outputs)}")
        for i, out in enumerate(outputs):
            print(f"  Output[{i}]: shape={out.shape}, size={out.size}, dtype={out.dtype}")
        
        if len(outputs) >= 2:
            print(f"  Expected proto: 32×160×160 = {32*160*160} elements")
            print(f"  Expected det: 37×8400 = {37*8400} elements")
        # === END DEBUG ===
        
        if not outputs or len(outputs) < 2:
            self.logger_py.warning("No boundary outputs received")
            return np.array([]), np.array([]), ([], [], 0, [])
        
        try:
            seg = outputs[0].astype(np.float32)
            det = outputs[1].reshape(37, 8400)
            scores = det[4]
            print(f"[CONF DEBUG] Boundary scores: min={scores.min():.4f}, max={scores.max():.4f}, "
              f"mean={scores.mean():.4f}, >0.25: {np.sum(scores > 0.25)}, >0.1: {np.sum(scores > 0.1)}")
            keep = scores > conf_threshold

            if not np.any(keep):
                self.logger_py.debug("No boundary detections above threshold")
                return np.array([]), np.array([]), ([], [], 0, [])

            mc = det[5:37, keep].astype(np.float32)
            N = mc.shape[1]
            mc_dev = cuda.mem_alloc(mc.nbytes)
            cuda.memcpy_htod(mc_dev, mc)

            seg = outputs[0].astype(np.float32)
            proto_dev = cuda.mem_alloc(seg.nbytes)
            cuda.memcpy_htod(proto_dev, seg)

            C, H, W = 32, 160, 160
            Hn, Wn = self.input_size, self.input_size
            proto_res_dev = cuda.mem_alloc(C * Hn * Wn * 4)

            block = (16, 16, 1)
            grid = ((Wn + 15) // 16, (Hn + 15) // 16, C)
            resizeK(proto_dev, proto_res_dev, np.int32(C), np.int32(H), np.int32(W),
                    np.int32(Hn), np.int32(Wn), block=block, grid=grid)
            cuda.Context.synchronize()

            HW = Hn * Wn
            lin_dev = cuda.mem_alloc(N * HW * 4)
            alpha_ct = ctypes.c_float(1.0)
            beta_ct = ctypes.c_float(0.0)

            status = _libcublas.cublasSgemm_v2(
                self.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                N, HW, 32, ctypes.byref(alpha_ct),
                ctypes.c_void_p(int(mc_dev)), 32,
                ctypes.c_void_p(int(proto_res_dev)), 32,
                ctypes.byref(beta_ct),
                ctypes.c_void_p(int(lin_dev)), N
            )
            if status != 0:
                raise RuntimeError(f"cublasSgemm_v2 failed with {status}")

            bin_dev = cuda.mem_alloc(N * HW)
            block = (256, 1, 1)
            grid = ((HW + 255) // 256, N, 1)
            sigmK(lin_dev, bin_dev, np.int32(N), np.int32(HW), block=block, grid=grid)
            cuda.Context.synchronize()

            masks = np.empty((N, HW), dtype=np.uint8)
            cuda.memcpy_dtoh(masks, bin_dev)
            masks = masks.reshape(N, self.input_size, self.input_size)

            for d in (mc_dev, proto_dev, proto_res_dev, lin_dev, bin_dev):
                d.free()

            # NMS on boundaries to suppress duplicates/close ones
            boundary_masks, lane_areas, num_lanes, centroids = self._analyze_boundary_lanes(masks, self.input_size)
            
            # Apply ROI filter
            boundary_masks = self._apply_roi_mask(boundary_masks)
            lane_areas = self._apply_roi_mask(lane_areas)
            
            # Recalculate centroids for ROI-filtered lanes
            centroids = []
            for area in lane_areas:
                if np.any(area):
                    ys, xs = np.nonzero(area)
                    centroids.append((np.mean(xs), np.mean(ys)))
                else:
                    centroids.append((self.input_size/2, self.input_size/2))
            
            num_lanes = len(lane_areas)
            
            return masks, scores[keep], (boundary_masks, lane_areas, num_lanes, centroids)
        except Exception as e:
            self.logger_py.error(f"Boundary postprocessing failed: {e}")
            return np.array([]), np.array([]), ([], [], 0, [])

    def postprocess_semantic_masks(self, outputs, conf_threshold=0.25):
        """Postprocess semantic engine outputs"""
        if not outputs:
            self.logger_py.warning("No semantic outputs received")
            return [], []

        try:
            det = outputs[1].reshape(37, 8400)
            scores = det[4]
            keep = scores > conf_threshold

            if not np.any(keep):
                self.logger_py.debug("No semantic detections above threshold")
                return [], []

            mc = det[5:37, keep].astype(np.float32)
            N = mc.shape[1]
            mc_dev = cuda.mem_alloc(mc.nbytes)
            cuda.memcpy_htod(mc_dev, mc)

            seg = outputs[0].astype(np.float32)
            proto_dev = cuda.mem_alloc(seg.nbytes)
            cuda.memcpy_htod(proto_dev, seg)

            C, H, W = 32, 160, 160
            Hn, Wn = self.input_size, self.input_size
            proto_res_dev = cuda.mem_alloc(C * Hn * Wn * 4)

            block = (16, 16, 1)
            grid = ((Wn + 15) // 16, (Hn + 15) // 16, C)
            resizeK(proto_dev, proto_res_dev, np.int32(C), np.int32(H), np.int32(W),
                    np.int32(Hn), np.int32(Wn), block=block, grid=grid)
            cuda.Context.synchronize()

            HW = Hn * Wn
            lin_dev = cuda.mem_alloc(N * HW * 4)
            alpha_ct = ctypes.c_float(1.0)
            beta_ct = ctypes.c_float(0.0)

            status = _libcublas.cublasSgemm_v2(
                self.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                N, HW, 32, ctypes.byref(alpha_ct),
                ctypes.c_void_p(int(mc_dev)), 32,
                ctypes.c_void_p(int(proto_res_dev)), 32,
                ctypes.byref(beta_ct),
                ctypes.c_void_p(int(lin_dev)), N
            )
            if status != 0:
                raise RuntimeError(f"cublasSgemm_v2 failed with {status}")

            bin_dev = cuda.mem_alloc(N * HW)
            block = (256, 1, 1)
            grid = ((HW + 255) // 256, N, 1)
            sigmK(lin_dev, bin_dev, np.int32(N), np.int32(HW), block=block, grid=grid)
            cuda.Context.synchronize()

            masks = np.empty((N, HW), dtype=np.uint8)
            cuda.memcpy_dtoh(masks, bin_dev)
            masks = masks.reshape(N, self.input_size, self.input_size)

            for d in (mc_dev, proto_dev, proto_res_dev, lin_dev, bin_dev):
                d.free()

            # Strengthen segmentation with morph ops (erosion then dilation to remove noise and fill gaps)
            kernel = np.ones((3, 3), np.uint8)
            masks = [cv2.erode(mask, kernel, iterations=1) for mask in masks]
            masks = [cv2.dilate(mask, kernel, iterations=1) for mask in masks]  # Reduced to 1 iter for speed

            return masks, scores[keep]
        except Exception as e:
            self.logger_py.error(f"Semantic postprocessing failed: {e}")
            return [], []

    def _validate_lane_spatial(self, lane_mask):
        """Reject detections outside reasonable lane region"""
        H, W = lane_mask.shape
        
        # Check if extends too far vertically (shouldn't reach top 20% of image)
        vertical_extent = np.any(lane_mask[:int(H*0.2), :])
        if vertical_extent:
            return False
        
        # Check width - lane shouldn't span >60% of image width
        horizontal_pixels = np.sum(lane_mask, axis=0)
        max_consecutive = 0
        current = 0
        for px in horizontal_pixels:
            if px > 0:
                current += 1
                max_consecutive = max(max_consecutive, current)
            else:
                current = 0
        
        if max_consecutive > W * 0.6:
            return False
        
        return True
    
    def _analyze_boundary_lanes(self, masks, input_size):
        """Analyze boundary masks for multi lanes using CUDA kernels"""
        if masks.size == 0:
            self.logger_py.debug("No boundary masks to analyze")
            return [], [], 0, []

        N, H, W = masks.shape
        if N == 0:
            return [], [], 0, []

        masks_dev = cuda.mem_alloc(masks.nbytes)
        cuda.memcpy_htod(masks_dev, masks)

        minX_dev = cuda.mem_alloc(N * H * 4)
        maxX_dev = cuda.mem_alloc(N * H * 4)
        block = (256, 1, 1)
        grid = ((H + 255) // 256, 1, N)
        minMaxK(masks_dev, minX_dev, maxX_dev, np.int32(N), np.int32(H), np.int32(W), block=block, grid=grid)
        cuda.Context.synchronize()

        minX = np.empty((N, H), dtype=np.int32)
        maxX = np.empty((N, H), dtype=np.int32)
        cuda.memcpy_dtoh(minX, minX_dev)
        cuda.memcpy_dtoh(maxX, maxX_dev)

        boundary_info = []
        for n in range(N):
            valid = maxX[n] >= 0
            mean_x = np.mean(maxX[n][valid]) if np.any(valid) else -1
            if mean_x >= 0:
                boundary_info.append((mean_x, n))

        if len(boundary_info) < 2:
            for d in (masks_dev, minX_dev, maxX_dev):
                d.free()
            self.logger_py.debug(f"Insufficient boundaries for lanes: {len(boundary_info)}")
            return [], [], 0, []

        # NMS on boundaries (suppress if mean_x difference < threshold, e.g., 20 pixels)
        nms_threshold = 20
        kept = []
        boundary_info.sort(key=lambda x: x[0])
        i = 0
        while i < len(boundary_info):
            kept.append(boundary_info[i])
            j = i + 1
            while j < len(boundary_info) and boundary_info[j][0] - boundary_info[i][0] < nms_threshold:
                j += 1
            i = j

        sorted_indices = [idx for _, idx in kept]
        num_boundaries = len(sorted_indices)
        num_lanes = max(0, num_boundaries - 1)

        if num_lanes == 0:
            for d in (masks_dev, minX_dev, maxX_dev):
                d.free()
            return [], [], 0, []

        # Prepare maxLeft/minRight for lanes
        maxLeft = maxX[sorted_indices[:-1]]
        minRight = minX[sorted_indices[1:]]
        maxLeft_dev = cuda.mem_alloc(maxLeft.nbytes)
        minRight_dev = cuda.mem_alloc(minRight.nbytes)
        cuda.memcpy_htod(maxLeft_dev, maxLeft)
        cuda.memcpy_htod(minRight_dev, minRight)

        # Fill lanes
        areas_dev = cuda.mem_alloc(num_lanes * H * W)
        block = (16, 16, 1)
        grid = ((W + 15) // 16, (H + 15) // 16, num_lanes)
        fillK(maxLeft_dev, minRight_dev, areas_dev, np.int32(num_lanes), np.int32(H), np.int32(W), block=block, grid=grid)
        cuda.Context.synchronize()

        areas = np.empty((num_lanes, H, W), dtype=np.uint8)
        cuda.memcpy_dtoh(areas, areas_dev)

        boundary_masks = [masks[idx] > 0 for idx in sorted_indices]
        lane_areas = [areas[i] > 0 for i in range(num_lanes)]

        centroids = []
        for area in lane_areas:
            if np.any(area):
                ys, xs = np.nonzero(area)
                centroid_x = np.mean(xs)
                centroid_y = np.mean(ys)
                centroids.append((centroid_x, centroid_y))
            else:
                centroids.append((W / 2, H / 2))

        for d in (masks_dev, minX_dev, maxX_dev, maxLeft_dev, minRight_dev, areas_dev):
            d.free()

        self.logger_py.debug(f"Detected {num_lanes} lanes after NMS")
        lane_areas = [area for area in lane_areas if self._validate_lane_spatial(area)]
        return boundary_masks, lane_areas, num_lanes, centroids

    def initialize_kalman_filter(self, lane_id, centroid):
        """Initialize Kalman filter for a lane"""
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.x = np.array([centroid[0], centroid[1], 0, 0], dtype=np.float32)  # [x, y, vx, vy]
        kf.F = np.array([[1, 0, 1, 0],
                         [0, 1, 0, 1],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype=np.float32)  # State transition
        kf.H = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]], dtype=np.float32)  # Measurement
        kf.P *= 1000.0  # Initial uncertainty
        kf.R = np.array([[5, 0], [0, 5]], dtype=np.float32)  # Measurement noise
        kf.Q = np.eye(4) * 0.1  # Process noise
        self.kalman_filters[lane_id] = kf
        return kf

    def update_kalman_filters(self, centroids):
        """Update Kalman filters with new lane centroids"""
        updated_centroids = []
        for i, centroid in enumerate(centroids):
            lane_id = f"lane_{i}"
            if lane_id not in self.kalman_filters:
                self.kalman_filters[lane_id] = self.initialize_kalman_filter(lane_id, centroid)
            kf = self.kalman_filters[lane_id]
            kf.update(np.array(centroid, dtype=np.float32))
            kf.predict()
            updated_centroids.append((kf.x[0], kf.x[1]))
        return updated_centroids

    def analyze_confidence(self, boundary_result, semantic_masks, semantic_scores):
        """Analyze confidence for both engine outputs"""
        boundary_masks, lane_areas, num_lanes, centroids = boundary_result if len(boundary_result) == 4 else ([], [], 0, [])
        boundary_conf = 0.5
        if len(boundary_masks) > 0:
            edge_strength = 0
            for mask in boundary_masks:
                if np.any(mask):
                    grad_x = cv2.Sobel(mask.astype(np.uint8), cv2.CV_64F, 1, 0, ksize=3)
                    grad_y = cv2.Sobel(mask.astype(np.uint8), cv2.CV_64F, 0, 1, ksize=3)
                    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
                    edge_strength += np.mean(grad_mag[mask > 0]) if np.any(mask) else 0
            edge_strength = edge_strength / len(boundary_masks) / 100.0
            continuity = self._calculate_continuity(boundary_masks)
            parallelism = self._calculate_parallelism(boundary_masks)
            boundary_conf = np.mean([edge_strength, continuity, parallelism])
            boundary_conf = min(boundary_conf, 1.0)

        semantic_conf = 0.5
        if len(semantic_masks) > 0:
            total_pixels = self.input_size * self.input_size
            coverage = np.sum([np.sum(mask) for mask in semantic_masks]) / total_pixels
            coverage_score = self._score_coverage(coverage)
            consistency_score = self._calculate_semantic_consistency(semantic_masks)
            semantic_conf = np.mean([coverage_score, consistency_score, np.mean(semantic_scores)])
            semantic_conf = min(semantic_conf, 1.0)

        return boundary_conf, semantic_conf

    def _calculate_continuity(self, boundary_masks):
        """Calculate boundary continuity score"""
        if len(boundary_masks) == 0:
            return 0.0
        continuity_scores = []
        for mask in boundary_masks:
            if not np.any(mask):
                continuity_scores.append(0.0)
                continue
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continuity_scores.append(0.0)
                continue
            main_contour = max(contours, key=cv2.contourArea)
            contour_length = cv2.arcLength(main_contour, False)
            hull = cv2.convexHull(main_contour)
            hull_length = cv2.arcLength(hull, False)
            continuity = min(contour_length / hull_length, 1.0) if hull_length > 0 else 0.0
            continuity_scores.append(continuity)
        return np.mean(continuity_scores)

    def _calculate_parallelism(self, boundary_masks):
        """Calculate parallelism score for boundary pairs"""
        if len(boundary_masks) < 2:
            return 0.5
        angles = []
        for mask in boundary_masks:
            if not np.any(mask):
                continue
            lines = cv2.HoughLinesP(mask.astype(np.uint8), 1, np.pi / 180,
                                    threshold=50, minLineLength=100, maxLineGap=20)
            if lines is not None:
                line_angles = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                    line_angles.append(angle)
                if line_angles:
                    angles.append(np.median(line_angles))
        if len(angles) < 2:
            return 0.3
        # For multi-boundaries, average diffs between consecutive
        pair_diffs = []
        for j in range(len(angles) - 1):
            angle_diff = abs(angles[j] - angles[j + 1])
            angle_diff = min(angle_diff, 180 - angle_diff)
            pair_diffs.append(angle_diff)
        if not pair_diffs:
            return 0.3
        avg_diff = np.mean(pair_diffs)
        parallelism_score = max(0, 1 - avg_diff / 15)
        return parallelism_score

    def _score_coverage(self, coverage):
        """Score semantic coverage"""
        min_cov = self.confidence_params['semantic']['coverage_min']
        max_cov = self.confidence_params['semantic']['coverage_max']
        if min_cov <= coverage <= max_cov:
            return 1.0
        elif coverage < min_cov:
            return coverage / min_cov
        else:
            return max(0, 1 - (coverage - max_cov) / (1.0 - max_cov))

    def _calculate_semantic_consistency(self, semantic_masks):
        """Calculate semantic shape consistency"""
        if len(semantic_masks) == 0:
            return 0.0
        consistency_scores = []
        for mask in semantic_masks:
            if not np.any(mask):
                consistency_scores.append(0.0)
                continue
            mask_uint8 = mask.astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                consistency_scores.append(0.0)
                continue
            main_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(main_contour)
            y_coords = main_contour[:, 0, 1]
            upper_region_ratio = np.sum(y_coords < self.input_size // 2) / len(y_coords)
            rect = cv2.minAreaRect(main_contour)
            width, height = rect[1]
            aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 0
            geometric_penalty = 0.1 if upper_region_ratio > 0.3 or aspect_ratio < 1.5 else 1.0
            perimeter = cv2.arcLength(main_contour, True)
            compactness = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0.0
            consistency = min(compactness * 2, 1.0) * geometric_penalty
            consistency_scores.append(consistency)
        return np.mean(consistency_scores)

    def _apply_temporal_smoothing(self, current_boundary_masks, current_semantic_masks, current_lane_areas):
        """Apply EMA smoothing to reduce flickering"""
        if self.previous_boundary_masks is None:
            self.previous_boundary_masks = current_boundary_masks
            self.previous_semantic_masks = current_semantic_masks
            self.previous_lane_areas = current_lane_areas
            return current_boundary_masks, current_semantic_masks, current_lane_areas

        smoothed_boundary = []
        for curr, prev in zip(current_boundary_masks, self.previous_boundary_masks or current_boundary_masks):
            smoothed = (self.ema_alpha * curr + (1 - self.ema_alpha) * prev).astype(np.bool_)
            smoothed_boundary.append(smoothed)

        smoothed_semantic = []
        for curr, prev in zip(current_semantic_masks, self.previous_semantic_masks or current_semantic_masks):
            smoothed = (self.ema_alpha * curr + (1 - self.ema_alpha) * prev).astype(np.bool_)
            smoothed_semantic.append(smoothed)

        smoothed_lanes = []
        for curr, prev in zip(current_lane_areas, self.previous_lane_areas or current_lane_areas):
            smoothed = (self.ema_alpha * curr + (1 - self.ema_alpha) * prev).astype(np.bool_)
            smoothed_lanes.append(smoothed)

        self.previous_boundary_masks = smoothed_boundary
        self.previous_semantic_masks = smoothed_semantic
        self.previous_lane_areas = smoothed_lanes

        return smoothed_boundary, smoothed_semantic, smoothed_lanes

    def fusion_algorithm(self, boundary_result, semantic_masks, semantic_scores, boundary_conf, semantic_conf):
        """Core fusion algorithm"""
        boundary_masks, lane_areas, num_lanes, centroids = boundary_result if len(boundary_result) == 4 else ([], [], 0, [])
        if not lane_areas and not semantic_masks:
            self.logger_py.warning("No valid detections for fusion")
            return np.zeros((self.input_size, self.input_size), dtype=np.bool_), "none"

        conf_diff = abs(boundary_conf - semantic_conf)
        if boundary_conf > 0.7 and conf_diff > 0.3:
            fusion_method = "boundary_dominant"
            fused_result = self._boundary_dominant_fusion(boundary_masks, lane_areas, semantic_masks)
        elif semantic_conf > 0.7 and conf_diff > 0.3:
            fusion_method = "semantic_dominant"
            fused_result = self._semantic_dominant_fusion(semantic_masks, boundary_masks)
        else:
            fusion_method = "balanced"
            fused_result = self._balanced_fusion(boundary_masks, lane_areas, semantic_masks, boundary_conf, semantic_conf)
        return fused_result, fusion_method

    def _boundary_dominant_fusion(self, boundary_masks, lane_areas, semantic_masks):
        """Boundary-guided fusion"""
        if not lane_areas:
            return np.zeros((self.input_size, self.input_size), dtype=np.bool_)
        primary_area = np.logical_or.reduce(lane_areas)
        if semantic_masks:
            semantic_combined = np.logical_or.reduce(semantic_masks)
            constrained_semantic = np.logical_and(primary_area, semantic_combined)
            result = np.logical_or(primary_area, constrained_semantic)
        else:
            result = primary_area
        return result

    def _semantic_dominant_fusion(self, semantic_masks, boundary_masks):
        """Semantic-guided fusion"""
        if not semantic_masks:
            return np.zeros((self.input_size, self.input_size), dtype=np.bool_)
        semantic_combined = np.logical_or.reduce(semantic_masks)
        if boundary_masks:
            corridor = self._create_boundary_corridor(boundary_masks)
            if np.any(corridor):
                validated_semantic = np.logical_and(semantic_combined, corridor)
                result = np.logical_or(validated_semantic, semantic_combined)
            else:
                result = semantic_combined
        else:
            result = semantic_combined
        return result

    def _balanced_fusion(self, boundary_masks, lane_areas, semantic_masks, b_conf, s_conf):
        """Balanced fusion approach"""
        boundary_combined = np.logical_or.reduce(lane_areas) if lane_areas else np.zeros((self.input_size, self.input_size), dtype=np.bool_)
        semantic_combined = np.logical_or.reduce(semantic_masks) if semantic_masks else np.zeros((self.input_size, self.input_size), dtype=np.bool_)
        total_weight = b_conf + s_conf
        b_weight = b_conf / total_weight if total_weight > 0 else 0.5
        s_weight = s_conf / total_weight if total_weight > 0 else 0.5
        boundary_float = boundary_combined.astype(np.float32) * b_weight
        semantic_float = semantic_combined.astype(np.float32) * s_weight
        combined_confidence = boundary_float + semantic_float
        return combined_confidence > 0.4

    def _create_boundary_corridor(self, boundary_masks):
        """Create corridor around boundaries"""
        corridor = np.zeros((self.input_size, self.input_size), dtype=np.bool_)
        if not boundary_masks:
            return corridor
        for mask in boundary_masks:
            mask_dev = cuda.mem_alloc(mask.nbytes)
            cuda.memcpy_htod(mask_dev, mask.astype(np.uint8))
            dilated_dev = cuda.mem_alloc(self.input_size * self.input_size)
            dilateK(mask_dev, dilated_dev, np.int32(self.input_size), np.int32(self.input_size),
                    block=(16, 16, 1), grid=((self.input_size + 15) // 16, (self.input_size + 15) // 16, 1))
            cuda.Context.synchronize()
            dilated = np.empty((self.input_size, self.input_size), np.uint8)
            cuda.memcpy_dtoh(dilated, dilated_dev)
            corridor |= (dilated > 0)
            mask_dev.free()
            dilated_dev.free()
        return corridor

    def create_boundary_constraint_mask(self, boundary_masks, lane_areas):
        """Create constraint mask using GPU-based dilation"""
        constraint_mask = np.zeros((self.input_size, self.input_size), dtype=np.bool_)
        if len(boundary_masks) >= 2:
            # Dilate all boundaries to create corridors
            for mask in boundary_masks:
                mask_dev = cuda.mem_alloc(mask.nbytes)
                cuda.memcpy_htod(mask_dev, mask.astype(np.uint8))
                dilated_dev = cuda.mem_alloc(self.input_size * self.input_size)
                dilateK(mask_dev, dilated_dev, np.int32(self.input_size), np.int32(self.input_size),
                        block=(16, 16, 1), grid=((self.input_size + 15) // 16, (self.input_size + 15) // 16, 1))
                cuda.Context.synchronize()
                dilated = np.empty((self.input_size, self.input_size), np.uint8)
                cuda.memcpy_dtoh(dilated, dilated_dev)
                constraint_mask |= (dilated > 0)
                mask_dev.free()
                dilated_dev.free()
            # Fill gaps between boundaries if needed
        elif lane_areas:
            for area in lane_areas:
                area_dev = cuda.mem_alloc(area.nbytes)
                dilated_dev = cuda.mem_alloc(self.input_size * self.input_size)
                cuda.memcpy_htod(area_dev, area.astype(np.uint8))
                dilateK(area_dev, dilated_dev, np.int32(self.input_size), np.int32(self.input_size),
                        block=(16, 16, 1), grid=((self.input_size + 15) // 16, (self.input_size + 15) // 16, 1))
                cuda.Context.synchronize()
                dilated = np.empty((self.input_size, self.input_size), np.uint8)
                cuda.memcpy_dtoh(dilated, dilated_dev)
                constraint_mask |= (dilated > 0)
                area_dev.free()
                dilated_dev.free()
        else:
            constraint_mask[int(self.input_size * 0.4):, :] = True
        return constraint_mask

    def apply_semantic_constraints(self, semantic_masks, constraint_mask):
        """Apply boundary constraints to semantic results"""
        constrained_masks = []
        for mask in semantic_masks:
            constrained_mask = np.logical_and(mask > 0, constraint_mask)
            constrained_masks.append(constrained_mask)
        return constrained_masks

    def process_frame(self, frame):
        """Main frame processing with integrated track management"""
        if not self.should_process_frame():
            return None
        
        self.current_frame_number += 1
        total_start = time.perf_counter()
        
        img, orig, (w0, h0) = self.preprocess_frame(frame)
        if img is None:
            self.logger_py.warning("Skipping frame due to preprocessing failure")
            return None

        # === BOUNDARY PATH ===
        t1 = time.perf_counter()
        boundary_outputs = self.infer_boundary_engine(img)
        t2 = time.perf_counter()
        
        boundary_masks_raw, boundary_scores, boundary_result = self.postprocess_boundary_masks(
            boundary_outputs, conf_threshold=self.BOUNDARY_THRESHOLD
        )
        boundary_masks, lane_areas, num_lanes, centroids = boundary_result if len(boundary_result) == 4 else ([], [], 0, [])
        constraint_mask = self.create_boundary_constraint_mask(boundary_masks, lane_areas)
        
        # === SEMANTIC PATH ===
        t3 = time.perf_counter()
        semantic_outputs = self.infer_semantic_engine(img)
        t4 = time.perf_counter()
        
        semantic_masks_raw, semantic_scores = self.postprocess_semantic_masks(
            semantic_outputs, conf_threshold=self.SEMANTIC_THRESHOLD
        )
        semantic_masks = self.apply_semantic_constraints(semantic_masks_raw, constraint_mask)
        
        # === CONFIDENCE ANALYSIS ===
        t5 = time.perf_counter()
        boundary_conf, semantic_conf = self.analyze_confidence(
            (boundary_masks, lane_areas, num_lanes, centroids), semantic_masks, semantic_scores
        )
        
        # === TEMPORAL SMOOTHING (EMA on masks) ===
        boundary_masks, semantic_masks, lane_areas = self._apply_temporal_smoothing(
            boundary_masks, semantic_masks, lane_areas
        )
        
        # === TRACK MANAGEMENT ===
        # Associate current lane detections with existing tracks
        matched_pairs, unmatched_tracks, unmatched_detections = self.track_manager.associate_lanes(
            lane_areas, centroids, self.current_frame_number
        )
        
        # Update matched tracks with temporal averaging
        for track_id, lane_mask, centroid in matched_pairs:
            self.track_manager.update_track(track_id, lane_mask, centroid, self.current_frame_number)
        
        # Create new tracks for unmatched detections
        for lane_mask, centroid in unmatched_detections:
            if len(self.track_manager.tracks) < self.track_manager.max_tracks:
                self.track_manager.create_track(lane_mask, centroid, self.current_frame_number)
        
        # Prune dead tracks
        self.track_manager.prune_dead_tracks(self.current_frame_number, max_age=15)
        
        # === FUSION (for constraint visualization) ===
        centroids = self.update_kalman_filters(centroids)
        fused_lanes, fusion_method = self.fusion_algorithm(
            (boundary_masks, lane_areas, num_lanes, centroids), semantic_masks, semantic_scores,
            boundary_conf, semantic_conf
        )
        
        t6 = time.perf_counter()
        
        # === TIMING STATS ===
        self.timing_stats['boundary_inference'].append((t2 - t1) * 1000)
        self.timing_stats['semantic_inference'].append((t4 - t3) * 1000)
        self.timing_stats['fusion_processing'].append((t6 - t5) * 1000)
        self.timing_stats['total_processing'].append((t6 - total_start) * 1000)
        self.timing_stats['cuda_kernels'].append((t3 - t2 + t5 - t4) * 1000)
        
        result = FusionResult(
            fused_lanes=fused_lanes,
            boundary_masks=boundary_masks,
            semantic_masks=semantic_masks,
            confidence_map=constraint_mask.astype(np.float32) * max(boundary_conf, semantic_conf),
            boundary_confidence=boundary_conf,
            semantic_confidence=semantic_conf,
            fusion_method=fusion_method,
            processing_times={
                'boundary_inference': (t2 - t1) * 1000,
                'semantic_inference': (t4 - t3) * 1000,
                'fusion_processing': (t6 - t5) * 1000,
                'total': (t6 - total_start) * 1000
            },
            lane_centroids=centroids
        )
        
        self.previous_result = result
        return result

    def visualize_fusion_result(self, original_frame, fusion_result):
        """Visualize with tracked lanes showing persistent IDs and colors"""
        try:    
            vis_frame = cv2.resize(original_frame, (self.input_size, self.input_size))
            
            # Create separate overlay layers
            constraint_overlay = np.zeros_like(vis_frame)
            boundary_overlay = np.zeros_like(vis_frame)
            semantic_overlay = np.zeros_like(vis_frame)
            tracked_lanes_overlay = np.zeros_like(vis_frame)
            
            # === CONSTRAINT REGION (subtle blue background) ===
            if self.visualization_config['show_constraint']:
                constraint_mask = fusion_result.confidence_map > 0
                constraint_overlay[constraint_mask] = (100, 50, 0)  # Dark blue
                alpha_constraint = 0.15
            
            # === BOUNDARY CURVES (white/yellow/magenta lines) ===
            boundary_colors = [(255, 255, 255), (0, 255, 255), (255, 0, 255), (0, 128, 255)]  # White, Yellow, Magenta, Amber
            if self.visualization_config['show_boundary']:
                for i, mask in enumerate(fusion_result.boundary_masks):
                    if np.any(mask):
                        boundary_overlay[mask > 0] = boundary_colors[i % len(boundary_colors)]
                alpha_boundary = 0.7
            
            # === SEMANTIC MASKS (cyan fill - less important) ===
            if self.visualization_config['show_semantic']:
                for mask in fusion_result.semantic_masks:
                    if np.any(mask):
                        semantic_overlay[mask > 0] = (255, 255, 0)  # Cyan
                alpha_semantic = 0.25
            
            # === TRACKED LANES (colored with IDs - most important) ===
            if self.visualization_config['show_fused']:
                tracks = self.track_manager.get_sorted_tracks()
                for track in tracks:
                    if np.any(track.mask):
                        tracked_lanes_overlay[track.mask] = track.color
                        
                        # Draw centroid with ID label
                        cx, cy = int(track.centroid[0]), int(track.centroid[1])
                        
                        # White circle background
                        cv2.circle(tracked_lanes_overlay, (cx, cy), 8, (255, 255, 255), -1)
                        # Colored circle center
                        cv2.circle(tracked_lanes_overlay, (cx, cy), 6, track.color, -1)
                        
                        # ID label with background
                        label = f"ID:{track.track_id}"
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        label_x, label_y = cx + 12, cy + 5
                        
                        # Semi-transparent background for label
                        cv2.rectangle(tracked_lanes_overlay, 
                                    (label_x - 2, label_y - label_size[1] - 2),
                                    (label_x + label_size[0] + 2, label_y + 2),
                                    (0, 0, 0), -1)
                        
                        cv2.putText(tracked_lanes_overlay, label, (label_x, label_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                alpha_tracked = 0.6
            
            # === LAYER COMPOSITION ===
            result_frame = vis_frame.copy()
            
            if self.visualization_config['show_constraint']:
                result_frame = cv2.addWeighted(result_frame, 1.0, constraint_overlay, alpha_constraint, 0)
            
            if self.visualization_config['show_semantic']:
                result_frame = cv2.addWeighted(result_frame, 1.0, semantic_overlay, alpha_semantic, 0)
            
            if self.visualization_config['show_fused']:
                result_frame = cv2.addWeighted(result_frame, 1.0, tracked_lanes_overlay, alpha_tracked, 0)
            
            if self.visualization_config['show_boundary']:
                result_frame = cv2.addWeighted(result_frame, 1.0, boundary_overlay, alpha_boundary, 0)
            
            # === ROI OUTLINE  ===
            if self.visualization_config.get('show_roi', True) and self.roi_mask is not None:
                roi_overlay = np.zeros_like(vis_frame)
                contours, _ = cv2.findContours(self.roi_mask.astype(np.uint8),
                                            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(roi_overlay, contours, -1, (255, 0, 255), 2)  # magenta
                result_frame = cv2.addWeighted(result_frame, 1.0, roi_overlay, 0.5, 0)
            
            # === INFO OVERLAY ===
            tracks = self.track_manager.get_sorted_tracks()
            constraint_area = np.sum(fusion_result.confidence_map > 0) / (self.input_size * self.input_size) * 100
            
            # Stats box background
            cv2.rectangle(result_frame, (5, 5), (400, 180), (0, 0, 0), -1)
            cv2.rectangle(result_frame, (5, 5), (400, 180), (100, 100, 100), 2)
            
            text_info = [
                f"Frame: {self.current_frame_number}",
                f"Tracked Lanes: {len(tracks)}",
                f"Fusion: {fusion_result.fusion_method}",
                f"Boundary: {fusion_result.boundary_confidence:.3f}",
                f"Semantic: {fusion_result.semantic_confidence:.3f}",
                f"Processing: {fusion_result.processing_times['total']:.1f}ms",
                f"FPS: {1000.0 / fusion_result.processing_times['total']:.1f}"
            ]
            
            for i, text in enumerate(text_info):
                cv2.putText(result_frame, text, (10, 25 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            
            # === LEGEND ===
            legend_y = self.input_size - 110
            cv2.rectangle(result_frame, (5, legend_y - 15), (200, self.input_size - 5), (0, 0, 0), -1)
            cv2.rectangle(result_frame, (5, legend_y - 15), (200, self.input_size - 5), (100, 100, 100), 2)
            
            safe_track_color = (int(_colour_lut[0][0]), int(_colour_lut[0][1]), int(_colour_lut[0][2]))
            
            legend_items = [
                ("Tracked (f)", safe_track_color, self.visualization_config['show_fused']),
                ("Boundary (b)", (255, 255, 255), self.visualization_config['show_boundary']),
                ("Semantic (s)", (255, 255, 0), self.visualization_config['show_semantic']),
                ("Region (c)", (100, 50, 0), self.visualization_config['show_constraint'])
            ]
            
            for i, item in enumerate(legend_items):
                if len(item) != 3:
                    self.logger_py.error(f"Invalid legend item {i}: {item}")
                    continue
                    
                name, color, visible = item
                
                # Validate color is 3-element
                if not isinstance(color, (tuple, list)) or len(color) != 3:
                    self.logger_py.error(f"Invalid color for {name}: {color}")
                    color = (128, 128, 128)  # Fallback gray
                
                y_pos = legend_y + i * 22
                status = "ON" if visible else "OFF"
                status_color = (0, 255, 0) if visible else (100, 100, 100)
                
                cv2.rectangle(result_frame, (10, y_pos - 12), (28, y_pos), color, -1)
                cv2.rectangle(result_frame, (10, y_pos - 12), (28, y_pos), (200, 200, 200), 1)
                
                cv2.putText(result_frame, name, (33, y_pos - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                cv2.putText(result_frame, status, (150, y_pos - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1)
            
            return result_frame
        except Exception as e:
            self.logger_py.error(f"Visualization failed: {e}", exc_info=True)
            return cv2.resize(original_frame, (self.input_size, self.input_size))
            
    

    def toggle_visualization(self, key):
        """Toggle visualization layers based on key press"""
        key_map = {
            ord('c'): 'show_constraint',
            ord('b'): 'show_boundary',
            ord('s'): 'show_semantic',
            ord('f'): 'show_fused'
        }
        if key in key_map:
            self.visualization_config[key_map[key]] = not self.visualization_config[key_map[key]]
            self.logger_py.info(f"Toggled {key_map[key]} to {self.visualization_config[key_map[key]]}")

    def get_performance_stats(self):
        """Get comprehensive performance statistics"""
        stats = {}
        for key, times in self.timing_stats.items():
            if times:
                stats[key] = {
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'min': np.min(times),
                    'max': np.max(times),
                    'samples': len(times)
                }
        memory_info = cuda.mem_get_info()
        stats['memory'] = {
            'free': memory_info[0] / (1024 ** 2),
            'total': memory_info[1] / (1024 ** 2),
            'used': (memory_info[1] - memory_info[0]) / (1024 ** 2)
        }
        return stats

def setup_camera(source):
    """Camera setup function"""
    try:
        is_video_file = False
        if source.isdigit():
            cap = cv2.VideoCapture(int(source))
        elif source.startswith(('http://', 'https://', 'rtsp://')):
            cap = cv2.VideoCapture(source)
        else:
            cap = cv2.VideoCapture(source)
            is_video_file = True
        if not cap.isOpened():
            raise ValueError(f"Cannot open source: {source}")
        return cap, is_video_file, None
    except Exception as e:
        logger.error(f"Camera setup failed: {e}")
        raise

def load_config(config_path):
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load config {config_path}: {e}. Using default config.")
        return {
            'input_size': 640,
            'target_fps': 18,
            'confidence_params': {
                'boundary': {
                    'edge_strength_threshold': 50,
                    'continuity_threshold': 0.7,
                    'parallelism_tolerance': 15
                },
                'semantic': {
                    'coverage_min': 0.05,
                    'coverage_max': 0.4,
                    'consistency_threshold': 0.8
                }
            },
            'dilation_kernel_size': 15,
            'visualization': {
                'show_boundary': True,
                'show_semantic': True,
                'show_fused': True,
                'show_constraint': True,
                'base_alpha': 0.4
            }
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dual-Engine Lane Fusion Test')
    parser.add_argument('--boundary-engine', required=True, help='Path to boundary detection TensorRT engine')
    parser.add_argument('--semantic-engine', required=True, help='Path to semantic segmentation TensorRT engine')
    parser.add_argument('--source', default='0', help='Video source (file, camera index, or URL)')
    parser.add_argument('--input-size', type=int, default=640, help='Input size for processing')
    parser.add_argument('--display-size', type=int, default=640, help='Display size for visualization')
    parser.add_argument('--target-fps', type=int, default=18, help='Target processing FPS')
    parser.add_argument('--config', default='config.json', help='Path to configuration JSON file')
    args = parser.parse_args()

    config = load_config(args.config)
    config['input_size'] = args.input_size
    config['target_fps'] = args.target_fps

    print("Initializing dual-engine fusion system...")
    fusion_system = DualEngineTensorRT(args.boundary_engine, args.semantic_engine, config)
    cap, is_video, _ = setup_camera(args.source)

    print("Starting fusion processing...")
    print(f"Target FPS: {args.target_fps} (processing every {fusion_system.frame_skip_ratio:.1f} frames)")

    frame_count = 0
    processed_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if is_video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break
            frame_count += 1
            fusion_result = fusion_system.process_frame(frame)
            if fusion_result is not None:
                processed_count += 1
                vis_frame = fusion_system.visualize_fusion_result(frame, fusion_result)
                cv2.imshow('Dual-Engine Lane Fusion', vis_frame)
                if processed_count % 30 == 0:
                    stats = fusion_system.get_performance_stats()
                    print(f"\n=== Performance Stats (Frame {frame_count}, Processed {processed_count}) ===")
                    for key, stat in stats.items():
                        if key != 'memory':
                            print(f"{key}: {stat['mean']:.1f}±{stat['std']:.1f}ms "
                                  f"[{stat['min']:.1f}-{stat['max']:.1f}] ({stat['samples']} samples)")
                    print(f"Memory: {stats['memory']['used']:.1f}/{stats['memory']['total']:.1f} MB")
                    if fusion_result:
                        print(f"Latest: Method={fusion_result.fusion_method}, "
                              f"B_conf={fusion_result.boundary_confidence:.3f}, "
                              f"S_conf={fusion_result.semantic_confidence:.3f}, "
                              f"Lanes={len(fusion_result.lane_centroids)}")
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            fusion_system.toggle_visualization(key)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
    finally:
        print("\n=== Final Statistics ===")
        print(f"Total frames: {frame_count}")
        print(f"Processed frames: {processed_count}")
        print(f"Processing ratio: {processed_count / frame_count:.2f}" if frame_count > 0 else "Processing ratio: N/A")
        final_stats = fusion_system.get_performance_stats()
        for key, stat in final_stats.items():
            if key != 'memory':
                print(f"{key}: {stat['mean']:.1f}±{stat['std']:.1f}ms")
        print(f"Memory: {final_stats['memory']['used']:.1f}/{final_stats['memory']['total']:.1f} MB")
        cap.release()
        cv2.destroyAllWindows()