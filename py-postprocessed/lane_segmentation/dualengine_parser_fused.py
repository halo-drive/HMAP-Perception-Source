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
from typing import List, Dict, Tuple, Optional
import logging
import json
import os
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN

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

# Lane color palette
LANE_COLORS = [
    (255, 50, 50),    # Red
    (50, 255, 50),    # Green  
    (50, 50, 255),    # Blue
    (255, 255, 50),   # Yellow
    (255, 50, 255),   # Magenta
    (50, 255, 255),   # Cyan
    (255, 128, 50),   # Orange
    (128, 50, 255),   # Purple
]

# Load CUDA kernels
ptx_file = "postprocess_kernels2.ptx"
if not os.path.exists(ptx_file):
    raise FileNotFoundError(f"CUDA PTX file {ptx_file} not found.")
try:
    postproc_mod = cuda.module_from_file(ptx_file)
    resizeK = postproc_mod.get_function("resizePrototypesKernel")
    sigmK = postproc_mod.get_function("sigmoidThresholdKernel")
except Exception as e:
    logger.error(f"Failed to load CUDA kernels from {ptx_file}: {e}")
    raise

@dataclass(eq=False)
class LaneDetection:
    """Single lane detection instance from semantic engine"""
    mask: np.ndarray
    centroid: Tuple[float, float]
    score: float
    source: str
    geometry: np.ndarray
    bbox: Tuple[int, int, int, int]

@dataclass
class BoundaryCurve:
    """Represents a tracked boundary curve (NOT a lane)"""
    curve_id: int
    polynomial: np.ndarray  # [a, b, c] for x = ay² + by + c
    centroid_x: float  # Mean x position for left-right ordering
    kalman_filter: KalmanFilter  # Tracks polynomial coefficients
    confidence: float
    age: int
    last_updated: int

@dataclass
class LaneTrack:
    """Persistent lane track (semantic instances only)"""
    track_id: int
    color: Tuple[int, int, int]
    mask: np.ndarray
    centroid: Tuple[float, float]
    kalman_filter: KalmanFilter
    geometry: np.ndarray
    confidence: float
    age: int
    last_seen: int
    source: str

class BoundaryTracker:
    """Tracks boundary curves over time - provides geometric constraints"""
    
    def __init__(self, input_size=640):
        self.curves: Dict[int, BoundaryCurve] = {}
        self.next_curve_id = 0
        self.input_size = input_size
        self.logger = logging.getLogger(__name__)
        self.max_curves = 8  # Typically 4-6 boundaries for 2-3 lanes
    
    def process_boundary_detections(self, boundary_masks: np.ndarray, 
                                   scores: np.ndarray, 
                                   current_frame: int) -> List[BoundaryCurve]:
        """
        Main processing: cluster boundary points → fit curves → track temporally
        
        Args:
            boundary_masks: [N, H, W] binary masks (20-30 boundary point detections)
            scores: [N] confidence scores
            current_frame: Current frame number
            
        Returns:
            List of tracked boundary curves (4-6 curves typically)
        """
        if boundary_masks.shape[0] == 0:
            return list(self.curves.values())
        
        # Step 1: Cluster boundary point detections into curves
        clustered_curves = self._cluster_boundaries(boundary_masks, scores)
        
        # Step 2: Fit polynomials to each clustered curve
        for curve_data in clustered_curves:
            curve_data['polynomial'] = self._fit_robust_polynomial(curve_data['points'])
        
        # Step 3: Associate with existing tracked curves and update
        self._associate_and_update_curves(clustered_curves, current_frame)
        
        # Step 4: Prune old curves
        self._prune_dead_curves(current_frame, max_age=10)
        
        return self.get_sorted_curves()
    
    def _cluster_boundaries(self, masks: np.ndarray, scores: np.ndarray) -> List[Dict]:
        """
        Cluster 20-30 boundary point detections into 4-6 boundary curves
        
        Uses spatial clustering (DBSCAN) on boundary centroids
        """
        N, H, W = masks.shape
        
        # Extract centroid for each boundary point detection
        centroids = []
        valid_indices = []
        for i in range(N):
            mask = masks[i]
            if np.sum(mask) < 50:  # Minimum pixels
                continue
            ys, xs = np.where(mask > 0)
            if len(xs) == 0:
                continue
            centroid_x = np.mean(xs)
            centroid_y = np.mean(ys)
            centroids.append([centroid_x, centroid_y])
            valid_indices.append(i)
        
        if len(centroids) < 2:
            return []
        
        centroids = np.array(centroids)
        
        # DBSCAN clustering: group nearby boundary points into curves
        # eps = typical lane width / 2, min_samples = 2
        clustering = DBSCAN(eps=80, min_samples=2).fit(centroids)
        labels = clustering.labels_
        
        # Group masks by cluster
        clustered_curves = []
        unique_labels = set(labels)
        unique_labels.discard(-1)  # Remove noise label
        
        for label in unique_labels:
            cluster_indices = [valid_indices[i] for i, l in enumerate(labels) if l == label]
            
            if len(cluster_indices) < 2:
                continue
            
            # Combine all masks in this cluster
            combined_mask = np.logical_or.reduce([masks[idx] > 0 for idx in cluster_indices])
            
            # Extract points from combined mask
            ys, xs = np.where(combined_mask)
            if len(xs) < 10:
                continue
            
            points = np.column_stack((xs, ys))
            
            # Calculate cluster statistics
            mean_score = np.mean([scores[idx] for idx in cluster_indices])
            centroid_x = np.mean(xs)
            
            clustered_curves.append({
                'points': points,
                'centroid_x': centroid_x,
                'confidence': float(mean_score),
                'polynomial': None  # Will be filled by polynomial fitting
            })
        
        # Sort left-to-right
        clustered_curves.sort(key=lambda c: c['centroid_x'])
        
        self.logger.debug(f"Clustered {N} boundary points into {len(clustered_curves)} curves")
        return clustered_curves
    
    def _fit_robust_polynomial(self, points: np.ndarray) -> np.ndarray:
        """
        Fit polynomial to boundary curve points
        
        Args:
            points: [M, 2] array of (x, y) coordinates
            
        Returns:
            [a, b, c] coefficients for x = ay² + by + c
        """
        if len(points) < 10:
            return np.array([0.0, 1.0, 320.0])
        
        xs, ys = points[:, 0], points[:, 1]
        
        try:
            with np.errstate(all='ignore'):
                # Sample points if too many (reduce computation)
                if len(points) > 500:
                    indices = np.random.choice(len(points), 500, replace=False)
                    xs, ys = xs[indices], ys[indices]
                
                # Use unique y values to avoid collinearity
                ys_unique, indices = np.unique(ys, return_index=True)
                xs_unique = xs[indices]
                
                if len(ys_unique) < 3:
                    return np.array([0.0, 1.0, np.mean(xs)])
                
                # Quadratic fit: x = ay² + by + c
                coeffs = np.polyfit(ys_unique, xs_unique, 2)
                
                # Validate
                if np.any(np.isnan(coeffs)) or np.any(np.isinf(coeffs)) or np.any(np.abs(coeffs) > 1e6):
                    return np.array([0.0, 1.0, np.mean(xs)])
                
                return coeffs
        except:
            return np.array([0.0, 1.0, np.mean(xs)])
    
    def _associate_and_update_curves(self, new_curves: List[Dict], current_frame: int):
        """
        Associate new clustered curves with existing tracked curves
        Update Kalman filters for temporal stability
        """
        if not self.curves:
            # No existing curves, create all as new
            for curve_data in new_curves:
                self._create_curve(curve_data, current_frame)
            return
        
        if not new_curves:
            return
        
        # Build cost matrix based on x-position similarity
        existing_curves = list(self.curves.values())
        cost_matrix = np.zeros((len(existing_curves), len(new_curves)))
        
        for i, existing in enumerate(existing_curves):
            for j, new_data in enumerate(new_curves):
                # Distance between curve x-positions
                dist = abs(existing.centroid_x - new_data['centroid_x'])
                cost_matrix[i, j] = dist / self.input_size  # Normalize
        
        # Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Update matched curves
        matched_existing = []
        matched_new = []
        
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < 0.3:  # Threshold for valid match
                existing_curve_id = existing_curves[i].curve_id
                self._update_curve(existing_curve_id, new_curves[j], current_frame)
                matched_existing.append(i)
                matched_new.append(j)
        
        # Create new curves for unmatched detections
        for j, new_data in enumerate(new_curves):
            if j not in matched_new and len(self.curves) < self.max_curves:
                self._create_curve(new_data, current_frame)
    
    def _create_curve(self, curve_data: Dict, current_frame: int):
        """Create new tracked boundary curve"""
        curve_id = self.next_curve_id
        self.next_curve_id += 1
        
        # Initialize Kalman filter for polynomial coefficients
        # State: [a, b, c] (polynomial coefficients)
        kf = KalmanFilter(dim_x=3, dim_z=3)
        kf.x = curve_data['polynomial'].copy()
        kf.F = np.eye(3)  # Coefficients change slowly
        kf.H = np.eye(3)
        kf.P *= 100.0
        kf.R = np.eye(3) * 2.0  # Measurement noise
        kf.Q = np.eye(3) * 0.5  # Process noise
        
        curve = BoundaryCurve(
            curve_id=curve_id,
            polynomial=curve_data['polynomial'].copy(),
            centroid_x=curve_data['centroid_x'],
            kalman_filter=kf,
            confidence=curve_data['confidence'],
            age=0,
            last_updated=current_frame
        )
        
        self.curves[curve_id] = curve
        self.logger.debug(f"Created boundary curve {curve_id} at x={curve_data['centroid_x']:.1f}")
    
    def _update_curve(self, curve_id: int, new_data: Dict, current_frame: int):
        """Update existing curve with new measurement"""
        if curve_id not in self.curves:
            return
        
        curve = self.curves[curve_id]
        
        # Update Kalman filter with new polynomial measurement
        curve.kalman_filter.update(new_data['polynomial'])
        curve.kalman_filter.predict()
        
        # Update curve properties with filtered values
        curve.polynomial = curve.kalman_filter.x.copy()
        curve.centroid_x = new_data['centroid_x']
        curve.confidence = 0.7 * curve.confidence + 0.3 * new_data['confidence']
        curve.age += 1
        curve.last_updated = current_frame
    
    def _prune_dead_curves(self, current_frame: int, max_age: int = 10):
        """Remove curves that haven't been updated recently"""
        dead_curves = [
            cid for cid, curve in self.curves.items()
            if current_frame - curve.last_updated > max_age
        ]
        
        for cid in dead_curves:
            del self.curves[cid]
            self.logger.debug(f"Pruned boundary curve {cid}")
    
    def get_sorted_curves(self) -> List[BoundaryCurve]:
        """Get curves sorted left-to-right"""
        curves = list(self.curves.values())
        curves.sort(key=lambda c: c.centroid_x)
        return curves
    
    def validate_semantic_lane(self, lane_centroid: Tuple[float, float]) -> bool:
        """
        Validate if semantic lane centroid lies between boundary curves
        
        Args:
            lane_centroid: (x, y) centroid of semantic lane detection
            
        Returns:
            True if lane is valid (lies between boundaries), False otherwise
        """
        curves = self.get_sorted_curves()
        
        if len(curves) < 2:
            return True  # No boundary constraint available
        
        cx, cy = lane_centroid
        
        # Evaluate all boundary polynomials at this y-coordinate
        boundary_x_positions = []
        for curve in curves:
            a, b, c = curve.polynomial
            x_at_y = a * cy**2 + b * cy + c
            boundary_x_positions.append(x_at_y)
        
        boundary_x_positions.sort()
        
        # Check if centroid lies between any pair of consecutive boundaries
        for i in range(len(boundary_x_positions) - 1):
            left_x = boundary_x_positions[i]
            right_x = boundary_x_positions[i + 1]
            
            # Allow small margin
            margin = 20
            if left_x - margin < cx < right_x + margin:
                return True
        
        return False

class TrackManager:
    """Manages lane tracks (semantic instances only)"""
    
    def __init__(self, max_tracks=6):
        self.tracks: Dict[int, LaneTrack] = {}
        self.next_track_id = 0
        self.color_palette = deque(LANE_COLORS.copy())
        self.used_colors: Dict[int, Tuple] = {}
        self.max_tracks = max_tracks
        self.logger = logging.getLogger(__name__)
    
    def associate_detections(self, detections: List[LaneDetection], current_frame: int) -> Tuple[List, List, List]:
        """Match detections to existing tracks"""
        if not self.tracks:
            return [], [], detections
        
        if not detections:
            return [], list(self.tracks.keys()), []
        
        track_ids = list(self.tracks.keys())
        cost_matrix = np.zeros((len(track_ids), len(detections)))
        
        for i, track_id in enumerate(track_ids):
            track = self.tracks[track_id]
            for j, detection in enumerate(detections):
                # Centroid distance (primary metric)
                dist = np.linalg.norm(np.array(track.centroid) - np.array(detection.centroid))
                dist_normalized = dist / 640.0
                
                # IoU (secondary, only if close)
                if dist_normalized < 0.4:
                    iou = self._compute_iou_fast(track.mask, detection.mask, track.bbox, detection.bbox)
                else:
                    iou = 0.0
                
                cost = dist_normalized * 0.7 + (1 - iou) * 0.3
                cost_matrix[i, j] = cost
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matched_pairs = []
        matched_track_indices = []
        matched_detection_indices = []
        
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < 0.5:
                matched_pairs.append((track_ids[i], detections[j]))
                matched_track_indices.append(i)
                matched_detection_indices.append(j)
        
        unmatched_tracks = [track_ids[i] for i in range(len(track_ids)) if i not in matched_track_indices]
        unmatched_detections = [detections[j] for j in range(len(detections)) if j not in matched_detection_indices]
        
        return matched_pairs, unmatched_tracks, unmatched_detections
    
    def _compute_iou_fast(self, mask1, mask2, bbox1, bbox2):
        """Optimized IoU with bbox pre-check"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        if x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min:
            return 0.0
        
        intersection = np.logical_and(mask1, mask2).sum()
        if intersection == 0:
            return 0.0
        
        union = np.logical_or(mask1, mask2).sum()
        return float(intersection) / float(union)
    
    def create_track(self, detection: LaneDetection, current_frame: int) -> LaneTrack:
        """Create new track from detection"""
        track_id = self.next_track_id
        self.next_track_id += 1
        
        color = self.color_palette.popleft() if self.color_palette else LANE_COLORS[track_id % len(LANE_COLORS)]
        self.used_colors[track_id] = color
        
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.x = np.array([detection.centroid[0], detection.centroid[1], 0, 0], dtype=np.float32)
        kf.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        kf.P *= 1000.0
        kf.R = np.array([[5, 0], [0, 5]], dtype=np.float32)
        kf.Q = np.eye(4) * 0.1
        
        track = LaneTrack(
            track_id=track_id,
            color=color,
            mask=detection.mask.copy(),
            centroid=detection.centroid,
            kalman_filter=kf,
            geometry=detection.geometry.copy(),
            confidence=detection.score,
            age=0,
            last_seen=current_frame,
            source=detection.source
        )
        
        self.tracks[track_id] = track
        return track
    
    def update_track(self, track_id: int, detection: LaneDetection, current_frame: int):
        """Update existing track"""
        if track_id not in self.tracks:
            return
        
        track = self.tracks[track_id]
        track.kalman_filter.update(np.array(detection.centroid, dtype=np.float32))
        track.kalman_filter.predict()
        
        track.mask = detection.mask.copy()
        track.centroid = (track.kalman_filter.x[0], track.kalman_filter.x[1])
        track.geometry = detection.geometry.copy()
        track.confidence = 0.7 * track.confidence + 0.3 * detection.score
        track.age += 1
        track.last_seen = current_frame
    
    def prune_dead_tracks(self, current_frame: int, max_age: int = 5):
        """Remove old tracks"""
        dead_tracks = [tid for tid, t in self.tracks.items() if current_frame - t.last_seen > max_age]
        
        for tid in dead_tracks:
            track = self.tracks[tid]
            if track.color in self.used_colors.values():
                self.color_palette.append(track.color)
            del self.tracks[tid]
            del self.used_colors[tid]
    
    def get_sorted_tracks(self) -> List[LaneTrack]:
        """Get tracks sorted left-to-right"""
        tracks = list(self.tracks.values())
        tracks.sort(key=lambda t: t.centroid[0])
        return tracks

class DualEngineTensorRT:
    """Complete dual-path lane tracking system"""

    def __init__(self, boundary_engine_path, semantic_engine_path, config):
        self.logger_py = logging.getLogger(__name__)
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        # Initialize cuBLAS
        self.cublas_handle = ctypes.c_void_p()
        status = _libcublas.cublasCreate_v2(ctypes.byref(self.cublas_handle))
        if status != 0:
            raise RuntimeError(f"cublasCreate failed with {status}")

        # Load BOTH engines
        self.boundary_engine = self._load_engine(boundary_engine_path)
        self.semantic_engine = self._load_engine(semantic_engine_path)

        # Create execution contexts
        self.boundary_context = self.boundary_engine.create_execution_context()
        self.semantic_context = self.semantic_engine.create_execution_context()

        # Configuration
        self.input_size = config.get('input_size', 640)
        self.conf_threshold_boundary = config.get('conf_threshold_boundary', 0.3)
        self.conf_threshold_semantic = config.get('conf_threshold_semantic', 0.45)
        self.target_fps = config.get('target_fps', 25)
        self.frame_skip_interval = max(1, int(30 / self.target_fps))

        # Allocate I/O for BOTH engines
        self.boundary_io = self._allocate_io(self.boundary_engine, "boundary")
        self.semantic_io = self._allocate_io(self.semantic_engine, "semantic")

        # Tracking systems
        self.boundary_tracker = BoundaryTracker(input_size=self.input_size)
        self.track_manager = TrackManager(max_tracks=6)
        self.current_frame_number = 0
        self.frame_counter = 0

        # Performance tracking
        self.timing_stats = {
            'boundary_inference': deque(maxlen=30),
            'boundary_processing': deque(maxlen=30),
            'semantic_inference': deque(maxlen=30),
            'semantic_processing': deque(maxlen=30),
            'tracking': deque(maxlen=30),
            'total_processing': deque(maxlen=30)
        }

        self.logger_py.info("Dual-engine system initialized successfully")

    def _load_engine(self, engine_path):
        """Load TensorRT engine"""
        with open(engine_path, "rb") as f:
            engine = self.runtime.deserialize_cuda_engine(f.read())
            if engine is None:
                raise RuntimeError(f"Failed to deserialize {engine_path}")
            return engine

    def _allocate_io(self, engine, engine_name):
        """Allocate I/O tensors"""
        io_data = {'inputs': [], 'outputs': []}
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            shape = engine.get_tensor_shape(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            size = trt.volume(shape)

            host_mem = cuda.pagelocked_empty(size, dtype)
            dev_mem = cuda.mem_alloc(host_mem.nbytes)
            
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

        self.logger_py.info(f"Allocated I/O for {engine_name} engine")
        return io_data

    def should_process_frame(self) -> bool:
        """Frame skip logic"""
        self.frame_counter += 1
        return (self.frame_counter % self.frame_skip_interval) == 0

    def preprocess_frame(self, frame):
        """Preprocess frame"""
        orig = frame.copy()
        h0, w0 = frame.shape[:2]
        img = cv2.resize(frame, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[None, ...]
        return img, orig, (w0, h0)

    def infer_boundary_engine(self, img):
        """Boundary inference"""
        cuda.memcpy_htod(self.boundary_io['inputs'][0]["device"], img.ravel())
        for io in self.boundary_io['inputs']:
            self.boundary_context.set_tensor_address(io["name"], io["device"])
        for io in self.boundary_io['outputs']:
            self.boundary_context.set_tensor_address(io["name"], io["device"])
        self.boundary_context.execute_async_v3(0)
        
        outputs = []
        for out in self.boundary_io['outputs']:
            cuda.memcpy_dtoh(out["host"], out["device"])
            outputs.append(out["host"].copy())
        return outputs

    def infer_semantic_engine(self, img):
        """Semantic inference"""
        cuda.memcpy_htod(self.semantic_io['inputs'][0]["device"], img.ravel())
        for io in self.semantic_io['inputs']:
            self.semantic_context.set_tensor_address(io["name"], io["device"])
        for io in self.semantic_io['outputs']:
            self.semantic_context.set_tensor_address(io["name"], io["device"])
        self.semantic_context.execute_async_v3(0)
        
        outputs = []
        for out in self.semantic_io['outputs']:
            cuda.memcpy_dtoh(out["host"], out["device"])
            outputs.append(out["host"].copy())
        return outputs

    def postprocess_boundary_masks(self, outputs):
        """Postprocess boundary outputs → raw masks for clustering"""
        if not outputs:
            return np.array([]), np.array([])

        det = outputs[1].reshape(37, 8400)
        scores = det[4]
        keep = scores > self.conf_threshold_boundary

        if not np.any(keep):
            return np.array([]), np.array([])

        mc = det[5:37, keep].astype(np.float32)
        N = mc.shape[1]

        # GPU processing (standard YOLO seg postprocessing)
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

        _libcublas.cublasSgemm_v2(
            self.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
            N, HW, 32, ctypes.byref(alpha_ct),
            ctypes.c_void_p(int(mc_dev)), 32,
            ctypes.c_void_p(int(proto_res_dev)), 32,
            ctypes.byref(beta_ct),
            ctypes.c_void_p(int(lin_dev)), N
        )

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

        return masks, scores[keep]

    def postprocess_semantic_masks(self, outputs) -> List[LaneDetection]:
        """Postprocess semantic outputs → LaneDetection instances"""
        if not outputs:
            return []

        det = outputs[1].reshape(37, 8400)
        scores = det[4]
        keep = scores > self.conf_threshold_semantic

        if not np.any(keep):
            return []

        mc = det[5:37, keep].astype(np.float32)
        N = mc.shape[1]

        # GPU processing (same as boundary)
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

        _libcublas.cublasSgemm_v2(
            self.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
            N, HW, 32, ctypes.byref(alpha_ct),
            ctypes.c_void_p(int(mc_dev)), 32,
            ctypes.c_void_p(int(proto_res_dev)), 32,
            ctypes.byref(beta_ct),
            ctypes.c_void_p(int(lin_dev)), N
        )

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

        # Convert to LaneDetection instances
        detections = []
        for i in range(N):
            mask = masks[i] > 0
            score = float(scores[keep][i])
            
            if np.sum(mask) < 500:
                continue
            
            centroid = self._compute_centroid(mask)
            bbox = self._compute_bbox(mask)
            geometry = self._fit_polynomial(mask)
            
            # Strict geometric validation
            if not self._is_valid_lane(mask, centroid, bbox, score):
                continue
            
            detections.append(LaneDetection(
                mask=mask,
                centroid=centroid,
                score=score,
                source="semantic",
                geometry=geometry,
                bbox=bbox
            ))
        
        return detections

    def _compute_centroid(self, mask):
        """Compute centroid"""
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return (0.0, 0.0)
        return (float(np.mean(xs)), float(np.mean(ys)))

    def _compute_bbox(self, mask):
        """Compute bbox"""
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return (0, 0, 0, 0)
        return (int(np.min(xs)), int(np.min(ys)), int(np.max(xs)), int(np.max(ys)))

    def _fit_polynomial(self, mask):
        """Fit polynomial"""
        ys, xs = np.where(mask)
        if len(xs) < 10:
            return np.array([0.0, 1.0, 320.0])
        
        try:
            with np.errstate(all='ignore'):
                ys_unique, indices = np.unique(ys, return_index=True)
                xs_unique = xs[indices]
                
                if len(ys_unique) < 3:
                    if len(ys_unique) >= 2:
                        coeffs = np.polyfit(ys_unique, xs_unique, 1)
                        return np.array([0.0, coeffs[0], coeffs[1]])
                    return np.array([0.0, 1.0, np.mean(xs)])
                
                coeffs = np.polyfit(ys_unique, xs_unique, 2)
                
                if np.any(np.isnan(coeffs)) or np.any(np.isinf(coeffs)) or np.any(np.abs(coeffs) > 1e6):
                    return np.array([0.0, 1.0, np.mean(xs)])
                
                return coeffs
        except:
            return np.array([0.0, 1.0, np.mean(xs)])

    def _is_valid_lane(self, mask, centroid, bbox, score):
        """Strict validation"""
        pixel_count = np.sum(mask)
        
        if pixel_count < 500 or pixel_count > 120000:
            return False
        
        x1, y1, x2, y2 = bbox
        width, height = x2 - x1, y2 - y1
        
        if width == 0 or height == 0:
            return False
        
        # Position: lower 70%
        if y1 < self.input_size * 0.3:
            return False
        
        # Aspect ratio
        if height / width < 1.5:
            return False
        
        # Width bounds
        if width < 40 or width > 250:
            return False
        
        # Score threshold
        if score < 0.45:
            return False
        
        # Centroid check
        if centroid[1] < self.input_size * 0.4:
            return False
        
        return True

    def process_frame(self, frame):
        """Main processing pipeline - DUAL PATH"""
        
        if not self.should_process_frame():
            return self._render_cached_tracks(frame)
        
        self.current_frame_number += 1
        total_start = time.perf_counter()
        
        img, orig, (w0, h0) = self.preprocess_frame(frame)
        if img is None:
            return None
        
        # === PATH 1: BOUNDARY PROCESSING (Geometric Constraints) ===
        t1 = time.perf_counter()
        boundary_outputs = self.infer_boundary_engine(img)
        t2 = time.perf_counter()
        
        boundary_masks, boundary_scores = self.postprocess_boundary_masks(boundary_outputs)
        
        # Process boundaries: cluster → fit curves → track temporally
        boundary_curves = self.boundary_tracker.process_boundary_detections(
            boundary_masks, boundary_scores, self.current_frame_number
        )
        t3 = time.perf_counter()
        
        # === PATH 2: SEMANTIC PROCESSING (Lane Instances) ===
        t4 = time.perf_counter()
        semantic_outputs = self.infer_semantic_engine(img)
        t5 = time.perf_counter()
        
        semantic_detections = self.postprocess_semantic_masks(semantic_outputs)
        
        # CRITICAL: Validate semantic lanes against boundary constraints
        valid_detections = [
            det for det in semantic_detections
            if self.boundary_tracker.validate_semantic_lane(det.centroid)
        ]
        t6 = time.perf_counter()
        
        # === PATH 3: TRACK MANAGEMENT (Validated Semantic Lanes Only) ===
        t7 = time.perf_counter()
        matched_pairs, unmatched_tracks, unmatched_detections = \
            self.track_manager.associate_detections(valid_detections, self.current_frame_number)
        
        for track_id, detection in matched_pairs:
            self.track_manager.update_track(track_id, detection, self.current_frame_number)
        
        for detection in unmatched_detections:
            if detection.score > 0.5:
                self.track_manager.create_track(detection, self.current_frame_number)
        
        self.track_manager.prune_dead_tracks(self.current_frame_number, max_age=5)
        t8 = time.perf_counter()
        
        # Update timing stats
        self.timing_stats['boundary_inference'].append((t2 - t1) * 1000)
        self.timing_stats['boundary_processing'].append((t3 - t2) * 1000)
        self.timing_stats['semantic_inference'].append((t5 - t4) * 1000)
        self.timing_stats['semantic_processing'].append((t6 - t5) * 1000)
        self.timing_stats['tracking'].append((t8 - t7) * 1000)
        self.timing_stats['total_processing'].append((t8 - total_start) * 1000)
        
        return self._render_tracks(frame, boundary_curves)

    def _render_cached_tracks(self, frame):
        """Render without reprocessing"""
        boundary_curves = self.boundary_tracker.get_sorted_curves()
        return self._render_tracks(frame, boundary_curves)

    def _render_tracks(self, frame, boundary_curves):
        """Render with boundary curves visualization"""
        vis_frame = cv2.resize(frame, (self.input_size, self.input_size))
        overlay = np.zeros_like(vis_frame)
        
        # Draw boundary curves (white lines)
        for curve in boundary_curves:
            a, b, c = curve.polynomial
            pts = []
            for y in range(0, self.input_size, 10):
                x = a * y**2 + b * y + c
                x = int(np.clip(x, 0, self.input_size - 1))
                pts.append((x, y))
            if len(pts) > 1:
                pts = np.array(pts, dtype=np.int32)
                cv2.polylines(overlay, [pts], False, (255, 255, 255), 2)
        
        # Draw tracked lanes
        tracks = self.track_manager.get_sorted_tracks()
        for track in tracks:
            overlay[track.mask] = track.color
            cx, cy = int(track.centroid[0]), int(track.centroid[1])
            cv2.circle(overlay, (cx, cy), 5, (255, 255, 255), -1)
            cv2.putText(overlay, f"ID:{track.track_id}", (cx + 10, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        result = cv2.addWeighted(vis_frame, 1.0, overlay, 0.5, 0)
        
        # Info overlay
        cv2.putText(result, f"Lanes: {len(tracks)} | Boundaries: {len(boundary_curves)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(result, f"Frame: {self.current_frame_number}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if self.timing_stats['total_processing']:
            avg_time = np.mean(list(self.timing_stats['total_processing']))
            fps = 1000.0 / avg_time if avg_time > 0 else 0
            cv2.putText(result, f"Processing: {avg_time:.1f}ms ({fps:.1f} FPS)", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return result

    def get_performance_stats(self):
        """Get performance statistics"""
        stats = {}
        for key, times in self.timing_stats.items():
            if times:
                stats[key] = {
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'min': np.min(times),
                    'max': np.max(times)
                }
        memory_info = cuda.mem_get_info()
        stats['memory'] = {
            'free': memory_info[0] / (1024 ** 2),
            'total': memory_info[1] / (1024 ** 2),
            'used': (memory_info[1] - memory_info[0]) / (1024 ** 2)
        }
        return stats

def setup_camera(source):
    """Setup camera/video source"""
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

def load_config(config_path):
    """Load configuration"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except:
        return {
            'input_size': 640,
            'target_fps': 25,
            'conf_threshold_boundary': 0.3,
            'conf_threshold_semantic': 0.45
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dual-Path Lane Tracking')
    parser.add_argument('--boundary-engine', required=True)
    parser.add_argument('--semantic-engine', required=True)
    parser.add_argument('--source', default='0')
    parser.add_argument('--input-size', type=int, default=640)
    parser.add_argument('--target-fps', type=int, default=25)
    parser.add_argument('--config', default='config.json')
    args = parser.parse_args()

    config = load_config(args.config)
    config['input_size'] = args.input_size
    config['target_fps'] = args.target_fps

    print("Initializing dual-path lane tracking system...")
    fusion_system = DualEngineTensorRT(args.boundary_engine, args.semantic_engine, config)
    cap, is_video, _ = setup_camera(args.source)

    print(f"Starting at {args.target_fps} FPS target...")

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
            result_frame = fusion_system.process_frame(frame)
            
            if result_frame is not None:
                processed_count += 1
                cv2.imshow('Dual-Path Lane Tracking', result_frame)
                
                if processed_count % 30 == 0:
                    stats = fusion_system.get_performance_stats()
                    print(f"\n=== Frame {frame_count} (Processed {processed_count}) ===")
                    for key, stat in stats.items():
                        if key != 'memory':
                            print(f"{key}: {stat['mean']:.1f}±{stat['std']:.1f}ms")
                    print(f"Lanes: {len(fusion_system.track_manager.tracks)}, "
                          f"Boundaries: {len(fusion_system.boundary_tracker.curves)}")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        print(f"\nProcessed {processed_count}/{frame_count} frames")
        cap.release()
        cv2.destroyAllWindows()