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
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict
import logging

# Import existing components
fps_hist = deque(maxlen=120)
log_tick = time.perf_counter()

# Load cuBLAS via ctypes (existing code)
_libcublas = ctypes.cdll.LoadLibrary("libcublas.so")
_libcublas.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
_libcublas.cublasCreate_v2.restype  = ctypes.c_int
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

CUBLAS_OP_N = 0
CUBLAS_OP_T = 1

# Load CUDA kernels (existing code)
postproc_mod = cuda.module_from_file("postprocess_kernels.ptx")
resizeK  = postproc_mod.get_function("resizePrototypesKernel")
sigmK    = postproc_mod.get_function("sigmoidThresholdKernel")
laneK    = postproc_mod.get_function("laneFillKernel")
orReduceK = postproc_mod.get_function("orReduceMasks")
rowMinMaxK  = postproc_mod.get_function("rowMinMaxKernel")
buildLaneK  = postproc_mod.get_function("buildLaneMasksKernel")
colourizeK = postproc_mod.get_function("colourizeMasksKernel")

# Color LUT (existing code)
_colour_lut = np.array([
    (  0,255,255),  # yellow
    (255,  0,255),  # magenta
    (  0,255,  0),  # green
    (255,255,  0),  # cyan
    (255,128,  0),  # orange
    (  0,128,255),  # amber-blue
], dtype=np.uint8)

lut_dev = cuda.mem_alloc(_colour_lut.nbytes)
cuda.memcpy_htod(lut_dev, _colour_lut)
max_colours = _colour_lut.shape[0]

@dataclass
class FusionResult:
    """Container for fusion algorithm results"""
    fused_lanes: np.ndarray
    boundary_masks: List[np.ndarray]
    semantic_masks: List[np.ndarray] 
    confidence_map: np.ndarray
    boundary_confidence: float
    semantic_confidence: float
    fusion_method: str
    processing_times: Dict[str, float]

class DualEngineTensorRT:
    """Modified TensorRT inference class for dual-engine operation"""
    
    def __init__(self, boundary_engine_path, semantic_engine_path):
        # Initialize logging first
        logging.basicConfig(level=logging.INFO)
        self.logger_py = logging.getLogger(__name__)
        
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        # Initialize cuBLAS handle
        self.cublas_handle = ctypes.c_void_p()
        status = _libcublas.cublasCreate_v2(ctypes.byref(self.cublas_handle))
        if status != 0:
            raise RuntimeError(f"cublasCreate failed with {status}")
        
        # Load both engines
        self.boundary_engine = self._load_engine(boundary_engine_path)
        self.semantic_engine = self._load_engine(semantic_engine_path)
        
        # Create execution contexts
        self.boundary_context = self.boundary_engine.create_execution_context()
        self.semantic_context = self.semantic_engine.create_execution_context()
        
        # Allocate I/O for both engines
        self.boundary_io = self._allocate_io(self.boundary_engine, "boundary")
        self.semantic_io = self._allocate_io(self.semantic_engine, "semantic")
        
        # Fusion system parameters
        self.confidence_params = {
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
        }
        
        # Temporal tracking
        self.previous_result = None
        self.frame_counter = 0
        self.target_fps = 18
        self.frame_skip_ratio = 30 / self.target_fps
        
        # Performance tracking
        self.timing_stats = {
            'boundary_inference': deque(maxlen=30),
            'semantic_inference': deque(maxlen=30),
            'fusion_processing': deque(maxlen=30),
            'total_processing': deque(maxlen=30)
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger_py = logging.getLogger(__name__)
        
    def _load_engine(self, engine_path):
        """Load TensorRT engine from file"""
        with open(engine_path, "rb") as f:
            return self.runtime.deserialize_cuda_engine(f.read())
            
    def _allocate_io(self, engine, engine_name):
        """Allocate I/O tensors for an engine"""
        io_data = {'inputs': [], 'outputs': [], 'bindings': []}
        
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            shape = engine.get_tensor_shape(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            size = trt.volume(shape)
            
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
                
        self.logger_py.info(f"Allocated I/O for {engine_name} engine: "
                           f"{len(io_data['inputs'])} inputs, {len(io_data['outputs'])} outputs")
        return io_data
        
    def should_process_frame(self) -> bool:
        """Frame skip logic for target FPS"""
        self.frame_counter += 1
        # Process every Nth frame based on skip ratio
        return (self.frame_counter % max(1, int(self.frame_skip_ratio))) == 0
        
    def preprocess_frame(self, frame, input_size=640):
        """Preprocess frame for both engines (identical preprocessing)"""
        orig = frame.copy()
        h0, w0 = frame.shape[:2]
        img = cv2.resize(frame, (input_size, input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.transpose(img, (2,0,1))[None, ...]
        return img, orig, (w0, h0)
        
    def infer_boundary_engine(self, img):
        """Inference on boundary detection engine"""
        cuda.memcpy_htod(self.boundary_io['inputs'][0]["device"], img.ravel())
        
        # Set tensor addresses
        for io in self.boundary_io['inputs']:
            self.boundary_context.set_tensor_address(io["name"], io["device"])
        for io in self.boundary_io['outputs']:
            self.boundary_context.set_tensor_address(io["name"], io["device"])
            
        # Execute inference
        self.boundary_context.execute_async_v3(0)
        
        # Copy results back
        outputs = []
        for out in self.boundary_io['outputs']:
            cuda.memcpy_dtoh(out["host"], out["device"])
            outputs.append(out["host"].copy())
            
        return outputs
        
    def infer_semantic_engine(self, img):
        """Inference on semantic segmentation engine"""
        cuda.memcpy_htod(self.semantic_io['inputs'][0]["device"], img.ravel())
        
        # Set tensor addresses  
        for io in self.semantic_io['inputs']:
            self.semantic_context.set_tensor_address(io["name"], io["device"])
        for io in self.semantic_io['outputs']:
            self.semantic_context.set_tensor_address(io["name"], io["device"])
            
        # Execute inference
        self.semantic_context.execute_async_v3(0)
        
        # Copy results back
        outputs = []
        for out in self.semantic_io['outputs']:
            cuda.memcpy_dtoh(out["host"], out["device"])
            outputs.append(out["host"].copy())
            
        return outputs
        
    def postprocess_boundary_masks(self, outputs, conf_threshold=0.25, input_size=640):
        """Postprocess boundary engine outputs using existing boundary logic"""
        # Use existing boundary postprocessing logic
        det = outputs[1].reshape(37, 8400)
        scores = det[4]
        keep = scores > conf_threshold
        
        if not np.any(keep):
            return [], [], []
            
        # Process masks using existing pipeline
        mc = det[5:37, keep].astype(np.float32)
        N = mc.shape[1]
        
        # GPU-accelerated mask generation (existing code)
        mc_dev = cuda.mem_alloc(mc.nbytes)
        cuda.memcpy_htod(mc_dev, mc)
        
        seg = outputs[0].astype(np.float32)
        proto_dev = cuda.mem_alloc(seg.nbytes)
        cuda.memcpy_htod(proto_dev, seg)
        
        C, H, W = 32, 160, 160
        Hn, Wn = input_size, input_size
        proto_res_dev = cuda.mem_alloc(C*Hn*Wn*4)
        
        block = (16,16,1)
        grid = ((Wn+15)//16, (Hn+15)//16, C)
        resizeK(proto_dev, proto_res_dev,
                np.int32(C), np.int32(H), np.int32(W),
                np.int32(Hn), np.int32(Wn),
                block=block, grid=grid)
        
        # GEMM operation
        HW = Hn*Wn
        lin_dev = cuda.mem_alloc(N*HW*4)
        alpha_ct = ctypes.c_float(1.0)
        beta_ct = ctypes.c_float(0.0)
        
        status = _libcublas.cublasSgemm_v2(
            self.cublas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            N, HW, 32,
            ctypes.byref(alpha_ct),
            ctypes.c_void_p(int(mc_dev)), 32,
            ctypes.c_void_p(int(proto_res_dev)), 32,
            ctypes.byref(beta_ct),
            ctypes.c_void_p(int(lin_dev)), N
        )
        
        if status != 0:
            raise RuntimeError(f"cublasSgemm_v2 failed with {status}")
        
        # Sigmoid + threshold
        bin_dev = cuda.mem_alloc(N*HW)
        block = (256,1,1)
        grid = ((HW+255)//256, N, 1)
        sigmK(lin_dev, bin_dev, np.int32(N), np.int32(HW), block=block, grid=grid)
        
        masks = np.empty((N, HW), dtype=np.uint8)
        cuda.memcpy_dtoh(masks, bin_dev)
        masks = masks.reshape(N, input_size, input_size)
        
        # Cleanup GPU memory
        for d in (mc_dev, proto_dev, proto_res_dev, lin_dev, bin_dev):
            d.free()
            
        # Analyze lanes using existing boundary logic
        boundary_masks, lane_areas, num_lanes = self._analyze_boundary_lanes(masks, input_size)
        
        return masks, scores[keep], (boundary_masks, lane_areas, num_lanes)
        
    def postprocess_semantic_masks(self, outputs, conf_threshold=0.25, input_size=640):
        """Postprocess semantic engine outputs - identical to boundary processing"""
        # Note: Semantic engine has same output format, different interpretation
        det = outputs[1].reshape(37, 8400)
        scores = det[4]
        keep = scores > conf_threshold
        
        if not np.any(keep):
            return [], []
            
        # Same mask generation process as boundary
        mc = det[5:37, keep].astype(np.float32)
        N = mc.shape[1]
        
        # GPU processing (same as boundary)
        mc_dev = cuda.mem_alloc(mc.nbytes)
        cuda.memcpy_htod(mc_dev, mc)
        
        seg = outputs[0].astype(np.float32)
        proto_dev = cuda.mem_alloc(seg.nbytes)
        cuda.memcpy_htod(proto_dev, seg)
        
        C, H, W = 32, 160, 160
        Hn, Wn = input_size, input_size
        proto_res_dev = cuda.mem_alloc(C*Hn*Wn*4)
        
        block = (16,16,1)
        grid = ((Wn+15)//16, (Hn+15)//16, C)
        resizeK(proto_dev, proto_res_dev,
                np.int32(C), np.int32(H), np.int32(W),
                np.int32(Hn), np.int32(Wn),
                block=block, grid=grid)
        
        HW = Hn*Wn
        lin_dev = cuda.mem_alloc(N*HW*4)
        alpha_ct = ctypes.c_float(1.0)
        beta_ct = ctypes.c_float(0.0)
        
        status = _libcublas.cublasSgemm_v2(
            self.cublas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            N, HW, 32,
            ctypes.byref(alpha_ct),
            ctypes.c_void_p(int(mc_dev)), 32,
            ctypes.c_void_p(int(proto_res_dev)), 32,
            ctypes.byref(beta_ct),
            ctypes.c_void_p(int(lin_dev)), N
        )
        
        if status != 0:
            raise RuntimeError(f"cublasSgemm_v2 failed with {status}")
        
        bin_dev = cuda.mem_alloc(N*HW)
        block = (256,1,1)
        grid = ((HW+255)//256, N, 1)
        sigmK(lin_dev, bin_dev, np.int32(N), np.int32(HW), block=block, grid=grid)
        
        masks = np.empty((N, HW), dtype=np.uint8)
        cuda.memcpy_dtoh(masks, bin_dev)
        masks = masks.reshape(N, input_size, input_size)
        
        # Cleanup
        for d in (mc_dev, proto_dev, proto_res_dev, lin_dev, bin_dev):
            d.free()
            
        return masks, scores[keep]
        
    def _analyze_boundary_lanes(self, masks, input_size=640):
        """Lane analysis for boundary detection (existing logic)"""
        if masks.size == 0:
            return [], [], 0
            
        H = W = input_size
        N = masks.shape[0]
        
        # OR-reduce masks
        masks_dev = cuda.mem_alloc(masks.nbytes)
        cuda.memcpy_htod(masks_dev, masks)
        comb_dev = cuda.mem_alloc(H * W)
        orReduceK(masks_dev, comb_dev,
                  np.int32(N), np.int32(H), np.int32(W),
                  block=(16,16,1), grid=((W+15)//16, (H+15)//16, 1))
        
        # Row min/max analysis
        left_dev = cuda.mem_alloc(H * 4)
        right_dev = cuda.mem_alloc(H * 4)
        rowMinMaxK(comb_dev, left_dev, right_dev,
                   np.int32(H), np.int32(W),
                   block=(256,1,1), grid=((H+255)//256,1,1))
        
        leftX = np.empty(H, dtype=np.int32)
        rightX = np.empty(H, dtype=np.int32)
        cuda.memcpy_dtoh(leftX, left_dev)
        cuda.memcpy_dtoh(rightX, right_dev)
        
        if not np.any(rightX < W):
            for d in (masks_dev, comb_dev, left_dev, right_dev):
                d.free()
            return [], [], 0
        
        # Build lane masks
        leftB_dev = cuda.mem_alloc(H * W)
        rightB_dev = cuda.mem_alloc(H * W)
        area_dev = cuda.mem_alloc(H * W)
        buildLaneK(left_dev, right_dev, leftB_dev, rightB_dev, area_dev,
                   np.int32(H), np.int32(W),
                   block=(16,16,1), grid=((W+15)//16, (H+15)//16, 1))
        
        leftB = np.empty((H,W), np.uint8)
        rightB = np.empty((H,W), np.uint8)  
        area = np.empty((H,W), np.uint8)
        cuda.memcpy_dtoh(leftB, leftB_dev)
        cuda.memcpy_dtoh(rightB, rightB_dev)
        cuda.memcpy_dtoh(area, area_dev)
        
        # Cleanup
        for d in (masks_dev, comb_dev, left_dev, right_dev, leftB_dev, rightB_dev, area_dev):
            d.free()
            
        boundary_masks = [(leftB > 0), (rightB > 0)]
        lane_areas = [(area > 0)]
        return boundary_masks, lane_areas, 1
        
    def analyze_confidence(self, boundary_result, semantic_masks, semantic_scores):
        """Analyze confidence for both engine outputs"""
        if len(boundary_result) != 3:
            # Handle empty boundary result case
            boundary_masks, lane_areas, num_lanes = [], [], 0
        else:
            boundary_masks, lane_areas, num_lanes = boundary_result
        
        # Boundary confidence analysis
        boundary_conf = 0.5  # Default neutral
        if len(boundary_masks) > 0:
            # Calculate edge strength
            edge_strength = 0
            for mask in boundary_masks:
                if np.any(mask):
                    grad_x = cv2.Sobel(mask.astype(np.uint8), cv2.CV_64F, 1, 0, ksize=3)
                    grad_y = cv2.Sobel(mask.astype(np.uint8), cv2.CV_64F, 0, 1, ksize=3)
                    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
                    edge_strength += np.mean(grad_mag[mask > 0]) if np.any(mask) else 0
                    
            edge_strength = edge_strength / len(boundary_masks) / 100.0  # Normalize
            
            # Calculate continuity
            continuity = self._calculate_continuity(boundary_masks)
            
            # Calculate parallelism
            parallelism = self._calculate_parallelism(boundary_masks)
            
            boundary_conf = np.mean([edge_strength, continuity, parallelism])
            boundary_conf = min(boundary_conf, 1.0)
            
        # Semantic confidence analysis  
        semantic_conf = 0.5  # Default neutral
        if len(semantic_masks) > 0:
            # Coverage analysis
            total_pixels = 640 * 640
            coverage = np.sum([np.sum(mask) for mask in semantic_masks]) / total_pixels
            coverage_score = self._score_coverage(coverage)
            
            # Shape consistency
            consistency_score = self._calculate_semantic_consistency(semantic_masks)
            
            # Score combination
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
                
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
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
                
            lines = cv2.HoughLinesP(
                mask.astype(np.uint8), 1, np.pi/180, 
                threshold=50, minLineLength=100, maxLineGap=20
            )
            
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
            
        angle_diff = abs(angles[0] - angles[1])
        angle_diff = min(angle_diff, 180 - angle_diff)
        parallelism_score = max(0, 1 - angle_diff / 15)  # 15 degree tolerance
        return parallelism_score
        
    def _score_coverage(self, coverage):
        """Score semantic coverage (10-40% is optimal)"""
        if 0.1 <= coverage <= 0.4:
            return 1.0
        elif coverage < 0.1:
            return coverage / 0.1
        else:
            return max(0, 1 - (coverage - 0.4) / 0.6)
            
    def _calculate_semantic_consistency(self, semantic_masks):
        """Calculate semantic shape consistency with geometric constraints"""
        if len(semantic_masks) == 0:
            return 0.0
            
        consistency_scores = []
        for mask in semantic_masks:
            if not np.any(mask):
                consistency_scores.append(0.0)
                continue
                
            # Geometric plausibility analysis
            mask_uint8 = mask.astype(np.uint8)
            contours, _ = cv2.findContours(
                mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                consistency_scores.append(0.0)
                continue
                
            main_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(main_contour)
            
            # Reject areas extending into upper image regions (vegetation detection)
            y_coords = main_contour[:, 0, 1]
            upper_region_ratio = np.sum(y_coords < 320) / len(y_coords)  # Upper half penalty
            
            # Road surface geometric constraints
            rect = cv2.minAreaRect(main_contour)
            width, height = rect[1]
            aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 0
            
            # Penalize vertical structures and vegetation-like patterns
            if upper_region_ratio > 0.3 or aspect_ratio < 1.5:  # Too vertical or too much in upper region
                geometric_penalty = 0.1
            else:
                geometric_penalty = 1.0
                
            # Shape compactness analysis
            perimeter = cv2.arcLength(main_contour, True)
            if perimeter > 0:
                compactness = 4 * np.pi * area / (perimeter**2)
                consistency = min(compactness * 2, 1.0) * geometric_penalty
            else:
                consistency = 0.0
                
            consistency_scores.append(consistency)
            
        return np.mean(consistency_scores)
        
    def fusion_algorithm(self, boundary_result, semantic_masks, semantic_scores, 
                        boundary_conf, semantic_conf):
        """Core fusion algorithm"""
        boundary_masks, lane_areas, num_lanes = boundary_result
        
        # Determine fusion strategy
        conf_diff = abs(boundary_conf - semantic_conf)
        
        if boundary_conf > 0.7 and conf_diff > 0.3:
            fusion_method = "boundary_dominant"
            fused_result = self._boundary_dominant_fusion(boundary_masks, lane_areas, semantic_masks)
        elif semantic_conf > 0.7 and conf_diff > 0.3:
            fusion_method = "semantic_dominant" 
            fused_result = self._semantic_dominant_fusion(semantic_masks, boundary_masks)
        else:
            fusion_method = "balanced"
            fused_result = self._balanced_fusion(boundary_masks, lane_areas, semantic_masks,
                                               boundary_conf, semantic_conf)
            
        return fused_result, fusion_method
        
    def _boundary_dominant_fusion(self, boundary_masks, lane_areas, semantic_masks):
        """Boundary-guided fusion"""
        if len(lane_areas) == 0:
            return np.zeros((640, 640), dtype=np.bool_)
            
        primary_area = lane_areas[0]
        
        if len(semantic_masks) > 0:
            semantic_combined = np.zeros((640, 640), dtype=np.bool_)
            for mask in semantic_masks:
                semantic_combined = np.logical_or(semantic_combined, mask > 0)
                
            # Constrain semantic to boundary area
            constrained_semantic = np.logical_and(primary_area, semantic_combined)
            result = np.logical_or(primary_area, constrained_semantic)
        else:
            result = primary_area
            
    def create_boundary_constraint_mask(self, boundary_masks, lane_areas, input_size=640):
        """Create constraint mask for semantic processing from boundary detection results"""
        constraint_mask = np.zeros((input_size, input_size), dtype=np.bool_)
        
        if len(boundary_masks) >= 2:
            # Use detected boundary pairs to create lane corridor
            left_boundary = boundary_masks[0]
            right_boundary = boundary_masks[1]
            
            # Dilate boundaries to create search corridor
            kernel_boundary = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            left_dilated = cv2.dilate(left_boundary.astype(np.uint8), kernel_boundary)
            right_dilated = cv2.dilate(right_boundary.astype(np.uint8), kernel_boundary)
            
            # Create corridor between dilated boundaries
            corridor = np.logical_or(left_dilated > 0, right_dilated > 0)
            
            # Additional corridor fill between boundaries
            for y in range(input_size):
                left_points = np.where(left_dilated[y, :] > 0)[0]
                right_points = np.where(right_dilated[y, :] > 0)[0]
                
                if len(left_points) > 0 and len(right_points) > 0:
                    left_x = np.max(left_points)  # Rightmost point of left boundary
                    right_x = np.min(right_points)  # Leftmost point of right boundary
                    
                    if left_x < right_x:  # Valid corridor
                        corridor[y, left_x:right_x+1] = True
                        
            constraint_mask = corridor
            
        elif len(lane_areas) > 0:
            # Fallback: use detected lane areas with expansion
            for area in lane_areas:
                kernel_area = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
                expanded_area = cv2.dilate(area.astype(np.uint8), kernel_area)
                constraint_mask = np.logical_or(constraint_mask, expanded_area > 0)
                
        else:
            # No boundary constraints available - use lower road region only
            constraint_mask[int(input_size*0.4):, :] = True  # Lower 60% of image
            
        return constraint_mask
        
    def apply_semantic_constraints(self, semantic_masks, constraint_mask):
        """Apply boundary constraints to semantic segmentation results"""
        constrained_masks = []
        
        for mask in semantic_masks:
            # Apply constraint mask to eliminate areas outside boundary corridors
            constrained_mask = np.logical_and(mask > 0, constraint_mask)
            constrained_masks.append(constrained_mask)
            
        return constrained_masks
        
    def _semantic_dominant_fusion(self, semantic_masks, boundary_masks):
        """Semantic-guided fusion"""
        if len(semantic_masks) == 0:
            return np.zeros((640, 640), dtype=np.bool_)
            
        semantic_combined = np.zeros((640, 640), dtype=np.bool_)
        for mask in semantic_masks:
            semantic_combined = np.logical_or(semantic_combined, mask > 0)
            
        if len(boundary_masks) > 0:
            # Validate with boundary corridor
            corridor = self._create_boundary_corridor(boundary_masks)
            if np.any(corridor):
                validated_semantic = np.logical_and(semantic_combined, corridor)
                result = np.logical_or(validated_semantic, semantic_combined * 0.7)
            else:
                result = semantic_combined
        else:
            result = semantic_combined
            
        return result
        
    def _balanced_fusion(self, boundary_masks, lane_areas, semantic_masks, b_conf, s_conf):
        """Balanced fusion approach"""
        boundary_combined = np.zeros((640, 640), dtype=np.bool_)
        for area in lane_areas:
            boundary_combined = np.logical_or(boundary_combined, area > 0)
            
        semantic_combined = np.zeros((640, 640), dtype=np.bool_)
        for mask in semantic_masks:
            semantic_combined = np.logical_or(semantic_combined, mask > 0)
            
        # Weighted combination
        total_weight = b_conf + s_conf
        if total_weight > 0:
            b_weight = b_conf / total_weight
            s_weight = s_conf / total_weight
        else:
            b_weight = s_weight = 0.5
            
        boundary_float = boundary_combined.astype(np.float32) * b_weight
        semantic_float = semantic_combined.astype(np.float32) * s_weight
        
        combined_confidence = boundary_float + semantic_float
        result = combined_confidence > 0.4
        
        return result
        
    def _create_boundary_corridor(self, boundary_masks):
        """Create corridor around boundaries"""
        corridor = np.zeros((640, 640), dtype=np.bool_)
        
        if len(boundary_masks) < 1:
            return corridor
            
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        
        for mask in boundary_masks:
            dilated = cv2.dilate(mask.astype(np.uint8), kernel)
            corridor |= (dilated > 0)
            
        return corridor
        
    def process_frame(self, frame):
        """Main frame processing function with boundary-constrained semantic processing"""
        if not self.should_process_frame():
            return None
            
        total_start = time.perf_counter()
        
        # Preprocess
        img, orig, (w0, h0) = self.preprocess_frame(frame)
        
        # Step 1: Boundary inference first (defines geometric constraints)
        t1 = time.perf_counter()
        boundary_outputs = self.infer_boundary_engine(img)
        t2 = time.perf_counter()
        
        # Step 2: Process boundary results
        boundary_masks_raw, boundary_scores, boundary_result = self.postprocess_boundary_masks(
            boundary_outputs, conf_threshold=0.25
        )
        
        # Step 3: Create constraint mask from boundary detection
        if len(boundary_result) == 3:
            boundary_masks, lane_areas, num_lanes = boundary_result
        else:
            boundary_masks, lane_areas, num_lanes = [], [], 0
            
        constraint_mask = self.create_boundary_constraint_mask(
            boundary_masks, lane_areas, input_size=640
        )
        
        # Step 4: Semantic inference (will be constrained)
        semantic_outputs = self.infer_semantic_engine(img)
        t3 = time.perf_counter()
        
        # Step 5: Process semantic results
        semantic_masks_raw, semantic_scores = self.postprocess_semantic_masks(
            semantic_outputs, conf_threshold=0.25
        )
        
        # Step 6: Apply boundary constraints to semantic results
        semantic_masks = self.apply_semantic_constraints(semantic_masks_raw, constraint_mask)
        
        t4 = time.perf_counter()
        
        # Step 7: Confidence analysis (now with constrained semantic)
        boundary_conf, semantic_conf = self.analyze_confidence(
            (boundary_masks, lane_areas, num_lanes), semantic_masks, semantic_scores
        )
        
        # Step 8: Fusion with constrained inputs
        fused_lanes, fusion_method = self.fusion_algorithm(
            (boundary_masks, lane_areas, num_lanes), semantic_masks, semantic_scores,
            boundary_conf, semantic_conf
        )
        
        t5 = time.perf_counter()
        
        # Update timing stats
        self.timing_stats['boundary_inference'].append((t2-t1)*1000)
        self.timing_stats['semantic_inference'].append((t3-t2)*1000)
        self.timing_stats['fusion_processing'].append((t5-t4)*1000)
        self.timing_stats['total_processing'].append((t5-total_start)*1000)
        
        # Create result with constraint information
        result = FusionResult(
            fused_lanes=fused_lanes,
            boundary_masks=boundary_masks,
            semantic_masks=semantic_masks,  # Now constrained
            confidence_map=constraint_mask.astype(np.float32) * max(boundary_conf, semantic_conf),
            boundary_confidence=boundary_conf,
            semantic_confidence=semantic_conf,
            fusion_method=fusion_method,
            processing_times={
                'boundary_inference': (t2-t1)*1000,
                'semantic_inference': (t3-t2)*1000,
                'fusion_processing': (t5-t4)*1000,
                'total': (t5-total_start)*1000
            }
        )
        
        self.previous_result = result
        return result
        
    def visualize_fusion_result(self, original_frame, fusion_result, display_size=640):
        """Visualize fusion results with boundary constraint visualization"""
        vis_frame = cv2.resize(original_frame, (display_size, display_size))
        
        # Create colored overlays
        boundary_overlay = np.zeros_like(vis_frame)
        semantic_overlay = np.zeros_like(vis_frame) 
        fusion_overlay = np.zeros_like(vis_frame)
        constraint_overlay = np.zeros_like(vis_frame)
        
        # Constraint mask visualization (blue tint)
        constraint_mask = fusion_result.confidence_map > 0
        constraint_overlay[constraint_mask] = (100, 50, 0)  # Dark blue tint for allowed regions
        
        # Boundary masks (yellow/magenta)
        boundary_colors = [(0, 255, 255), (255, 0, 255)]  # Yellow, Magenta
        for i, mask in enumerate(fusion_result.boundary_masks):
            if i < len(boundary_colors) and np.any(mask):
                boundary_overlay[mask > 0] = boundary_colors[i]
                
        # Semantic masks (cyan) - now constrained
        for mask in fusion_result.semantic_masks:
            if np.any(mask):
                semantic_overlay[mask > 0] = (255, 255, 0)  # Cyan
            
        # Fused result (green)
        if np.any(fusion_result.fused_lanes):
            fusion_overlay[fusion_result.fused_lanes > 0] = (0, 255, 0)  # Green
        
        # Combine overlays with proper layering
        result_frame = vis_frame.copy()
        
        # Apply constraint region first (subtle background)
        result_frame = cv2.addWeighted(result_frame, 1.0, constraint_overlay, 0.15, 0)
        
        # Apply semantic areas (constrained)
        result_frame = cv2.addWeighted(result_frame, 1.0, semantic_overlay, 0.3, 0)
        
        # Apply fused result
        result_frame = cv2.addWeighted(result_frame, 1.0, fusion_overlay, 0.4, 0)
        
        # Apply lane markings on top (highest visibility)
        result_frame = cv2.addWeighted(result_frame, 1.0, boundary_overlay, 0.8, 0)
        
        # Enhanced text overlay with constraint info
        constraint_area = np.sum(constraint_mask) / (display_size * display_size) * 100
        
        text_info = [
            f"Fusion Method: {fusion_result.fusion_method}",
            f"Boundary Conf: {fusion_result.boundary_confidence:.3f}",
            f"Semantic Conf: {fusion_result.semantic_confidence:.3f}",
            f"Constraint Area: {constraint_area:.1f}%",
            f"Processing: {fusion_result.processing_times['total']:.1f}ms"
        ]
        
        for i, text in enumerate(text_info):
            cv2.putText(result_frame, text, (10, 30 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                       
        # Add constraint legend
        legend_y = display_size - 60
        cv2.putText(result_frame, "Constraint Region:", (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(result_frame, (150, legend_y-15), (170, legend_y-5), (100, 50, 0), -1)
        cv2.putText(result_frame, "Allowed", (175, legend_y-8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                       
        return result_frame
        
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
        return stats

def setup_camera(source):
    """Camera setup function"""
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

# Main testing function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dual-Engine Lane Fusion Test')
    parser.add_argument('--boundary-engine', required=True, 
                       help='Path to boundary detection TensorRT engine')
    parser.add_argument('--semantic-engine', required=True,
                       help='Path to semantic segmentation TensorRT engine') 
    parser.add_argument('--source', default='0', 
                       help='Video source (file, camera index, or URL)')
    parser.add_argument('--display-size', type=int, default=640,
                       help='Display size for visualization')
    parser.add_argument('--target-fps', type=int, default=18,
                       help='Target processing FPS')
    args = parser.parse_args()
    
    # Initialize dual-engine system
    print("Initializing dual-engine fusion system...")
    fusion_system = DualEngineTensorRT(args.boundary_engine, args.semantic_engine)
    fusion_system.target_fps = args.target_fps
    fusion_system.frame_skip_ratio = 30 / args.target_fps
    
    # Setup camera
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
                    # Restart video for continuous testing
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break
                    
            frame_count += 1
            
            # Process frame
            fusion_result = fusion_system.process_frame(frame)
            
            if fusion_result is not None:
                processed_count += 1
                
                # Visualize results
                vis_frame = fusion_system.visualize_fusion_result(frame, fusion_result, args.display_size)
                
                cv2.imshow('Dual-Engine Lane Fusion', vis_frame)
                
                # Print periodic stats
                if processed_count % 30 == 0:
                    stats = fusion_system.get_performance_stats()
                    print(f"\n=== Performance Stats (Frame {frame_count}, Processed {processed_count}) ===")
                    for key, stat in stats.items():
                        print(f"{key}: {stat['mean']:.1f}±{stat['std']:.1f}ms "
                              f"[{stat['min']:.1f}-{stat['max']:.1f}] ({stat['samples']} samples)")
                    
                    if fusion_result:
                        print(f"Latest: Method={fusion_result.fusion_method}, "
                              f"B_conf={fusion_result.boundary_confidence:.3f}, "
                              f"S_conf={fusion_result.semantic_confidence:.3f}")
                        
            # Exit on 'q' key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        
    finally:
        # Final statistics
        print(f"\n=== Final Statistics ===")
        print(f"Total frames: {frame_count}")
        print(f"Processed frames: {processed_count}")
        print(f"Processing ratio: {processed_count/frame_count:.2f}")
        
        final_stats = fusion_system.get_performance_stats()
        for key, stat in final_stats.items():
            print(f"{key}: {stat['mean']:.1f}±{stat['std']:.1f}ms")
            
        cap.release()
        cv2.destroyAllWindows()