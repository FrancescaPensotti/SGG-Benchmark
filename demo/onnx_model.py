import torch
import numpy as np
import time
import onnxruntime as ort
import os
import cv2
import seaborn as sns

from sgg_benchmark.config import load_config_from_file
from sgg_benchmark.data.build import build_transforms
from sgg_benchmark.data import get_dataset_statistics
from .demo_model import SGG_Model

class SGG_ONNX_Model(SGG_Model):
    def __init__(self, config, onnx_path, provider='CUDAExecutionProvider', dcs=100, tracking=False, rel_conf=0.1, box_conf=0.5, show_fps=True) -> None:
        # We don't call super().__init__ because it builds the PyTorch model
        self.show_fps = show_fps
        self.tracking = tracking
        self.rel_conf = rel_conf
        self.box_conf = box_conf

        # for visu
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.font_thickness = 1
        self.text_padding = 2

        # Load ONNX Session first so we can read embedded class names from metadata
        print(f"Loading ONNX model from {onnx_path} with {provider}...")
        self._fix_ld_library_path()
        try:
            self.session = ort.InferenceSession(onnx_path, providers=[provider, 'CPUExecutionProvider'])
            print(f"ONNX Session loaded with providers: {self.session.get_providers()}")
        except Exception as e:
            print(f"Failed to load ONNX session: {e}")
            raise e

        self.input_name = self.session.get_inputs()[0].name

        # Try to load class names from ONNX metadata (embedded at export time).
        # This requires no extra dependencies — only onnxruntime.
        _meta = self.session.get_modelmeta().custom_metadata_map
        if 'obj_classes' in _meta and 'rel_classes' in _meta:
            import json as _json
            _obj = _json.loads(_meta['obj_classes'])
            _rel = _json.loads(_meta['rel_classes'])
            # Normalise to the same dict format used by get_dataset_statistics.
            # Background (index 0) was stripped at export time, so obj classes start at 1
            # to match the 1-based label indices the model outputs.
            self.stats = {
                'obj_classes': {i: v for i, v in enumerate(_obj, start=1)},
                'rel_classes': {i: v for i, v in enumerate(_rel)},
            }
            print(f"Loaded {len(_obj)} object classes and {len(_rel)} relation classes "
                  f"from ONNX metadata.")
        else:
            # Fallback: load class names from config + dataset statistics
            if config is None:
                raise ValueError(
                    "This ONNX model does not have embedded class names. "
                    "Please provide --config pointing to the config.yml file."
                )
            self.cfg = load_config_from_file(config)
            self.cfg.test.custum_eval = True
            self.cfg.output_dir = os.path.dirname(config)
            self.stats = get_dataset_statistics(self.cfg)
            if 'obj_classes' in self.stats and 0 in self.stats['obj_classes']:
                _ = self.stats['obj_classes'].pop(0)

        self.obj_class_colors = sns.color_palette('Paired', len(self.stats['obj_classes'])+2)
        self.obj_class_colors = [(int(c[2]*255), int(c[1]*255), int(c[0]*255)) for c in self.obj_class_colors]
        
        self.last_time = 0
        if self.tracking:
            from boxmot import OcSort
            self.tracker = OcSort(per_class=True, det_thresh=0, max_age=20, min_hits=1, asso_threshold=0.2, delta_t=2, asso_func='giou', inertia=0.2, use_byte=True)

        self.pre_time_bench = []
        self.detec_time_bench = []
        self.post_time_bench = []
        self.last_time = time.time()

    def _fix_ld_library_path(self):
        try:
            import site
            site_dirs = site.getsitepackages()
            if hasattr(site, 'getusersitepackages'):
                site_dirs.append(site.getusersitepackages())
            
            additional_paths = []
            for packages_path in site_dirs:
                if not os.path.exists(packages_path): continue
                for item in os.listdir(packages_path):
                    if item.endswith("_libs") or item.endswith(".libs"):
                        additional_paths.append(os.path.join(packages_path, item))
                nvidia_path = os.path.join(packages_path, "nvidia")
                if os.path.exists(nvidia_path):
                    for sub in os.listdir(nvidia_path):
                        lib_path = os.path.join(nvidia_path, sub, "lib")
                        if os.path.exists(lib_path):
                            additional_paths.append(lib_path)
            
            if additional_paths:
                current_ld = os.environ.get("LD_LIBRARY_PATH", "")
                unique_paths = list(set(additional_paths))
                new_ld = ":".join(unique_paths)
                os.environ["LD_LIBRARY_PATH"] = f"{new_ld}:{current_ld}" if current_ld else new_ld
        except Exception:
            pass

    def _preprocess_for_onnx(self, image: np.ndarray, size: int = 640) -> np.ndarray:
        """Letterbox + BGR→RGB + CHW + /255 — matches export_onnx.py exactly."""
        h, w = image.shape[:2]
        r = min(size / h, size / w)
        nw, nh = int(round(w * r)), int(round(h * r))
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        top    = int(round((size - nh) / 2 - 0.1))
        bottom = int(round((size - nh) / 2 + 0.1))
        left   = int(round((size - nw) / 2 - 0.1))
        right  = int(round((size - nw) / 2 + 0.1))
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=(114, 114, 114))
        img = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1)).astype(np.float32) / 255.0
        return img[None, ...]  # (1, 3, H, W)

    def predict(self, image, visu_type='image', return_attention=False):
        out_img = image.copy()
        start_time = time.time()

        # 1. Pre-processing: letterbox BGR→RGB, matching the export script exactly.
        #    (The inherited _pre_processing does BGR→RGB→BGR due to ToTensorYOLO double-flip.)
        img_numpy = self._preprocess_for_onnx(image)  # (1, 3, 640, 640)  RGB float32
        
        pre_process_time = (time.time() - start_time) * 1000
        self.pre_time_bench.append(pre_process_time)
        
        # 2. ONNX Inference
        t_start = time.time()
        outputs = self.session.run(None, {self.input_name: img_numpy})
        det_time = (time.time() - start_time)
        self.detec_time_bench.append((time.time() - t_start) * 1000)
        
        # 3. Post-processing
        t_start2 = time.time()
        
        # Handle the case where the model returns boxes and rels directly (from GeneralizedYOLO.export=True)
        if len(outputs) >= 2:
            # shapes are (N, 6) and (M, 5)
            # boxes: x1, y1, x2, y2, label, score
            # rels: subj_idx, obj_idx, label, triplet_score, rel_score
            boxes_raw = outputs[0].copy()
            rels_raw = outputs[1].copy()
            
            # Rescale boxes to original image site
            orig_h, orig_w = image.shape[:2]
            
            # The model and pre-processing use centered Letterbox padding (padding on both sides).
            # This is standard for Ultralytics YOLO models.
            input_h, input_w = img_numpy.shape[2:] 
            
            # Standard Letterbox scaling
            gain = min(input_w / orig_w, input_h / orig_h)
            pad_x = (input_w - orig_w * gain) / 2
            pad_y = (input_h - orig_h * gain) / 2
            
            # Apply scaling and remove padding
            boxes_raw[:, [0, 2]] -= pad_x
            boxes_raw[:, [1, 3]] -= pad_y
            boxes_raw[:, :4] /= gain

            # Clip to original image boundaries
            boxes_raw[:, [0, 2]] = np.clip(boxes_raw[:, [0, 2]], 0, orig_w)
            boxes_raw[:, [1, 3]] = np.clip(boxes_raw[:, [1, 3]], 0, orig_h)
            
            # Filter rels by triplet score only.
            # Do NOT gate rels through box_conf — the ONNX already filtered boxes at
            # obj_thres=0.05 during export; a second stricter box_conf filter would
            # cascade and kill most relations (their referenced boxes get removed).
            if len(rels_raw) > 0:
                rels_raw = rels_raw[rels_raw[:, 3] >= self.rel_conf]

            # box_conf controls which *extra* (non-relation) boxes to display.
            # Boxes involved in any kept relation are always shown.
            rel_box_set: set = set()
            if len(rels_raw) > 0:
                rel_box_set = set(rels_raw[:, 0].astype(np.int32).tolist() +
                                  rels_raw[:, 1].astype(np.int32).tolist())

            box_keep = np.array([
                i for i in range(len(boxes_raw))
                if boxes_raw[i, 5] >= self.box_conf or i in rel_box_set
            ], dtype=np.int32)

            if len(box_keep) < len(boxes_raw):
                # Remap rel indices to the kept-box subset
                old_to_new = np.full(len(boxes_raw), -1, dtype=np.int32)
                old_to_new[box_keep] = np.arange(len(box_keep))
                boxes_raw = boxes_raw[box_keep]
                if len(rels_raw) > 0:
                    rels_raw[:, 0] = old_to_new[rels_raw[:, 0].astype(np.int32)]
                    rels_raw[:, 1] = old_to_new[rels_raw[:, 1].astype(np.int32)]

            # Convert to [x1, y1, x2, y2, score, label] for demo_model compatibility
            bboxes = np.concatenate([boxes_raw[:, :4], boxes_raw[:, 5:6], boxes_raw[:, 4:5]], axis=1)
            rels = rels_raw
        else:
            # Fallback for old export or partial export
            print("Error: ONNX model does not have expected SGG outputs. Please re-export the full model.")
            return out_img, rels

        # update tracker
        if self.tracking and len(bboxes) > 0:
            tracks = self.tracker.update(bboxes, image)
            bboxes = np.concatenate((bboxes, np.zeros((len(bboxes), 1))), axis=1)
            if len(tracks) > 0:
                for i, track in enumerate(tracks):
                    cur_id = int(track[7])
                    class_label = str(int(bboxes[cur_id][5]))
                    bboxes[cur_id][6] = int(class_label + str(int(track[4])))
                    bboxes[cur_id][0:4] = track[0:4]

        # 4. Rendering
        if visu_type == 'video':
            # Use a copy for drawing to avoid modifying the original if shared
            out_img = self.draw_full_graph(image.copy(), bboxes, rels)
            
            true_fps = 1.0 / det_time if det_time > 0 else float('inf')
            
            # Use nicer text display from demo_model or similar
            image_height, image_width = out_img.shape[:2]
            font_scale = (0.3 * image_width) / 500
            # White text with black shadow for better visibility
            for i, text in enumerate([f"FPS: {true_fps:.1f}", f"Inference: {det_time*1000:.1f}ms"]):
                pos = (15, 30 + i * 40)
                cv2.putText(out_img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(out_img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (50, 255, 50), 2, cv2.LINE_AA)
            
            # The out_img is BGR (since image was BGR and we use OpenCV to draw)
            return out_img, (bboxes, rels)
        
        elif visu_type == 'image':
            graph_img = self.visualize_graph(rels, bboxes)
            out_img = self.draw_boxes_image(bboxes, out_img)
            return out_img, graph_img
            
        return bboxes, rels
