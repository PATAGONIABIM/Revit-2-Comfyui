
# ComfyUI_WabiSabi_Bridge/nodes/segmentation_node.py
"""Smart segmentation node for architectural elements"""

import numpy as np
import torch
from typing import Dict, Any, Tuple, List, Optional
import cv2
from collections import defaultdict

from .base_node import WabiSabiBaseNode


class SmartSegmentationNode(WabiSabiBaseNode):
    """Extract segmentation masks by category"""
    
    RETURN_TYPES = ("IMAGE", "MASKS", "STRING")
    RETURN_NAMES = ("segmentation_viz", "masks_list", "labels_json")
    OUTPUT_NODE = False
    
    # Architectural categories mapping
    CATEGORY_COLORS = {
        "wall": [255, 255, 255],
        "floor": [128, 128, 128],
        "ceiling": [200, 200, 200],
        "window": [100, 200, 255],
        "door": [150, 100, 50],
        "furniture": [255, 200, 100],
        "structural": [200, 100, 100],
        "mechanical": [100, 100, 200],
        "vegetation": [100, 255, 100],
        "other": [255, 100, 255]
    }
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "segmentation_source": (["from_data", "from_image"], {"default": "from_data"}),
                "filter_by": (["category", "material", "element_id", "custom_param"], {"default": "category"}),
                "output_mode": (["individual_masks", "combined_viz", "both"], {"default": "both"}),
            },
            "optional": {
                "segmentation_image": ("IMAGE",),
                "element_ids_image": ("IMAGE",),
                "filter_values": ("STRING", {"default": "wall,floor,window", "multiline": False}),
                "min_area": ("INT", {"default": 100, "min": 0, "max": 10000}),
            }
        }
    
    async def async_process(self, segmentation_source: str, filter_by: str,
                          output_mode: str, segmentation_image: Optional[torch.Tensor] = None,
                          element_ids_image: Optional[torch.Tensor] = None,
                          filter_values: str = "", min_area: int = 100,
                          **kwargs) -> Tuple[torch.Tensor, List[torch.Tensor], str]:
        """Process segmentation data"""
        
        # Get segmentation data
        if segmentation_source == "from_image" and segmentation_image is not None:
            seg_data = self.tensor_to_image(segmentation_image)
        else:
            data = await self.get_latest_data()
            if not data or 'segmentation_data' not in data:
                return self.get_default_output()
            seg_data = data['segmentation_data']
        
        # Get element mapping if available
        element_mapping = {}
        if element_ids_image is not None:
            ids_data = self.tensor_to_image(element_ids_image)
            element_mapping = self._extract_element_mapping(ids_data, data.get('metadata', {}))
        elif segmentation_source == "from_data":
            data = await self.get_latest_data()
            if data and 'element_mapping' in data:
                element_mapping = data['element_mapping']
        
        # Parse filter values
        filter_list = [v.strip() for v in filter_values.split(',') if v.strip()]
        
        # Extract masks based on filter
        masks, labels = self._extract_masks(seg_data, element_mapping, filter_by, filter_list, min_area)
        
        # Create visualization
        viz_image = self._create_visualization(seg_data, masks, labels)
        
        # Prepare outputs
        viz_tensor = self.image_to_tensor(viz_image)
        mask_tensors = [self.image_to_tensor(mask) for mask in masks]
        
        # Create labels JSON
        labels_data = {
            "filter_by": filter_by,
            "labels": labels,
            "mask_count": len(masks),
            "categories": list(set(label['category'] for label in labels if 'category' in label))
        }
        labels_json = json.dumps(labels_data, indent=2)
        
        return viz_tensor, mask_tensors, labels_json
    
    def _extract_element_mapping(self, ids_image: np.ndarray, metadata: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
        """Extract element mapping from ID image"""
        unique_ids = np.unique(ids_image)
        
        mapping = {}
        for element_id in unique_ids:
            if element_id == 0:  # Skip background
                continue
                
            # Get element info from metadata if available
            element_info = metadata.get('elements', {}).get(str(element_id), {})
            
            mapping[element_id] = {
                'id': element_id,
                'category': element_info.get('category', 'other'),
                'material': element_info.get('material', 'unknown'),
                'name': element_info.get('name', f'Element_{element_id}')
            }
        
        return mapping
    
    def _extract_masks(self, seg_image: np.ndarray, element_mapping: Dict[int, Dict[str, Any]],
                      filter_by: str, filter_values: List[str], min_area: int) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
        """Extract individual masks based on filter criteria"""
        
        masks = []
        labels = []
        
        # Get unique segments
        if len(seg_image.shape) == 3:
            # Convert RGB segmentation to label image
            label_image = self._rgb_to_labels(seg_image)
        else:
            label_image = seg_image
        
        unique_labels = np.unique(label_image)
        
        for label in unique_labels:
            if label == 0:  # Skip background
                continue
            
            # Create mask for this label
            mask = (label_image == label).astype(np.uint8) * 255
            
            # Check area
            area = np.sum(mask > 0)
            if area < min_area:
                continue
            
            # Get element info
            element_info = element_mapping.get(label, {})
            
            # Apply filter
            if filter_by == "category":
                category = element_info.get('category', 'other')
                if not filter_values or category in filter_values:
                    masks.append(mask)
                    labels.append({
                        'label': int(label),
                        'category': category,
                        'area': int(area),
                        'name': element_info.get('name', f'Element_{label}')
                    })
            
            elif filter_by == "material":
                material = element_info.get('material', 'unknown')
                if not filter_values or material in filter_values:
                    masks.append(mask)
                    labels.append({
                        'label': int(label),
                        'material': material,
                        'area': int(area),
                        'name': element_info.get('name', f'Element_{label}')
                    })
            
            elif filter_by == "element_id":
                if not filter_values or str(label) in filter_values:
                    masks.append(mask)
                    labels.append({
                        'label': int(label),
                        'element_id': int(label),
                        'area': int(area),
                        'name': element_info.get('name', f'Element_{label}')
                    })
        
        return masks, labels
    
    def _rgb_to_labels(self, rgb_image: np.ndarray) -> np.ndarray:
        """Convert RGB segmentation to label image"""
        h, w = rgb_image.shape[:2]
        label_image = np.zeros((h, w), dtype=np.uint32)
        
        # Create label from RGB values (simple hash)
        if len(rgb_image.shape) == 3:
            label_image = (rgb_image[:,:,0].astype(np.uint32) << 16) + \
                         (rgb_image[:,:,1].astype(np.uint32) << 8) + \
                         rgb_image[:,:,2].astype(np.uint32)
        
        return label_image
    
    def _create_visualization(self, seg_image: np.ndarray, masks: List[np.ndarray], 
                            labels: List[Dict[str, Any]]) -> np.ndarray:
        """Create visualization of segmentation with overlays"""
        
        # Create base visualization
        if len(seg_image.shape) == 2:
            viz = cv2.applyColorMap(seg_image.astype(np.uint8), cv2.COLORMAP_HSV)
            viz = cv2.cvtColor(viz, cv2.COLOR_BGR2RGB)
        else:
            viz = seg_image.copy()
        
        # Overlay mask boundaries
        for i, (mask, label) in enumerate(zip(masks, labels)):
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw contours
            color = self.CATEGORY_COLORS.get(label.get('category', 'other'), [255, 255, 255])
            cv2.drawContours(viz, contours, -1, color, 2)
            
            # Add label
            if contours:
                M = cv2.moments(contours[0])
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(viz, label.get('name', '')[:20], (cx, cy),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return viz
    
    def get_default_output(self) -> Tuple[torch.Tensor, List[torch.Tensor], str]:
        """Return default output"""
        blank = np.zeros((512, 512, 3), dtype=np.uint8)
        return self.image_to_tensor(blank), [], "{}"

