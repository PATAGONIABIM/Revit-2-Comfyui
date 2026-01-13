# ComfyUI_WabiSabi_Bridge/nodes/watch_image_node.py
"""Advanced image watching node with multi-channel support"""

import numpy as np
import torch
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path
import cv2
import asyncio
from PIL import Image
import time

from .base_node import WabiSabiBaseNode
from ..core.cache.differential import DifferentialProcessor


class WatchImageProNode(WabiSabiBaseNode):
    """Advanced image watching node with caching and optimization"""
    
    RETURN_TYPES = ("IMAGE", "MASK", "METADATA")
    RETURN_NAMES = ("image", "mask", "metadata")
    OUTPUT_NODE = False
    
    def __init__(self):
        super().__init__()
        self.differential_processor = DifferentialProcessor()
        self._last_image_hash = None
        self._resolution_cache = {}
        
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "source": (["file", "pipe", "memory", "auto"], {"default": "auto"}),
                "path_or_id": ("STRING", {"default": ""}),
                "channel": (["RGB", "RGBA", "DEPTH16", "NORMAL"], {"default": "RGB"}),
                "resolution": (["native", "half", "quarter", "custom"], {"default": "native"}),
                "cache_mode": (["aggressive", "normal", "disabled"], {"default": "normal"}),
            },
            "optional": {
                "custom_width": ("INT", {"default": 1920, "min": 1, "max": 8192}),
                "custom_height": ("INT", {"default": 1080, "min": 1, "max": 8192}),
                "differential_update": ("BOOLEAN", {"default": True}),
                "preview_enabled": ("BOOLEAN", {"default": True}),
            }
        }
    
    async def async_process(self, source: str, path_or_id: str, channel: str, 
                          resolution: str, cache_mode: str, 
                          custom_width: int = 1920, custom_height: int = 1080,
                          differential_update: bool = True,
                          preview_enabled: bool = True, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Process image data from various sources"""
        
        # Configure based on inputs
        self.config.channel.type = source if source != "auto" else self.config.channel.type
        if source == "file":
            self.config.channel.file_path = path_or_id
        
        # Set cache mode
        self.config.cache.mode = cache_mode
        self.config.cache.enabled = cache_mode != "disabled"
        
        # Get latest data
        data = await self.get_latest_data()
        
        if not data:
            return self.get_default_output()
        
        # Extract image based on channel type
        image_data = await self._extract_channel_data(data, channel)
        
        if image_data is None:
            return self.get_default_output()
        
        # Check cache
        image_hash = self.cache_manager.compute_hash(image_data) if self.cache_manager else None
        
        if cache_mode != "disabled" and image_hash == self._last_image_hash:
            # Return cached result
            cached = await self.cache_manager.get(f"processed_{image_hash}")
            if cached:
                return cached['image'], cached['mask'], cached['metadata']
        
        # Apply resolution scaling
        scaled_image = self._apply_resolution_scaling(image_data, resolution, custom_width, custom_height)
        
        # Process differential update if enabled
        mask = np.ones(scaled_image.shape[:2], dtype=np.uint8) * 255
        diff_stats = {}
        
        if differential_update:
            mask, diff_stats = self.differential_processor.compute_diff_mask(scaled_image)
        
        # Convert to tensors
        image_tensor = self.image_to_tensor(scaled_image)
        mask_tensor = self.image_to_tensor(mask)
        
        # Prepare metadata
        metadata = {
            "source": source,
            "channel": channel,
            "resolution": f"{scaled_image.shape[1]}x{scaled_image.shape[0]}",
            "original_resolution": f"{image_data.shape[1]}x{image_data.shape[0]}",
            "timestamp": time.time(),
            "latency_ms": self.channel_manager.get_channel().get_latency() if self.channel_manager else 0,
            "differential_stats": diff_stats,
            **data.get('metadata', {})
        }
        
        # Cache processed result
        if cache_mode != "disabled" and self.cache_manager:
            await self.cache_manager.put(f"processed_{image_hash}", {
                'image': image_tensor,
                'mask': mask_tensor,
                'metadata': metadata
            })
        
        self._last_image_hash = image_hash
        
        # Send preview if enabled
        if preview_enabled:
            # TODO: Send to preview widget via WebSocket
            pass
        
        return image_tensor, mask_tensor, metadata
    
    async def _extract_channel_data(self, data: Dict[str, Any], channel: str) -> Optional[np.ndarray]:
        """Extract specific channel data from input"""
        
        if data.get('type') == 'image' and 'path' in data:
            # Load image from file
            image_path = Path(data['path'])
            if image_path.exists():
                image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
                if image is not None:
                    # Convert BGR to RGB
                    if len(image.shape) == 3 and image.shape[2] >= 3:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    return image
        
        elif 'image_data' in data:
            # Direct image data
            image_data = data['image_data']
            
            if isinstance(image_data, np.ndarray):
                return image_data
            elif isinstance(image_data, bytes):
                # Decode from bytes
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                if image is not None and len(image.shape) == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return image
        
        elif channel == "DEPTH16" and 'depth_data' in data:
            # Handle depth data
            depth = data['depth_data']
            if isinstance(depth, np.ndarray):
                # Normalize to 0-1 range
                if depth.dtype == np.uint16:
                    return depth.astype(np.float32) / 65535.0
                else:
                    return depth
        
        elif channel == "NORMAL" and 'normal_data' in data:
            # Handle normal map data
            return data['normal_data']
        
        return None
    
    def _apply_resolution_scaling(self, image: np.ndarray, resolution: str, 
                                custom_width: int, custom_height: int) -> np.ndarray:
        """Apply resolution scaling to image"""
        
        h, w = image.shape[:2]
        
        # Check resolution cache
        cache_key = f"{w}x{h}_{resolution}"
        if cache_key in self._resolution_cache:
            target_size = self._resolution_cache[cache_key]
        else:
            if resolution == "native":
                target_size = (w, h)
            elif resolution == "half":
                target_size = (w // 2, h // 2)
            elif resolution == "quarter":
                target_size = (w // 4, h // 4)
            elif resolution == "custom":
                target_size = (custom_width, custom_height)
            else:
                target_size = (w, h)
            
            self._resolution_cache[cache_key] = target_size
        
        if target_size == (w, h):
            return image
        
        # Use high-quality interpolation
        return cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
    
    def get_default_output(self) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Return default output when no data available"""
        # Create blank image
        blank_image = np.zeros((512, 512, 3), dtype=np.uint8)
        blank_mask = np.zeros((512, 512), dtype=np.uint8)
        
        return (
            self.image_to_tensor(blank_image),
            self.image_to_tensor(blank_mask),
            {"error": "No data available"}
        )


