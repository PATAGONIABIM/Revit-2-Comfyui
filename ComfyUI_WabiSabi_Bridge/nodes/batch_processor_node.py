# ComfyUI_WabiSabi_Bridge/nodes/batch_processor_node.py
"""Batch image processing node"""

import numpy as np
import torch
from typing import Dict, Any, Tuple, List, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import cv2

from .base_node import WabiSabiBaseNode


class BatchImageProcessorNode(WabiSabiBaseNode):
    """Process multiple images in parallel"""
    
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("batch_images", "batch_size")
    OUTPUT_NODE = False
    
    def __init__(self):
        super().__init__()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "batch_source": (["sequence", "multi_view", "time_series"], {"default": "sequence"}),
                "max_batch_size": ("INT", {"default": 8, "min": 1, "max": 32}),
                "processing_mode": (["parallel", "sequential"], {"default": "parallel"}),
            },
            "optional": {
                "apply_transform": (["none", "normalize", "augment"], {"default": "none"}),
                "frame_interval": ("INT", {"default": 1, "min": 1, "max": 60}),
            }
        }
    
    async def async_process(self, batch_source: str, max_batch_size: int,
                          processing_mode: str, apply_transform: str = "none",
                          frame_interval: int = 1, **kwargs) -> Tuple[torch.Tensor, int]:
        """Process batch of images"""
        
        # Collect batch data
        batch_data = await self._collect_batch_data(batch_source, max_batch_size, frame_interval)
        
        if not batch_data:
            return self.get_default_output()
        
        # Process batch
        if processing_mode == "parallel":
            processed_images = await self._process_parallel(batch_data, apply_transform)
        else:
            processed_images = await self._process_sequential(batch_data, apply_transform)
        
        # Stack into batch tensor
        batch_tensor = torch.stack(processed_images)
        
        return batch_tensor, len(processed_images)
    
    async def _collect_batch_data(self, source: str, max_size: int, interval: int) -> List[np.ndarray]:
        """Collect batch data based on source type"""
        batch = []
        
        if source == "sequence":
            # Collect sequential frames
            for i in range(max_size):
                data = await self.get_latest_data(timeout=0.1)
                if data and 'image_data' in data:
                    batch.append(data['image_data'])
                    
                if i < max_size - 1:
                    await asyncio.sleep(1.0 / 30 * interval)  # Wait based on interval
                    
        elif source == "multi_view":
            # Collect from multiple views simultaneously
            data = await self.get_latest_data()
            if data and 'multi_view_data' in data:
                for view_data in data['multi_view_data'][:max_size]:
                    if 'image_data' in view_data:
                        batch.append(view_data['image_data'])
                        
        elif source == "time_series":
            # Collect time series data
            # Implementation depends on specific time series format
            pass
        
        return batch
    
    async def _process_parallel(self, batch: List[np.ndarray], transform: str) -> List[torch.Tensor]:
        """Process batch in parallel"""
        loop = asyncio.get_event_loop()
        
        tasks = []
        for image in batch:
            task = loop.run_in_executor(
                self.executor,
                self._process_single_image,
                image,
                transform
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return [self.image_to_tensor(img) for img in results]
    
    async def _process_sequential(self, batch: List[np.ndarray], transform: str) -> List[torch.Tensor]:
        """Process batch sequentially"""
        results = []
        
        for image in batch:
            processed = self._process_single_image(image, transform)
            results.append(self.image_to_tensor(processed))
        
        return results
    
    def _process_single_image(self, image: np.ndarray, transform: str) -> np.ndarray:
        """Process a single image with optional transforms"""
        
        if transform == "normalize":
            # Normalize to 0-1 range
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            
            # Standardize
            mean = np.mean(image, axis=(0, 1), keepdims=True)
            std = np.std(image, axis=(0, 1), keepdims=True) + 1e-8
            image = (image - mean) / std
            
        elif transform == "augment":
            # Simple augmentation
            if np.random.random() > 0.5:
                image = cv2.flip(image, 1)  # Horizontal flip
            
            # Random brightness
            brightness = np.random.uniform(0.8, 1.2)
            image = np.clip(image * brightness, 0, 1 if image.dtype == np.float32 else 255)
        
        return image
    
    def get_default_output(self) -> Tuple[torch.Tensor, int]:
        """Return default output"""
        blank = np.zeros((512, 512, 3), dtype=np.uint8)
        return torch.stack([self.image_to_tensor(blank)]), 1

