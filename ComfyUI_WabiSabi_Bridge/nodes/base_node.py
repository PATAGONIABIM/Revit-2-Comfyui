
# ComfyUI_WabiSabi_Bridge/nodes/base_node.py
"""Base node class for all WabiSabi nodes"""

import asyncio
from typing import Dict, Any, Tuple, Optional, List, Callable
from abc import ABC, abstractmethod
import numpy as np
import torch
from PIL import Image
import time

from ..core.channel_manager import ChannelManager
from ..core.config import WabiSabiConfig
from ..core.cache.manager import CacheManager
from ..core.communication.base import ChannelStatus


class WabiSabiBaseNode(ABC):
    """Base class for all WabiSabi Bridge nodes"""
    
    # ComfyUI required class variables
    CATEGORY = "WabiSabi Bridge"
    FUNCTION = "process"
    
    def __init__(self):
        self.config = WabiSabiConfig.default()
        self.channel_manager: Optional[ChannelManager] = None
        self.cache_manager: Optional[CacheManager] = None
        self._initialized = False
        self._last_data: Optional[Dict[str, Any]] = None
        self._metrics = {
            'process_time': 0,
            'last_update': 0,
            'frames_processed': 0
        }
        
    @classmethod
    @abstractmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """Define input types for the node"""
        pass
    
    @abstractmethod
    def get_return_types(self) -> Tuple[str, ...]:
        """Define return types for the node"""
        pass
    
    @classmethod
    def RETURN_TYPES(cls) -> Tuple[str, ...]:
        """ComfyUI return types"""
        # This will be overridden by child classes
        return ("IMAGE",)
    
    async def initialize_if_needed(self) -> bool:
        """Initialize managers if not already done"""
        if self._initialized:
            return True
        
        try:
            # Initialize channel manager
            self.channel_manager = ChannelManager(self.config)
            success = await self.channel_manager.initialize()
            
            if not success:
                print("Failed to initialize channel manager")
                return False
            
            # Initialize cache manager
            self.cache_manager = CacheManager(self.config.cache)
            await self.cache_manager.initialize()
            
            # Subscribe to channel updates
            channel = self.channel_manager.get_channel()
            if channel:
                channel.subscribe(self._on_data_received)
            
            self._initialized = True
            return True
            
        except Exception as e:
            print(f"Initialization error: {e}")
            return False
    
    def _on_data_received(self, data: Dict[str, Any]) -> None:
        """Handle data received from channel"""
        self._last_data = data
        self._metrics['last_update'] = time.time()
    
    async def get_latest_data(self, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """Get latest data with timeout"""
        if not await self.initialize_if_needed():
            return None
        
        channel = self.channel_manager.get_channel()
        if not channel:
            return None
        
        # If we have recent data, return it
        if self._last_data and (time.time() - self._metrics['last_update']) < 0.1:
            return self._last_data
        
        # Otherwise, try to read
        try:
            data = await asyncio.wait_for(channel.read_data(), timeout=timeout)
            if data:
                self._last_data = data
                self._metrics['last_update'] = time.time()
            return data
        except asyncio.TimeoutError:
            print(f"Timeout reading data")
            return self._last_data
    
    def process(self, **kwargs) -> Tuple[Any, ...]:
        """Main processing function called by ComfyUI"""
        start_time = time.time()
        
        try:
            # Run async initialization and processing
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            result = loop.run_until_complete(self.async_process(**kwargs))
            
            # Update metrics
            self._metrics['process_time'] = time.time() - start_time
            self._metrics['frames_processed'] += 1
            
            return result
            
        except Exception as e:
            print(f"Process error: {e}")
            return self.get_default_output()
        finally:
            loop.close()
    
    @abstractmethod
    async def async_process(self, **kwargs) -> Tuple[Any, ...]:
        """Async processing to be implemented by child classes"""
        pass
    
    @abstractmethod
    def get_default_output(self) -> Tuple[Any, ...]:
        """Get default output when processing fails"""
        pass
    
    def tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert PyTorch tensor to numpy image"""
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        # Convert to numpy
        image = tensor.cpu().numpy()
        
        # Rearrange dimensions if needed (CHW to HWC)
        if image.shape[0] in [1, 3, 4]:
            image = np.transpose(image, (1, 2, 0))
        
        # Convert to uint8
        if image.dtype != np.uint8:
            image = (image * 255).clip(0, 255).astype(np.uint8)
        
        return image
    
    def image_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert numpy image to PyTorch tensor"""
        # Ensure correct shape
        if image.ndim == 2:
            image = np.expand_dims(image, axis=2)
        
        # Convert to float32
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        # Rearrange dimensions (HWC to CHW)
        image = np.transpose(image, (2, 0, 1))
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(image).unsqueeze(0)
        
        return tensor
    
    def get_status(self) -> Dict[str, Any]:
        """Get node status information"""
        status = {
            'initialized': self._initialized,
            'metrics': self._metrics.copy()
        }
        
        if self.channel_manager:
            channel = self.channel_manager.get_channel()
            if channel:
                status['channel'] = {
                    'status': channel.get_status().value,
                    'latency_ms': channel.get_latency(),
                    'metrics': channel.metrics.__dict__
                }
        
        if self.cache_manager:
            status['cache'] = self.cache_manager.get_stats()
        
        return status
    
    @classmethod
    def IS_CHANGED(cls, **kwargs) -> float:
        """Tell ComfyUI when the node output has changed"""
        # Return current time to force updates when in auto mode
        return time.time()

