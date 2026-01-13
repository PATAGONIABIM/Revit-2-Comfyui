# ComfyUI_WabiSabi_Bridge/nodes/lighting_control_node.py
"""Lighting control node for generating lighting maps"""

import numpy as np
import torch
from typing import Dict, Any, Tuple, Optional
import cv2

from .base_node import WabiSabiBaseNode


class LightingControlNode(WabiSabiBaseNode):
    """Extract and generate lighting control maps"""
    
    RETURN_TYPES = ("IMAGE", "CONDITIONING", "FLOAT")
    RETURN_NAMES = ("lighting_map", "lighting_conditioning", "sun_intensity")
    OUTPUT_NODE = False
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "map_type": (["shadow_map", "light_direction", "ambient_occlusion", "combined"], 
                           {"default": "combined"}),
                "resolution": ("INT", {"default": 512, "min": 128, "max": 2048}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
            },
            "optional": {
                "reference_image": ("IMAGE",),
                "sun_override": ("BOOLEAN", {"default": False}),
                "sun_azimuth": ("FLOAT", {"default": 45.0, "min": 0.0, "max": 360.0}),
                "sun_elevation": ("FLOAT", {"default": 45.0, "min": 0.0, "max": 90.0}),
            }
        }
    
    async def async_process(self, map_type: str, resolution: int, intensity: float,
                          reference_image: Optional[torch.Tensor] = None,
                          sun_override: bool = False, sun_azimuth: float = 45.0,
                          sun_elevation: float = 45.0, **kwargs) -> Tuple[torch.Tensor, Dict[str, Any], float]:
        """Generate lighting control maps"""
        
        # Get lighting data
        data = await self.get_latest_data()
        if not data or 'lighting' not in data:
            return self.get_default_output()
        
        lighting_data = data['lighting']
        
        # Get or calculate sun direction
        if sun_override:
            sun_direction = self._calculate_sun_direction(sun_azimuth, sun_elevation)
        else:
            sun_direction = lighting_data.get('sun_direction', [0.7, 0.3, 0.6])
        
        # Get reference dimensions
        if reference_image is not None:
            ref_img = self.tensor_to_image(reference_image)
            height, width = ref_img.shape[:2]
        else:
            width = height = resolution
        
        # Generate lighting map based on type
        if map_type == "shadow_map":
            lighting_map = self._generate_shadow_map(width, height, sun_direction, lighting_data)
        elif map_type == "light_direction":
            lighting_map = self._generate_light_direction_map(width, height, sun_direction)
        elif map_type == "ambient_occlusion":
            lighting_map = self._generate_ao_map(width, height, data)
        else:  # combined
            lighting_map = self._generate_combined_map(width, height, sun_direction, lighting_data, data)
        
        # Apply intensity
        lighting_map = np.clip(lighting_map * intensity, 0, 1)
        
        # Convert to tensor
        lighting_tensor = self.image_to_tensor(lighting_map)
        
        # Create conditioning data for ControlNet
        conditioning = {
            "type": "lighting",
            "sun_direction": sun_direction,
            "intensity": intensity,
            "time_of_day": lighting_data.get('time_of_day', 'unknown'),
            "map_type": map_type
        }
        
        # Calculate sun intensity
        sun_intensity = lighting_data.get('sun_intensity', 1.0) * intensity
        
        return lighting_tensor, conditioning, sun_intensity
    
    def _calculate_sun_direction(self, azimuth: float, elevation: float) -> list:
        """Calculate sun direction vector from angles"""
        # Convert to radians
        az_rad = np.radians(azimuth)
        el_rad = np.radians(elevation)
        
        # Calculate direction vector
        x = np.cos(el_rad) * np.sin(az_rad)
        y = np.sin(el_rad)
        z = np.cos(el_rad) * np.cos(az_rad)
        
        return [float(x), float(y), float(z)]
    
    def _generate_shadow_map(self, width: int, height: int, 
                           sun_direction: list, lighting_data: Dict[str, Any]) -> np.ndarray:
        """Generate shadow map"""
        # Create gradient based on sun direction
        shadow_map = np.ones((height, width), dtype=np.float32)
        
        # Simple directional shadow simulation
        cx, cy = width // 2, height // 2
        
        for y in range(height):
            for x in range(width):
                # Distance from center
                dx = (x - cx) / width
                dy = (y - cy) / height
                
                # Dot product with sun direction (projected to 2D)
                dot = dx * sun_direction[0] + dy * (-sun_direction[1])
                
                # Shadow intensity
                shadow = 0.5 + 0.5 * np.clip(dot, -1, 1)
                shadow_map[y, x] = shadow
        
        # Add some noise for realism
        noise = np.random.normal(0, 0.02, (height, width))
        shadow_map = np.clip(shadow_map + noise, 0, 1)
        
        # Convert to RGB
        return np.stack([shadow_map] * 3, axis=2)
    
    def _generate_light_direction_map(self, width: int, height: int, sun_direction: list) -> np.ndarray:
        """Generate light direction map (normal map style)"""
        # Create normal map representing light direction
        direction_map = np.zeros((height, width, 3), dtype=np.float32)
        
        # Normalize sun direction
        sun_norm = np.array(sun_direction)
        sun_norm = sun_norm / (np.linalg.norm(sun_norm) + 1e-8)
        
        # Create gradient from sun direction
        for y in range(height):
            for x in range(width):
                # Base normal pointing up
                normal = np.array([0, 0, 1])
                
                # Perturb based on position
                dx = (x - width/2) / width * 0.5
                dy = (y - height/2) / height * 0.5
                
                # Blend with sun direction
                direction = normal + sun_norm * 0.3
                direction = direction / (np.linalg.norm(direction) + 1e-8)
                
                # Convert to 0-1 range
                direction_map[y, x] = (direction + 1) * 0.5
        
        return direction_map
    
    def _generate_ao_map(self, width: int, height: int, data: Dict[str, Any]) -> np.ndarray:
        """Generate ambient occlusion map"""
        ao_map = np.ones((height, width), dtype=np.float32)
        
        # If we have depth data, use it to estimate AO
        if 'depth_data' in data:
            depth = cv2.resize(data['depth_data'], (width, height))
            
            # Simple SSAO approximation
            kernel_size = 5
            blurred = cv2.GaussianBlur(depth, (kernel_size, kernel_size), 0)
            
            # Calculate occlusion
            occlusion = np.abs(depth - blurred)
            occlusion = 1 - np.clip(occlusion * 10, 0, 1)
            
            ao_map = occlusion
        else:
            # Fake AO - darker at edges and corners
            for y in range(height):
                for x in range(width):
                    # Distance from edges
                    edge_dist_x = min(x, width - x) / width
                    edge_dist_y = min(y, height - y) / height
                    edge_dist = min(edge_dist_x, edge_dist_y) * 2
                    
                    ao_map[y, x] = 0.5 + 0.5 * np.clip(edge_dist, 0, 1)
        
        # Convert to RGB
        return np.stack([ao_map] * 3, axis=2)
    
    def _generate_combined_map(self, width: int, height: int, sun_direction: list,
                             lighting_data: Dict[str, Any], data: Dict[str, Any]) -> np.ndarray:
        """Generate combined lighting map"""
        # Start with shadow map
        combined = self._generate_shadow_map(width, height, sun_direction, lighting_data)
        
        # Multiply by AO
        ao = self._generate_ao_map(width, height, data)
        combined = combined * ao
        
        # Add color temperature based on time of day
        time_of_day = lighting_data.get('time_of_day', 'midday')
        
        color_temp = {
            'early morning': [1.0, 0.8, 0.6],  # Warm
            'morning': [1.0, 0.9, 0.8],
            'midday': [1.0, 1.0, 1.0],  # Neutral
            'afternoon': [1.0, 0.95, 0.9],
            'evening': [1.0, 0.7, 0.5]  # Very warm
        }
        
        temp = color_temp.get(time_of_day, [1.0, 1.0, 1.0])
        for i in range(3):
            combined[:, :, i] *= temp[i]
        
        return np.clip(combined, 0, 1)
    
    def get_default_output(self) -> Tuple[torch.Tensor, Dict[str, Any], float]:
        """Return default output"""
        # Create neutral lighting map
        neutral_map = np.ones((512, 512, 3), dtype=np.float32) * 0.5
        return self.image_to_tensor(neutral_map), {"type": "lighting"}, 1.0


