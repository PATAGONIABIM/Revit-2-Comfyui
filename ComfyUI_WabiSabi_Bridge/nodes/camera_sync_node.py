
# ComfyUI_WabiSabi_Bridge/nodes/camera_sync_node.py
"""Camera synchronization node"""

import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import json

from .base_node import WabiSabiBaseNode


class CameraSyncNode(WabiSabiBaseNode):
    """Synchronize camera between Revit and ComfyUI"""
    
    RETURN_TYPES = ("STRING", "STRING", "FLOAT", "BOOLEAN")
    RETURN_NAMES = ("camera_params", "view_matrix", "fov", "sync_active")
    OUTPUT_NODE = True
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "sync_mode": (["read_only", "write_back", "bidirectional"], {"default": "read_only"}),
                "interpolation": (["none", "linear", "smooth"], {"default": "linear"}),
                "sync_enabled": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "target_position": ("STRING", {"default": ""}),
                "target_rotation": ("STRING", {"default": ""}),
                "target_fov": ("FLOAT", {"default": 60.0, "min": 10.0, "max": 120.0}),
                "interpolation_speed": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 1.0}),
                "save_preset": ("BOOLEAN", {"default": False}),
                "preset_name": ("STRING", {"default": ""}),
                "load_preset": ("STRING", {"default": ""}),
            }
        }
    
    def __init__(self):
        super().__init__()
        self.camera_presets = {}
        self.current_camera = {
            "position": [0, 0, 1.6],
            "target": [0, 0, 0],
            "fov": 60.0
        }
        self.interpolation_state = None
    
    async def async_process(self, sync_mode: str, interpolation: str, sync_enabled: bool,
                          target_position: str = "", target_rotation: str = "",
                          target_fov: float = 60.0, interpolation_speed: float = 0.1,
                          save_preset: bool = False, preset_name: str = "",
                          load_preset: str = "", **kwargs) -> Tuple[str, str, float, bool]:
        """Process camera synchronization"""
        
        if not sync_enabled:
            return self._format_output(self.current_camera, False)
        
        # Get current camera data from Revit
        data = await self.get_latest_data()
        
        if data and 'camera' in data:
            revit_camera = data['camera']
            
            # Update current camera with interpolation
            if interpolation != "none":
                self.current_camera = self._interpolate_camera(
                    self.current_camera, revit_camera, interpolation, interpolation_speed
                )
            else:
                self.current_camera = revit_camera.copy()
        
        # Handle preset operations
        if save_preset and preset_name:
            self.camera_presets[preset_name] = self.current_camera.copy()
        
        if load_preset and load_preset in self.camera_presets:
            target_camera = self.camera_presets[load_preset]
            
            if sync_mode in ["write_back", "bidirectional"]:
                # Send camera update to Revit
                await self._send_camera_update(target_camera)
            
            self.current_camera = target_camera.copy()
        
        # Handle manual target setting
        if target_position or target_rotation:
            manual_camera = self._parse_manual_camera(
                target_position, target_rotation, target_fov
            )
            
            if sync_mode in ["write_back", "bidirectional"]:
                await self._send_camera_update(manual_camera)
            
            if sync_mode != "read_only":
                self.current_camera = manual_camera
        
        return self._format_output(self.current_camera, sync_enabled)
    
    def _interpolate_camera(self, current: Dict[str, Any], target: Dict[str, Any],
                          mode: str, speed: float) -> Dict[str, Any]:
        """Interpolate between camera positions"""
        
        result = current.copy()
        
        # Initialize interpolation state if needed
        if self.interpolation_state is None:
            self.interpolation_state = {
                "position": np.array(current["position"]),
                "target": np.array(current["target"]),
                "fov": current["fov"]
            }
        
        # Target values
        target_pos = np.array(target.get("position", current["position"]))
        target_tgt = np.array(target.get("target", current["target"]))
        target_fov = target.get("fov", current["fov"])
        
        if mode == "linear":
            # Linear interpolation
            self.interpolation_state["position"] += (target_pos - self.interpolation_state["position"]) * speed
            self.interpolation_state["target"] += (target_tgt - self.interpolation_state["target"]) * speed
            self.interpolation_state["fov"] += (target_fov - self.interpolation_state["fov"]) * speed
        
        elif mode == "smooth":
            # Smooth interpolation with easing
            ease_speed = 1 - (1 - speed) ** 3  # Cubic easing
            self.interpolation_state["position"] += (target_pos - self.interpolation_state["position"]) * ease_speed
            self.interpolation_state["target"] += (target_tgt - self.interpolation_state["target"]) * ease_speed
            self.interpolation_state["fov"] += (target_fov - self.interpolation_state["fov"]) * ease_speed
        
        # Update result
        result["position"] = self.interpolation_state["position"].tolist()
        result["target"] = self.interpolation_state["target"].tolist()
        result["fov"] = float(self.interpolation_state["fov"])
        
        return result
    
    def _parse_manual_camera(self, position_str: str, rotation_str: str, fov: float) -> Dict[str, Any]:
        """Parse manual camera input"""
        camera = self.current_camera.copy()
        
        # Parse position
        if position_str:
            try:
                camera["position"] = [float(x.strip()) for x in position_str.split(',')][:3]
            except:
                pass
        
        # Parse rotation/target
        if rotation_str:
            try:
                # Could be euler angles or target position
                values = [float(x.strip()) for x in rotation_str.split(',')][:3]
                
                # Assume it's a target position for now
                camera["target"] = values
            except:
                pass
        
        camera["fov"] = fov
        
        return camera
    
    async def _send_camera_update(self, camera: Dict[str, Any]) -> bool:
        """Send camera update to Revit"""
        if not self.channel_manager:
            return False
        
        channel = self.channel_manager.get_channel()
        if not channel:
            return False
        
        command = {
            "type": "command",
            "command": "set_camera",
            "parameters": camera
        }
        
        return await channel.write_data(command)
    
    def _calculate_view_matrix(self, camera: Dict[str, Any]) -> np.ndarray:
        """Calculate view matrix from camera parameters"""
        position = np.array(camera["position"])
        target = np.array(camera["target"])
        up = np.array([0, 0, 1])  # Assume Z-up
        
        # Calculate camera basis
        forward = target - position
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        
        right = np.cross(forward, up)
        right = right / (np.linalg.norm(right) + 1e-8)
        
        up = np.cross(right, forward)
        
        # Build view matrix
        view_matrix = np.eye(4)
        view_matrix[:3, 0] = right
        view_matrix[:3, 1] = up
        view_matrix[:3, 2] = -forward
        view_matrix[:3, 3] = position
        
        return view_matrix
    
    def _format_output(self, camera: Dict[str, Any], sync_active: bool) -> Tuple[str, str, float, bool]:
        """Format output values"""
        # Camera parameters as JSON
        camera_params = json.dumps(camera, indent=2)
        
        # View matrix as string
        view_matrix = self._calculate_view_matrix(camera)
        matrix_str = np.array2string(view_matrix, precision=3, suppress_small=True)
        
        return camera_params, matrix_str, camera["fov"], sync_active
    
    def get_default_output(self) -> Tuple[str, str, float, bool]:
        """Return default output"""
        default_camera = {
            "position": [0, 0, 1.6],
            "target": [0, 0, 0],
            "fov": 60.0
        }
        return self._format_output(default_camera, False)


# Update the node mappings
from .metadata_node import WatchMetadataNode
from .lighting_control_node import LightingControlNode
from .command_node import SendCommandNode
from .camera_sync_node import CameraSyncNode

# Add to existing mappings
NODE_CLASS_MAPPINGS.update({
    "WatchMetadata": WatchMetadataNode,
    "LightingControl": LightingControlNode,
    "SendCommand": SendCommandNode,
    "CameraSync": CameraSyncNode,
})

NODE_DISPLAY_NAME_MAPPINGS.update({
    "WatchMetadata": "Watch Metadata 🌉",
    "LightingControl": "Lighting Control 🌉",
    "SendCommand": "Send Command 🌉",
    "CameraSync": "Camera Sync 🌉",
})