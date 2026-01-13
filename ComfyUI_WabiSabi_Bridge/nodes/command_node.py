# ComfyUI_WabiSabi_Bridge/nodes/command_node.py
"""Command sending node for bidirectional control"""

import json
import asyncio
from typing import Dict, Any, Tuple, Optional
import time

from .base_node import WabiSabiBaseNode


class SendCommandNode(WabiSabiBaseNode):
    """Send commands back to Revit"""
    
    RETURN_TYPES = ("BOOLEAN", "STRING", "FLOAT")
    RETURN_NAMES = ("success", "response", "execution_time")
    OUTPUT_NODE = True
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "command": (["change_view", "update_material", "toggle_element", 
                           "set_camera", "refresh_view", "custom"], {"default": "change_view"}),
                "trigger": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "view_name": ("STRING", {"default": ""}),
                "element_id": ("STRING", {"default": ""}),
                "material_name": ("STRING", {"default": ""}),
                "camera_position": ("STRING", {"default": "0,0,1.6"}),
                "camera_target": ("STRING", {"default": "0,0,0"}),
                "custom_command": ("STRING", {"default": "{}"}),
                "wait_for_response": ("BOOLEAN", {"default": True}),
                "timeout": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 30.0}),
            }
        }
    
    async def async_process(self, command: str, trigger: bool,
                          view_name: str = "", element_id: str = "",
                          material_name: str = "", camera_position: str = "0,0,1.6",
                          camera_target: str = "0,0,0", custom_command: str = "{}",
                          wait_for_response: bool = True, timeout: float = 5.0,
                          **kwargs) -> Tuple[bool, str, float]:
        """Send command to Revit"""
        
        if not trigger:
            return False, "Not triggered", 0.0
        
        # Initialize if needed
        if not await self.initialize_if_needed():
            return False, "Failed to initialize", 0.0
        
        channel = self.channel_manager.get_channel()
        if not channel:
            return False, "No channel available", 0.0
        
        # Build command data
        command_data = self._build_command(command, view_name, element_id, 
                                         material_name, camera_position, 
                                         camera_target, custom_command)
        
        # Send command
        start_time = time.time()
        
        try:
            success = await channel.write_data(command_data)
            
            if not success:
                return False, "Failed to send command", time.time() - start_time
            
            # Wait for response if requested
            if wait_for_response:
                response = await self._wait_for_response(command_data['id'], timeout)
            else:
                response = "Command sent (no response requested)"
            
            execution_time = time.time() - start_time
            
            return True, response, execution_time
            
        except Exception as e:
            return False, f"Error: {str(e)}", time.time() - start_time
    
    def _build_command(self, command: str, view_name: str, element_id: str,
                      material_name: str, camera_position: str, camera_target: str,
                      custom_command: str) -> Dict[str, Any]:
        """Build command data structure"""
        
        command_id = f"cmd_{int(time.time() * 1000)}"
        
        if command == "change_view":
            return {
                "id": command_id,
                "type": "command",
                "command": "change_view",
                "parameters": {
                    "view_name": view_name
                }
            }
        
        elif command == "update_material":
            return {
                "id": command_id,
                "type": "command",
                "command": "update_material",
                "parameters": {
                    "element_id": element_id,
                    "material_name": material_name
                }
            }
        
        elif command == "toggle_element":
            return {
                "id": command_id,
                "type": "command",
                "command": "toggle_element",
                "parameters": {
                    "element_id": element_id,
                    "visible": True  # Could be made configurable
                }
            }
        
        elif command == "set_camera":
            # Parse position and target
            try:
                pos = [float(x.strip()) for x in camera_position.split(',')]
                target = [float(x.strip()) for x in camera_target.split(',')]
            except:
                pos = [0, 0, 1.6]
                target = [0, 0, 0]
            
            return {
                "id": command_id,
                "type": "command",
                "command": "set_camera",
                "parameters": {
                    "position": pos,
                    "target": target
                }
            }
        
        elif command == "refresh_view":
            return {
                "id": command_id,
                "type": "command",
                "command": "refresh_view",
                "parameters": {}
            }
        
        elif command == "custom":
            try:
                custom_data = json.loads(custom_command)
                custom_data["id"] = command_id
                custom_data["type"] = "command"
                return custom_data
            except:
                return {
                    "id": command_id,
                    "type": "command",
                    "command": "invalid",
                    "error": "Invalid custom command JSON"
                }
        
        return {
            "id": command_id,
            "type": "command",
            "command": "unknown"
        }
    
    async def _wait_for_response(self, command_id: str, timeout: float) -> str:
        """Wait for command response"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            data = await self.get_latest_data(timeout=0.1)
            
            if data and data.get('type') == 'response' and data.get('command_id') == command_id:
                if data.get('success'):
                    return f"Success: {data.get('message', 'Command executed')}"
                else:
                    return f"Failed: {data.get('error', 'Unknown error')}"
            
            await asyncio.sleep(0.1)
        
        return "Timeout waiting for response"
    
    def get_default_output(self) -> Tuple[bool, str, float]:
        """Return default output"""
        return False, "No command sent", 0.0

