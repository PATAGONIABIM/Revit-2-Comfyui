# ComfyUI_WabiSabi_Bridge/nodes/metadata_node.py
"""Metadata extraction and processing node"""

import json
import numpy as np
import torch
from typing import Dict, Any, Tuple, Optional, List
from datetime import datetime

from .base_node import WabiSabiBaseNode


class WatchMetadataNode(WabiSabiBaseNode):
    """Extract and process BIM metadata"""
    
    RETURN_TYPES = ("METADATA", "STRING", "STRING")
    RETURN_NAMES = ("metadata_dict", "prompt_text", "json_output")
    OUTPUT_NODE = False
    
    # Prompt templates for different scenarios
    PROMPT_TEMPLATES = {
        "architectural": "A {room_type} with {material_list}, {lighting_description}, {style_hints}",
        "material_focused": "Interior space featuring {primary_material} and {secondary_material}, {surface_qualities}",
        "lighting_focused": "{time_of_day} lighting, {sun_direction}, {ambient_description}",
        "detailed": "Architectural visualization of {room_type}, {material_details}, {lighting_setup}, {camera_description}",
    }
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "data_type": (["all", "materials", "lighting", "camera", "elements"], {"default": "all"}),
                "prompt_style": (["architectural", "material_focused", "lighting_focused", "detailed", "custom"], 
                               {"default": "architectural"}),
                "include_counts": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "custom_template": ("STRING", {
                    "default": "A {room_type} space with {materials}",
                    "multiline": True
                }),
                "material_mapping": ("STRING", {
                    "default": "concrete:industrial concrete\nglass:reflective glass\nwood:natural wood",
                    "multiline": True
                }),
            }
        }
    
    async def async_process(self, data_type: str, prompt_style: str,
                          include_counts: bool, custom_template: str = "",
                          material_mapping: str = "", **kwargs) -> Tuple[Dict[str, Any], str, str]:
        """Process metadata from Revit"""
        
        # Get latest data
        data = await self.get_latest_data()
        if not data:
            return self.get_default_output()
        
        # Extract metadata based on type
        metadata = {}
        
        if data_type in ["all", "materials"]:
            metadata["materials"] = await self._extract_materials(data, material_mapping)
        
        if data_type in ["all", "lighting"]:
            metadata["lighting"] = self._extract_lighting(data)
        
        if data_type in ["all", "camera"]:
            metadata["camera"] = self._extract_camera(data)
        
        if data_type in ["all", "elements"]:
            metadata["elements"] = self._extract_elements(data, include_counts)
        
        # Add general metadata
        metadata["timestamp"] = data.get("timestamp", datetime.now().isoformat())
        metadata["source"] = "WabiSabi Bridge"
        
        # Generate prompt
        prompt_text = self._generate_prompt(metadata, prompt_style, custom_template)
        
        # Create JSON output
        json_output = json.dumps(metadata, indent=2)
        
        return metadata, prompt_text, json_output
    
    async def _extract_materials(self, data: Dict[str, Any], mapping_str: str) -> Dict[str, Any]:
        """Extract material information"""
        materials_data = data.get("materials", {})
        
        # Parse material mapping
        material_map = {}
        if mapping_str:
            for line in mapping_str.strip().split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    material_map[key.strip()] = value.strip()
        
        # Process materials
        materials = {
            "list": materials_data.get("list", []),
            "primary": materials_data.get("primary", "unknown"),
            "secondary": materials_data.get("secondary", "unknown"),
            "mapped": {},
            "counts": {}
        }
        
        # Apply mapping
        for material in materials["list"]:
            mapped_name = material_map.get(material, material)
            materials["mapped"][material] = mapped_name
            
            # Count occurrences
            count = materials_data.get("counts", {}).get(material, 0)
            materials["counts"][material] = count
        
        # Determine surface qualities
        materials["qualities"] = self._analyze_material_qualities(materials["list"])
        
        return materials
    
    def _extract_lighting(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract lighting information"""
        lighting_data = data.get("lighting", {})
        
        # Process sun direction
        sun_dir = lighting_data.get("sun_direction", [0.7, 0.3, 0.6])
        sun_angle = np.arccos(sun_dir[1]) * 180 / np.pi if len(sun_dir) >= 3 else 45
        
        # Determine time of day from sun angle
        if sun_angle < 30:
            time_of_day = "early morning"
        elif sun_angle < 60:
            time_of_day = "morning"
        elif sun_angle < 90:
            time_of_day = "midday"
        elif sun_angle < 120:
            time_of_day = "afternoon"
        else:
            time_of_day = "evening"
        
        return {
            "sun_direction": sun_dir,
            "sun_angle": float(sun_angle),
            "time_of_day": time_of_day,
            "timestamp": lighting_data.get("timestamp", ""),
            "ambient_level": lighting_data.get("ambient_level", 0.5),
            "description": f"{time_of_day} lighting with sun at {sun_angle:.1f} degrees"
        }
    
    def _extract_camera(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract camera information"""
        camera_data = data.get("camera", {})
        
        # Calculate camera height and angle
        position = camera_data.get("position", [0, 0, 1.6])
        target = camera_data.get("target", [0, 0, 0])
        
        height = position[2] if len(position) > 2 else 1.6
        
        # Determine perspective type
        if height < 0.5:
            perspective = "worm's eye view"
        elif height < 1.2:
            perspective = "low angle"
        elif height < 2.0:
            perspective = "eye level"
        elif height < 5.0:
            perspective = "elevated"
        else:
            perspective = "bird's eye view"
        
        return {
            "position": position,
            "target": target,
            "fov": camera_data.get("fov", 60),
            "height": float(height),
            "perspective": perspective,
            "description": f"{perspective} from {height:.1f}m height"
        }
    
    def _extract_elements(self, data: Dict[str, Any], include_counts: bool) -> Dict[str, Any]:
        """Extract element information"""
        elements_data = data.get("elements", {})
        
        # Categorize elements
        categories = {}
        total_count = 0
        
        for element_id, element_info in elements_data.items():
            category = element_info.get("category", "other")
            if category not in categories:
                categories[category] = {
                    "count": 0,
                    "elements": []
                }
            
            categories[category]["count"] += 1
            if include_counts:
                categories[category]["elements"].append({
                    "id": element_id,
                    "name": element_info.get("name", f"Element_{element_id}")
                })
            
            total_count += 1
        
        return {
            "total_count": total_count,
            "categories": categories,
            "room_types": list(set(
                elem.get("room_type", "unknown") 
                for elem in elements_data.values() 
                if "room_type" in elem
            ))
        }
    
    def _analyze_material_qualities(self, materials: List[str]) -> List[str]:
        """Analyze material qualities for prompt generation"""
        qualities = []
        
        material_qualities = {
            "concrete": ["industrial", "raw", "brutalist"],
            "glass": ["transparent", "reflective", "modern"],
            "wood": ["warm", "natural", "organic"],
            "metal": ["sleek", "contemporary", "industrial"],
            "stone": ["solid", "timeless", "textured"],
            "fabric": ["soft", "comfortable", "acoustic"],
        }
        
        for material in materials:
            material_lower = material.lower()
            for key, quals in material_qualities.items():
                if key in material_lower:
                    qualities.extend(quals)
        
        return list(set(qualities))  # Remove duplicates
    
    def _generate_prompt(self, metadata: Dict[str, Any], style: str, custom_template: str) -> str:
        """Generate prompt text from metadata"""
        
        # Choose template
        if style == "custom" and custom_template:
            template = custom_template
        else:
            template = self.PROMPT_TEMPLATES.get(style, self.PROMPT_TEMPLATES["architectural"])
        
        # Prepare substitution values
        materials = metadata.get("materials", {})
        lighting = metadata.get("lighting", {})
        camera = metadata.get("camera", {})
        elements = metadata.get("elements", {})
        
        # Create material list
        material_list = materials.get("mapped", {}).values() if materials.get("mapped") else materials.get("list", [])
        material_list_str = ", ".join(list(material_list)[:3])  # Top 3 materials
        
        # Substitution dictionary
        subs = {
            "room_type": elements.get("room_types", ["space"])[0] if elements.get("room_types") else "architectural",
            "material_list": material_list_str,
            "materials": material_list_str,
            "primary_material": materials.get("mapped", {}).get(materials.get("primary", ""), materials.get("primary", "")),
            "secondary_material": materials.get("mapped", {}).get(materials.get("secondary", ""), materials.get("secondary", "")),
            "surface_qualities": ", ".join(materials.get("qualities", [])[:3]),
            "material_details": f"primarily {materials.get('primary', 'unknown')} surfaces",
            "lighting_description": lighting.get("description", "natural lighting"),
            "time_of_day": lighting.get("time_of_day", "daytime"),
            "sun_direction": f"sun from {'east' if lighting.get('sun_direction', [0])[0] > 0 else 'west'}",
            "ambient_description": f"{'bright' if lighting.get('ambient_level', 0.5) > 0.6 else 'soft'} ambient light",
            "lighting_setup": lighting.get("description", ""),
            "camera_description": camera.get("description", "standard view"),
            "style_hints": ", ".join(materials.get("qualities", [])[:2]) + " aesthetic"
        }
        
        # Format template
        try:
            prompt = template.format(**subs)
        except KeyError as e:
            # If template has unknown keys, use a simple fallback
            prompt = f"Architectural visualization with {material_list_str}, {lighting.get('description', 'natural lighting')}"
        
        return prompt
    
    def get_default_output(self) -> Tuple[Dict[str, Any], str, str]:
        """Return default output"""
        default_metadata = {
            "error": "No metadata available",
            "timestamp": datetime.now().isoformat()
        }
        return default_metadata, "No data available", json.dumps(default_metadata, indent=2)


