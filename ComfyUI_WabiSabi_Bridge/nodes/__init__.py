
# ComfyUI_WabiSabi_Bridge/nodes/__init__.py
"""Node definitions for WabiSabi Bridge"""

from .watch_image_node import WatchImageProNode
from .batch_processor_node import BatchImageProcessorNode
from .segmentation_node import SmartSegmentationNode

# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "WatchImagePro": WatchImageProNode,
    "BatchImageProcessor": BatchImageProcessorNode,
    "SmartSegmentation": SmartSegmentationNode,
}

# Display names for nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "WatchImagePro": "Watch Image Pro 🌉",
    "BatchImageProcessor": "Batch Image Processor 🌉",
    "SmartSegmentation": "Smart Segmentation 🌉",
}

# Add import statement for JSON
import json