
# ComfyUI_WabiSabi_Bridge/nodes/preview_widget.py
"""Live preview widget for ComfyUI UI"""

import json
from typing import Dict, Any, Optional
import numpy as np
import base64
from io import BytesIO
from PIL import Image


class PreviewWidget:
    """Widget for live preview in ComfyUI"""
    
    @staticmethod
    def create_widget_html() -> str:
        """Create HTML for the preview widget"""
        return '''
        <div class="wabisabi-preview-widget">
            <div class="preview-header">
                <span class="status-indicator" id="ws-status">●</span>
                <span class="status-text">Disconnected</span>
                <span class="latency">--ms</span>
            </div>
            <div class="preview-container">
                <img id="preview-image" src="" style="max-width: 100%; height: auto;">
                <div class="preview-overlay">
                    <div class="fps-counter">-- FPS</div>
                </div>
            </div>
            <div class="preview-controls">
                <button onclick="togglePreview()">Toggle Preview</button>
                <select id="preview-quality">
                    <option value="low">Low Quality</option>
                    <option value="medium" selected>Medium Quality</option>
                    <option value="high">High Quality</option>
                </select>
            </div>
        </div>
        
        <style>
        .wabisabi-preview-widget {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 4px;
            padding: 8px;
            margin: 4px 0;
        }
        
        .preview-header {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 8px;
            font-size: 12px;
        }
        
        .status-indicator {
            font-size: 10px;
        }
        
        .status-indicator.connected { color: #4caf50; }
        .status-indicator.connecting { color: #ff9800; }
        .status-indicator.disconnected { color: #f44336; }
        
        .preview-container {
            position: relative;
            background: #000;
            min-height: 100px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .preview-overlay {
            position: absolute;
            top: 4px;
            right: 4px;
            background: rgba(0,0,0,0.7);
            padding: 2px 6px;
            border-radius: 2px;
            font-size: 10px;
            color: #fff;
        }
        
        .preview-controls {
            margin-top: 8px;
            display: flex;
            gap: 8px;
        }
        
        .preview-controls button, .preview-controls select {
            flex: 1;
            padding: 4px;
            background: #333;
            border: 1px solid #555;
            color: #fff;
            border-radius: 2px;
            font-size: 12px;
        }
        </style>
        
        <script>
        let previewEnabled = true;
        let ws = null;
        let fpsCounter = 0;
        let lastFpsUpdate = Date.now();
        
        function togglePreview() {
            previewEnabled = !previewEnabled;
            if (!previewEnabled) {
                document.getElementById('preview-image').src = '';
            }
        }
        
        function updatePreview(data) {
            if (!previewEnabled) return;
            
            // Update image
            if (data.image) {
                document.getElementById('preview-image').src = 'data:image/jpeg;base64,' + data.image;
            }
            
            // Update FPS
            fpsCounter++;
            const now = Date.now();
            if (now - lastFpsUpdate > 1000) {
                document.querySelector('.fps-counter').textContent = fpsCounter + ' FPS';
                fpsCounter = 0;
                lastFpsUpdate = now;
            }
            
            // Update latency
            if (data.latency) {
                document.querySelector('.latency').textContent = data.latency.toFixed(1) + 'ms';
            }
        }
        
        function connectWebSocket() {
            ws = new WebSocket('ws://localhost:8188/ws/wabisabi-preview');
            
            ws.onopen = () => {
                document.getElementById('ws-status').className = 'status-indicator connected';
                document.querySelector('.status-text').textContent = 'Connected';
            };
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                updatePreview(data);
            };
            
            ws.onclose = () => {
                document.getElementById('ws-status').className = 'status-indicator disconnected';
                document.querySelector('.status-text').textContent = 'Disconnected';
                // Reconnect after 2 seconds
                setTimeout(connectWebSocket, 2000);
            };
        }
        
        // Connect on load
        connectWebSocket();
        </script>
        '''
    
    @staticmethod
    def create_preview_data(image: np.ndarray, quality: str = "medium", latency: float = 0) -> Dict[str, Any]:
        """Create preview data for sending to UI"""
        # Resize based on quality
        quality_map = {
            "low": (160, 90),
            "medium": (320, 180),
            "high": (640, 360)
        }
        
        target_size = quality_map.get(quality, quality_map["medium"])
        
        # Convert to PIL Image
        if image.dtype != np.uint8:
            image = (image * 255).clip(0, 255).astype(np.uint8)
        
        pil_image = Image.fromarray(image)
        
        # Resize maintaining aspect ratio
        pil_image.thumbnail(target_size, Image.Resampling.LANCZOS)
        
        # Convert to base64
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG", quality=85)
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return {
            "image": img_base64,
            "latency": latency,
            "timestamp": time.time()
        }