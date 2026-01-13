"""
ComfyUI WabiSabi Bridge
Plugin para conectar Autodesk Revit con ComfyUI en tiempo real
"""

# Importar los nodos del módulo principal
from .wabisabi_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Información del plugin
__version__ = "0.1.0"
__author__ = "WabiSabi Bridge Team"

# Mensaje de bienvenida
print(f"[WabiSabi Bridge] Inicializando plugin v{__version__}")
print("[WabiSabi Bridge] Conectando Revit con ComfyUI...")

# Exportar para ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']