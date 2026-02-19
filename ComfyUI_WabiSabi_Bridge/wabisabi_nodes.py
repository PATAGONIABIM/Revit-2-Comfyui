import os
import json
import time
import mmap
import struct
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

import numpy as np
from PIL import Image
import torch

# ComfyUI imports
import folder_paths
import comfy.utils

# =============================================================================
# UTILIDADES (Sin cambios)
# =============================================================================

def load_image_as_tensor(filepath: str) -> Optional[torch.Tensor]:
    """Carga una imagen y la convierte en tensor de ComfyUI (BHWC)"""
    try:
        for _ in range(3):
            try:
                img = Image.open(filepath)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img_np = np.array(img).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_np)[None,]
                return img_tensor
            except IOError:
                time.sleep(0.05)
        raise IOError(f"No se pudo leer el archivo de imagen después de varios intentos: {filepath}")
    except Exception as e:
        print(f"[WabiSabi] Error cargando la imagen {filepath}: {e}")
        return None

def load_metadata(metadata_path: str) -> Dict[str, Any]:
    """Carga el archivo de metadatos JSON"""
    try:
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"[WabiSabi] Error cargando los metadatos: {e}")
    return {}

# =============================================================================
# CLASE BASE PARA NODOS DE VIGILANCIA DE IMAGEN (Legacy File-Based)
# =============================================================================

class WabiSabiWatcherBase:
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "timestamp")
    CATEGORY = "WabiSabi Bridge/Legacy"

    # La lógica de detección de cambios es la misma para todos
    @classmethod
    def IS_CHANGED(cls, folder_path, auto_reload, trigger=0, filename=""):
        if not auto_reload:
            return str(trigger)
        
        filepath = os.path.join(folder_path, filename)
        if not os.path.exists(filepath):
            return float("NaN")
            
        stats = os.stat(filepath)
        return f"{stats.st_mtime}-{stats.st_size}"

    # La lógica de carga es la misma, solo cambia el nombre del archivo
    def load_watched_file(self, folder_path, auto_reload, trigger, filename):
        filepath = os.path.join(folder_path, filename)
        
        if not os.path.exists(filepath):
            print(f"[WabiSabi] Archivo no encontrado: {filepath}")
            # Devuelve un tensor vacío del tamaño correcto
            return (torch.zeros((1, 512, 512, 3)), "File not found")

        print(f"[WabiSabi] Cambio detectado. Cargando: {filename}")
        image_tensor = load_image_as_tensor(filepath)
        
        if image_tensor is None:
            return (torch.zeros((1, 512, 512, 3)), "Error loading")
        
        timestamp = datetime.fromtimestamp(os.path.getmtime(filepath)).strftime("%Y-%m-%d %H:%M:%S")
        return (image_tensor, timestamp)

# =============================================================================
# LECTOR DE MEMORIA COMPARTIDA (MMF)
# =============================================================================

class WabiSabiMMFReader:
    def __init__(self, map_name="WabiSabiBridge_ImageStream"):
        self.map_name = map_name
        # No persistimos el mmap para evitar bloqueos en Windows
        # self.mmf = None 
        
    def read(self, channel_type: str) -> Tuple[Optional[torch.Tensor], int]:
        """Lee un canal específico desde la memoria compartida."""
        try:
            # Abrir MMF fresco en cada lectura para evitar bloqueos
            # IMPORTANTE: Usamos 'access=mmap.ACCESS_READ' para no bloquear escrituras del C++
            with mmap.mmap(-1, 1024*1024*256, self.map_name, access=mmap.ACCESS_READ) as mm:
                
                # Asegurarse de leer desde el inicio del MMF
                mm.seek(0)
                
                # Leer Header (Struct C++: 4 int32, 2 int64, 4 int64, 4 int32)
                # Total Header Size: 4*2 + 8*2 + 8*4 + 4*4 = 8 + 16 + 32 + 16 = 72 bytes
                # Estructura: width, height, timestamp, seq, offsets(4), sizes(4)
                
                header_data = mm.read(72)
                if len(header_data) < 72:
                    return None, 0 # Return 0 for sequence if header is incomplete
                    
                unpacked = struct.unpack('<iiqqqqqqiiii', header_data)
                
                # Desempaquetar
                width, height, timestamp, seq_num = unpacked[0:4]
                offsets_tuple = unpacked[4:8]
                sizes_tuple = unpacked[8:12]
                
                # Mapear channel type a index
                idx = -1
                if channel_type == "depth": idx = 0
                elif channel_type == "normal": idx = 1
                elif channel_type == "lines": idx = 2
                elif channel_type == "segmentation": idx = 3
                
                if idx == -1: 
                    print(f"[WabiSabi MMF] Canal desconocido: {channel_type}")
                    return None, 0
                
                offset = offsets_tuple[idx]
                size = sizes_tuple[idx]
                
                if size <= 0:
                    # Canal vacío (aún no renderizado o deshabilitado)
                    return None, seq_num
                    
                # Leer datos RAW
                mm.seek(offset)
                raw_data = mm.read(size)
            
            # --- PROCESAMIENTO FUERA DEL CONTEXTO MMAP (SAFE) ---
            
            if channel_type == "depth" or channel_type == "lines":
                # Float monocromático (H, W, 1)
                array = np.frombuffer(raw_data, dtype=np.float32).reshape(height, width, 1)
                
                if channel_type == "depth":
                    # --- FIX DEPTH ISSUES (ROBUST) ---
                    # 1. Crear mascara de infinitos
                    infinity_mask = array > 1e10
                    
                    # 2. Extraer pixels válidos para calcular rango
                    valid_pixels = array[~infinity_mask]
                    
                    if valid_pixels.size > 0:
                        min_val = valid_pixels.min()
                        max_val = valid_pixels.max()
                        
                        # Evitar división por cero
                        range_val = max_val - min_val
                        if range_val < 1e-6: range_val = 1.0
                        
                        # 3. Normalizar al rango 0..1 (Maximizar contraste)
                        norm_array = np.zeros_like(array)
                        norm_array[~infinity_mask] = (array[~infinity_mask] - min_val) / range_val
                        
                        # 4. Invertir (Blanco=Cerca, Negro=Lejos) y poner Fondo=Negro
                        norm_array = 1.0 - norm_array
                        norm_array[infinity_mask] = 0.0 
                        
                        array = norm_array
                        # print(f"[DEPTH AUTO-NORM] Range: {min_val:.1f}m to {max_val:.1f}m -> Mapped to 1.0-0.0")
                    else:
                        array = np.zeros_like(array)
                        
                # Duplicar canales para RGB (H, W, 3)
                array = np.repeat(array, 3, axis=2) 
            
            elif channel_type == "normal":
                # Float3 (H, W, 3)
                array = np.frombuffer(raw_data, dtype=np.float32).reshape(height, width, 3)
                # Normalizar de -1..1 a 0..1
                array = array * 0.5 + 0.5
                
            elif channel_type == "segmentation":
                # Float4 (H, W, 4) - RGBA
                array = np.frombuffer(raw_data, dtype=np.float32).reshape(height, width, 4)
                # Descartar Alpha para ComfyUI (RGB)
                array = array[:, :, :3]
                
            else:
                return None, seq_num
            
            # --- CORRECCIÓN DE ORIENTACIÓN ---
            array = np.flipud(array)
            
            # COPIA para asegurar memoria contigua y escribible
            array = array.copy()
            
            # Normalizar y convertir a Tensor
            image_tensor = torch.from_numpy(array)[None,] # Añadir batch dim
            
            return image_tensor, seq_num
            
        except FileNotFoundError:
            return None, 0
        except Exception as e:
            print(f"[WabiSabi MMF] Error crítico en read: {e}")
            return None, 0

# =============================================================================
# NUEVO NODO DE LECTURA MMF
# =============================================================================

class LoadImageFromMMFNode:
    """Nodo ultra-rápido que lee imágenes desde la memoria RAM (Shared Memory)."""
    
    def __init__(self):
        self.reader = WabiSabiMMFReader()
        self.last_sequence = -1
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "channel": (["depth", "normal", "lines", "segmentation"],),
                "auto_reload": ("BOOLEAN", {"default": True}),
            },
            "optional": { "trigger": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}), }
        }
    
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("image", "sequence_number")
    CATEGORY = "WabiSabi Bridge"
    FUNCTION = "load_mmf"

    @classmethod
    def IS_CHANGED(cls, channel, auto_reload, trigger=0):
        # Esta función le dice a ComfyUI si el nodo debe re-ejecutarse.
        # Leemos el SequenceNumber del MMF.
        try:
            # Header size 72 bytes
            # int32 width, int32 height, int64 timestamp, int64 sequence
            # sequence está en offset 4+4+8 = 16
            
            # Usar modo seguro (Open/Close) para no bloquear
            with mmap.mmap(-1, 72, "WabiSabiBridge_ImageStream", access=mmap.ACCESS_READ) as mm:
                mm.seek(16)
                seq_bytes = mm.read(8)
                sequence = struct.unpack('<q', seq_bytes)[0]
                
                # Descomentar para debug extremo si no actualiza
                print(f"[MMF POLL] Seq: {sequence}") 
                return float(sequence)
        except FileNotFoundError:
             return float(0.0) # Si no existe, no ha cambiado (estado 0)
        except Exception as e:
            # print(f"[MMF POLL ERROR] {e}")
            return float(-1.0) # Error state

    def load_mmf(self, channel, auto_reload, trigger=0):
        image, seq = self.reader.read(channel)
        
        if image is None:
            # Retornar imagen negra si falla
            return (torch.zeros((1, 512, 512, 3)), seq)
            
        return (image, seq)


# =============================================================================
# NODOS LEGACY (FILE BASED)
# =============================================================================

class WatchImageNode(WabiSabiWatcherBase):
    """Nodo que vigila cambios en un archivo de imagen (render o depth)."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": os.path.join(os.path.expanduser("~"), "Documents", "WabiSabiBridge")}),
                "image_type": (["render", "depth"],),
                "auto_reload": ("BOOLEAN", {"default": True}),
            },
            "optional": { "trigger": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}), }
        }
    
    FUNCTION = "load_watched_image"

    @classmethod
    def IS_CHANGED(cls, folder_path, image_type, auto_reload, trigger=0):
        filename = f"current_{image_type}.png"
        return super().IS_CHANGED(folder_path, auto_reload, trigger, filename)

    def load_watched_image(self, folder_path, image_type, auto_reload, trigger=0):
        filename = f"current_{image_type}.png"
        return self.load_watched_file(folder_path, auto_reload, trigger, filename)

class WatchLinesNode(WabiSabiWatcherBase):
    FUNCTION = "load_lines"
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { "folder_path": ("STRING", {"default": "..."}), "auto_reload": ("BOOLEAN", {"default": True}) },
            "optional": { "trigger": ("INT", {"default": 0}), }
        }
    @classmethod
    def IS_CHANGED(cls, folder_path, auto_reload, trigger=0):
        return super().IS_CHANGED(folder_path, auto_reload, trigger, filename="current_lines.png")
    def load_lines(self, folder_path, auto_reload, trigger=0):
        return self.load_watched_file(folder_path, auto_reload, trigger, filename="current_lines.png")

class WatchNormalsNode(WabiSabiWatcherBase):
    FUNCTION = "load_normals"
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { "folder_path": ("STRING", {"default": "..."}), "auto_reload": ("BOOLEAN", {"default": True}) },
             "optional": { "trigger": ("INT", {"default": 0}), }
        }
    @classmethod
    def IS_CHANGED(cls, folder_path, auto_reload, trigger=0):
        return super().IS_CHANGED(folder_path, auto_reload, trigger, filename="current_normals.png")
    def load_normals(self, folder_path, auto_reload, trigger=0):
        return self.load_watched_file(folder_path, auto_reload, trigger, filename="current_normals.png")

class WatchSegmentationNode(WabiSabiWatcherBase):
    FUNCTION = "load_segmentation"
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { "folder_path": ("STRING", {"default": "..."}), "auto_reload": ("BOOLEAN", {"default": True}) },
             "optional": { "trigger": ("INT", {"default": 0}), }
        }
    @classmethod
    def IS_CHANGED(cls, folder_path, auto_reload, trigger=0):
        return super().IS_CHANGED(folder_path, auto_reload, trigger, filename="current_segmentation.png")
    def load_segmentation(self, folder_path, auto_reload, trigger=0):
        return self.load_watched_file(folder_path, auto_reload, trigger, filename="current_segmentation.png")


class WatchMetadataNode:
    """Nodo que vigila y carga los metadatos exportados por Revit."""
    @classmethod
    def INPUT_TYPES(cls):
        return { "required": {
                    "folder_path": ("STRING", {"default": os.path.join(os.path.expanduser("~"), "Documents", "WabiSabiBridge")}),
                    "auto_reload": ("BOOLEAN", {"default": True}),
                },
                "optional": { "trigger": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}), }
            }
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("metadata_json", "view_name", "timestamp")
    FUNCTION = "watch_metadata"
    CATEGORY = "WabiSabi Bridge"

    @classmethod
    def IS_CHANGED(cls, folder_path, auto_reload, trigger=0):
        if not auto_reload: return str(trigger)
        metadata_path = os.path.join(folder_path, "current_metadata.json")
        if not os.path.exists(metadata_path): return float("NaN")
        stats = os.stat(metadata_path)
        return f"{stats.st_mtime}-{stats.st_size}"
    
    def watch_metadata(self, folder_path, auto_reload, trigger=0):
        metadata_path = os.path.join(folder_path, "current_metadata.json")
        if not os.path.exists(metadata_path):
            return ("", "No metadata", "")
        # print("[WabiSabi] Cambio detectado. Cargando: current_metadata.json")
        metadata = load_metadata(metadata_path)
        view_name = metadata.get("view_name", "Unknown View")
        timestamp = metadata.get("timestamp", "")
        metadata_json = json.dumps(metadata, indent=2)
        return (metadata_json, view_name, timestamp)

class CombineRenderDepthNode:
    """Nodo utilidad que combina render y depth map en una sola imagen."""
    @classmethod
    def INPUT_TYPES(cls):
        return { "required": { "render_image": ("IMAGE",), "depth_image": ("IMAGE",), } }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "combine_images"
    CATEGORY = "WabiSabi Bridge"
    
    def combine_images(self, render_image, depth_image):
        if render_image.shape[1:] != depth_image.shape[1:]:
             depth_image = comfy.utils.common_upscale(depth_image.movedim(-1,1), render_image.shape[2], render_image.shape[1], "bilinear", "center").movedim(1,-1)
        if depth_image.shape[3] == 3: depth_gray = depth_image.mean(dim=3, keepdim=True)
        else: depth_gray = depth_image
        combined = torch.cat((render_image, depth_gray), dim=-1)
        return (combined,)

# =============================================================================
# REGISTRO DE NODOS
# =============================================================================

NODE_CLASS_MAPPINGS = {
    # Nuevo nodo MMF
    "LoadImageFromMMF": LoadImageFromMMFNode,
    
    # Nodos Legacy
    "WatchImage": WatchImageNode,
    "WatchMetadata": WatchMetadataNode,
    "CombineRenderDepth": CombineRenderDepthNode,
    "WatchLines": WatchLinesNode,
    "WatchNormals": WatchNormalsNode,
    "WatchSegmentation": WatchSegmentationNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageFromMMF": "Load from MMF (WabiSabi)", # <--- Nombre corto
    
    "WatchImage": "Watch Image (File)",
    "WatchMetadata": "Watch Metadata (File)",
    "CombineRenderDepth": "Combine Render+Depth",
    "WatchLines": "Watch Lines (File)",
    "WatchNormals": "Watch Normals (File)",
    "WatchSegmentation": "Watch Segmentation (File)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("[WabiSabi Bridge] Nodos cargados exitosamente - v0.3.5 (Incluye LoadFromMMF)")
