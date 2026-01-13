import mmap
import struct
import numpy as np
import os
import time
from PIL import Image

# Configuración
MMF_NAME = "WabiSabiBridge_ImageStream"
HEADER_SIZE = 72
WIDTH = 1920 # Ajustar si es necesario, aunque el script leerá del header
HEIGHT = 1080

def read_mmf():
    print(f"Intentando abrir MMF: {MMF_NAME}...")
    
    # PASO 1: Leer solo el encabezado para obtener dimensiones y tamaños
    try:
        # En Windows con -1 (memoria compartida), size debe ser > 0
        mm_header = mmap.mmap(-1, HEADER_SIZE, MMF_NAME, access=mmap.ACCESS_READ)
    except FileNotFoundError:
        print("ERROR: No se encontró el MMF. Asegúrate de que WabiSabiRenderer.exe esté ejecutándose.")
        return
    except OSError as e:
        print(f"ERROR SO al abrir MMF (Header): {e}")
        return

    # Leer encabezado
    header_data = mm_header.read(HEADER_SIZE)
    mm_header.close() # Cerramos para reabrir con el tamaño completo después

    # Desempaquetar
    try:
        unpacked = struct.unpack('<iiqqqqqqiiii', header_data)
        width = unpacked[0]
        height = unpacked[1]
        timestamp = unpacked[2]
        sequence = unpacked[3]
        
        offsets = {
            'depth': unpacked[4],
            'normal': unpacked[5],
            'lines': unpacked[6],
            'segmentation': unpacked[7]
        }
        sizes = {
            'depth': unpacked[8],
            'normal': unpacked[9],
            'lines': unpacked[10],
            'segmentation': unpacked[11]
        }
    except struct.error as e:
        print(f"Error desempaquetando header: {e}")
        return

    print(f"\n--- Detectado: Seq {sequence} | {width}x{height} ---")
    
    # Calcular tamaño total necesario
    # El offset más lejano + su tamaño
    total_size = HEADER_SIZE
    max_end = 0
    for ch, off in offsets.items():
        if sizes[ch] > 0:
            end_pos = off + sizes[ch]
            if end_pos > max_end:
                max_end = end_pos
    
    if max_end > total_size:
        total_size = max_end
        
    print(f"Tamaño total del MMF estimado: {total_size} bytes")

    # PASO 2: Abrir el MMF con el tamaño completo para leer datos
    try:
        mm = mmap.mmap(-1, total_size, MMF_NAME, access=mmap.ACCESS_READ)
    except OSError as e:
        print(f"ERROR: No se pudo abrir el MMF con tamaño {total_size}: {e}")
        return

    while True:
        # Re-leer header del MMF abierto para detectar cambios
        mm.seek(0)
        current_header = mm.read(HEADER_SIZE)
        curr_seq = struct.unpack('<q', current_header[16:24])[0] # Offset 16 es timestamp, 24 es seq? No, timestamp(8)+width(4)+height(4).. espera
        # struct: w(4), h(4), time(8), seq(8)... 
        # offsets: 0-4(w), 4-8(h), 8-16(time), 16-24(seq) -> Correcto, seq empieza en 16 si alignment es packed?
        # SharedMemory.h: int32, int32, int64, int64. 
        # Alignment: 4, 4, 8 (offset 8), 8 (offset 16). SI.
        
        # Simplemente imprimimos lo que procesamos antes (snapshot)
        # Ojo: si el renderer escribe mientras leemos, podríamos ver tearing.
        # Para debug, asumimos que pausamos o capturamos un frame.
        
        print(f"Leyendo frame actual (Seq: {curr_seq})...")
        
        # Leer y Guardar
        output_dir = "debug_output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for channel, size in sizes.items():
            if size > 0:
                offset = offsets[channel]
                mm.seek(offset)
                buffer = mm.read(size)
                
                try:
                    array = np.frombuffer(buffer, dtype=np.float32)
                    
                    print(f"[{channel.upper()}] Min: {array.min():.4f}, Max: {array.max():.4f}, Mean: {array.mean():.4f}")
                    
                    # Guardar imagen (lógica anterior...)
                    if len(array) == width * height:
                        if channel == 'depth':
                             # Normalización simple para visualización
                             if array.max() > 0:
                                arr_norm = array / array.max()
                             else:
                                arr_norm = array
                        else:
                            arr_norm = np.clip(array, 0, 1)
                            
                        img_data = (arr_norm * 255).astype(np.uint8).reshape((height, width))
                        img = Image.fromarray(img_data, mode='L')
                        img.save(f"{output_dir}/preview_{channel}.png")

                    elif len(array) == width * height * 3:
                         arr_reshaped = array.reshape((height, width, 3))
                         # Asumir 0..1
                         arr_u8 = (np.clip(arr_reshaped, 0, 1) * 255).astype(np.uint8)
                         img = Image.fromarray(arr_u8, mode='RGB')
                         img.save(f"{output_dir}/preview_{channel}.png")
                         
                except Exception as e:
                    print(f"   Error: {e}")
            else:
                pass # vacio

        print("Captura finalizada. Ejecuta de nuevo para otra captura.")
        mm.close()
        break 


if __name__ == "__main__":
    read_mmf()
