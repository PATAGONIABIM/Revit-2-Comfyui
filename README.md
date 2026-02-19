# WabiSabi Revit Bridge - Revit 2026

**A High-Performance Real-Time Bridge to ComfyUI**

WabiSabi Revit Bridge is an advanced plugin for Autodesk Revit 2026 designed to act as a high-speed data bridge to ComfyUI. By extracting geometric information, materials, and metadata directly from an active 3D view, it enables real-time AI visualization workflows with minimal latency.

## 🚀 Key Features

*   **Real-Time Link**: Targets 60 FPS synchronization for camera movements.
*   **Smart Geometry Cache**: Implements a "Geometry Cache Manager" that extracts static geometry only once (or on change), dramatically reducing overhead for camera updates.
*   **GPU Acceleration**: Offloads heavy rendering tasks (Deep Maps, Normal Maps, Edge Detection) to an external GPU-based renderer.
*   **Zero-Latency Communication**: Uses Shared Memory (Memory Mapped Files) and Named Pipes for instant data transfer between Revit and the renderer.
*   **Three Operation Modes**:
    *   **Manual**: On-demand precise updates.
    *   **Automatic (Optimized)**: Smart change detection with configurable throttling.
    *   **Batch**: Sequential processing of multiple views.

## 🏗️ Architecture

The system operates on a **Producer/Consumer** model to bypass Revit's API performance limitations:

1.  **The Producer (Revit Plugin)**:
    *   Captures camera data (Position, Rotation, FOV) continuously.
    *   Extracts scene geometry only when necessary (Smart Cache).
    *   Writes data to a high-speed Shared Memory block.
2.  **The Consumer (External Renderer)**:
    *   A standalone GPU application (C++/CUDA/Unity/Unreal).
    *   Reads the Shared Memory stream.
    *   Renders technical passes (Depth, Normals, Segmentation) instantly using the cached geometry and live camera data.
3.  **The Destination (ComfyUI Node)**:
    *   Receives the rendered frames.
    *   Feeds them into ControlNet or other AI pipelines for image generation.

## 📦 Components

### 1. Revit Plugin (Producer)
The core add-in installed in Revit 2026. It handles the UI, manages the extraction logic, and orchestrates the caching system. It listens for model changes to invalidate the cache only when needed.

### 2. External Renderer (Consumer)
A lightweight, high-performance rendering engine. It visualizes the data sent by the plugin. By decoupled rendering from the Revit process, we achieve frame rates impossible within the Revit API context.

### 3. ComfyUI Node
A custom node for ComfyUI that acts as the receiver. It integrates the real-time stream into your node graph, allowing for "Live Interactive Rendering" where your Revit view drives the AI generation.

---

# WabiSabi Revit Bridge - Revit 2026 (Español)

**Un Puente de Alto Rendimiento en Tiempo Real hacia ComfyUI**

WabiSabi Revit Bridge es un plugin avanzado para Autodesk Revit 2026 diseñado para actuar como un puente de datos de alta velocidad hacia ComfyUI. Al extraer información geométrica, materiales y metadatos directamente de una vista 3D activa, permite flujos de trabajo de visualización con IA en tiempo real con una latencia mínima.

## 🚀 Características Principales

*   **Enlace en Tiempo Real**: Objetivo de 60 FPS para la sincronización de movimientos de cámara.
*   **Caché de Geometría Inteligente**: Implementa un "Gestor de Caché de Geometría" que extrae la geometría estática solo una vez (o cuando hay cambios), reduciendo drásticamente la carga en las actualizaciones de cámara.
*   **Aceleración por GPU**: Delega las tareas pesadas de renderizado (Mapas de Profundidad, Normales, Detección de Bordes) a un renderizador externo basado en GPU.
*   **Comunicación de Latencia Cero**: Utiliza Memoria Compartida (Memory Mapped Files) y Tuberías con Nombre (Named Pipes) para la transferencia instantánea de datos entre Revit y el renderizador.
*   **Tres Modos de Operación**:
    *   **Manual**: Actualizaciones precisas bajo demanda.
    *   **Automático (Optimizado)**: Detección inteligente de cambios con limitación (throttling) configurable.
    *   **Batch**: Procesamiento secuencial de múltiples vistas.

## 🏗️ Arquitectura

El sistema opera bajo un modelo **Productor/Consumidor** para evitar las limitaciones de rendimiento de la API de Revit:

1.  **El Productor (Plugin de Revit)**:
    *   Captura datos de la cámara (Posición, Rotación, FOV) continuamente.
    *   Extrae la geometría de la escena solo cuando es necesario (Caché Inteligente).
    *   Escribe los datos en un bloque de Memoria Compartida de alta velocidad.
2.  **El Consumidor (Renderizador Externo)**:
    *   Una aplicación GPU independiente (C++/CUDA/Unity/Unreal).
    *   Lee el flujo de Memoria Compartida.
    *   Renderiza pases técnicos (Profundidad, Normales, Segmentación) instantáneamente usando la geometría en caché y los datos de cámara en vivo.
3.  **El Destino (Nodo de ComfyUI)**:
    *   Recibe los fotogramas renderizados.
    *   Los alimenta a ControlNet u otros flujos de trabajo de IA para la generación de imágenes.

## 📦 Componentes

### 1. Plugin de Revit (Productor)
El complemento principal instalado en Revit 2026. Maneja la interfaz de usuario, gestiona la lógica de extracción y orquesta el sistema de caché. Escucha los cambios del modelo para invalidar el caché solo cuando es estrictamente necesario.

### 2. Renderizador Externo (Consumidor)
Un motor de renderizado ligero y de alto rendimiento. Visualiza los datos enviados por el plugin. Al desacoplar el renderizado del proceso de Revit, logramos tasas de cuadros por segundo imposibles dentro del contexto de la API de Revit.

### 3. Nodo de ComfyUI
Un nodo personalizado para ComfyUI que actúa como el receptor. Integra el flujo en tiempo real en tu gráfico de nodos, permitiendo un "Renderizado Interactivo en Vivo" donde tu vista de Revit impulsa la generación de IA.
