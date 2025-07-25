WabiSabi Bridge - Módulo de Revit 2026

El objetivo de este proyecto es desarrollar un plugin de alto rendimiento para Autodesk Revit que actúe como un puente de datos en tiempo real hacia ComfyUI. El plugin extraerá información geométrica, de materiales y metadatos adicionales directamente de una vista 3D activa, procesándolos de manera eficiente y transmitiéndolos a través de un sistema de comunicación optimizado.

Esta versión mejorada implementará técnicas avanzadas de renderizado como submuestreo adaptativo, paralelización de procesos, y un sistema de caché inteligente para minimizar el impacto en el rendimiento de Revit. La comunicación se realizará mediante named pipes o memoria compartida para eliminar la latencia de las operaciones de disco, manteniendo la opción de exportación a archivos para compatibilidad.

El plugin operará en tres modos: **Modo Manual** para actualizaciones precisas bajo demanda, **Modo Automático Optimizado** que detectará cambios significativos con throttling configurable y exportara toda la interacción de la cámara con el modelo, ya se estatica, paseos, orbits, paneos, etc., y un nuevo **Modo Batch** para procesar múltiples vistas secuencialmente.

estamos en el paso real_time_link y puedes encontrar información de el en la carpeta docs\real_time_link.txt

archivo con información del sistema:
en la carpeta \docs\
revit_wabisabi_bridge.txt
Optimization Strategy.txt
idea_cache.txt
plan_caché.txt
real_time_link.txt