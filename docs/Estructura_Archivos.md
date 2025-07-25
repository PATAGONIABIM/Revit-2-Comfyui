Estructura recomendada para futuras expansiones:
WabiSabiBridge/
├── 📁 src/
│   ├── 📁 Core/                         # Lógica principal
│   │   ├── 📄 WabiSabiBridge.cs
│   │   └── 📄 Configuration.cs
│   ├── 📁 Extractors/                   # Extractores de datos
│   │   ├── 📄 HiddenLineExtractor.cs
│   │   ├── 📄 DepthExtractor.cs
│   │   └── 📄 SegmentationExtractor.cs
│   ├── 📁 Communication/                # Canales de comunicación
│   │   ├── 📄 FileChannel.cs
│   │   └── 📄 NamedPipeChannel.cs
│   └── 📁 UI/                          # Interfaz de usuario
│       └── 📄 MainWindow.cs
├── 📁 tests/                           # Pruebas unitarias
├── 📁 samples/                         # Ejemplos de uso
└── 📁 tools/                          # Herramientas adicionales