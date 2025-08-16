// WabiSabiViewer.js
class WabiSabiViewer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.renderer = new THREE.WebGLRenderer({ alpha: false }); // alpha:false es más eficiente
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setClearColor(0x1a1a1a, 1); // Fondo gris oscuro para ver si el canvas aparece
        this.container.appendChild(this.renderer.domElement);
        
        // Texturas para los 4 mapas
        this.depthTexture = null;
        this.normalTexture = null;
        this.linesTexture = null;
        this.segmentationTexture = null;
        this.viewMode = 'single'; // 'single' o 'grid'
        this.gridQuads = {}; // Almacenará los 4 quads de la grilla
        this.autoUpdate = true;
        
        // WebSocket para recibir actualizaciones
        this.ws = null;
        
        this.init();
    }
    
    init() {
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setClearColor(0xff00ff);
        this.container.appendChild(this.renderer.domElement);
        
        // Crear quad para mostrar las texturas
        const geometry = new THREE.PlaneGeometry(2, 2);
        const material = new THREE.ShaderMaterial({
            uniforms: {
                depthMap: { value: null },
                normalMap: { value: null },
                linesMap: { value: null },
                segmentationMap: { value: null },
                displayMode: { value: 0 } // 0: depth, 1: normal, 2: lines, 3: segmentation
            },
            vertexShader: `
                varying vec2 vUv;
                void main() {
                    vUv = uv;
                    gl_Position = vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform sampler2D depthMap;
                uniform sampler2D normalMap;
                uniform sampler2D linesMap;
                uniform sampler2D segmentationMap;
                uniform int displayMode;
                varying vec2 vUv;
                
                void main() {
                    vec4 color;
                    if (displayMode == 0) {
                        float depth = texture2D(depthMap, vUv).r;
                        color = vec4(depth, depth, depth, 1.0);
                    } else if (displayMode == 1) {
                        color = texture2D(normalMap, vUv);
                    } else if (displayMode == 2) {
                        color = texture2D(linesMap, vUv);
                    } else {
                        color = texture2D(segmentationMap, vUv);
                    }
                    gl_FragColor = color;
                }
            `
        });
        
        this.quad = new THREE.Mesh(geometry, material);
        this.scene.add(this.quad);

        this.createGridView();
        
        // Conectar WebSocket
        this.connectWebSocket();
        
        // Controles de teclado
        this.setupControls();
        
        this.animate();
    }
    
    connectWebSocket() {
        this.ws = new WebSocket('ws://localhost:9001');
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            if (data.type === 'texture_update') {
                // Actualizar texturas con datos base64
                this.updateTexture(data.mapType, data.imageData);
            } else if (data.type === 'camera_update') {
                // Sincronizar cámara con Revit
                this.updateCamera(data.camera);
            }
        };
    }
    
    updateTexture(mapType, base64Data) {
        if (!this.autoUpdate) return;
        const image = new Image();
        
        image.onerror = (error) => {
            console.error(`[WabiSabiViewer] Error al cargar la imagen para el mapa: ${mapType}.`);
            console.error("Esto usualmente significa que los datos Base64 recibidos del servidor están corruptos o no son un PNG válido.");
            console.error("Datos Base64 recibidos (primeros 100 caracteres):", base64Data.substring(0, 100));
        };
        
        image.onload = () => {
            // Si ves este mensaje, la imagen se decodificó correctamente.
            console.log(`[WabiSabiViewer] Imagen para '${mapType}' cargada exitosamente.`);
            const texture = new THREE.Texture(image);
            texture.needsUpdate = true;
            
            // --- INICIO DE LA MODIFICACIÓN ---
            switch(mapType) {
                case 'depth':
                    this.depthTexture = texture;
                    this.quad.material.uniforms.depthMap.value = texture;
                    if (this.gridQuads.depth) this.gridQuads.depth.material.uniforms.map.value = texture;
                    break;
                case 'normal':
                    this.normalTexture = texture;
                    this.quad.material.uniforms.normalMap.value = texture;
                    if (this.gridQuads.normal) this.gridQuads.normal.material.uniforms.map.value = texture;
                    break;
                case 'lines':
                    this.linesTexture = texture;
                    this.quad.material.uniforms.linesMap.value = texture;
                    if (this.gridQuads.lines) this.gridQuads.lines.material.uniforms.map.value = texture;
                    break;
                case 'segmentation':
                    this.segmentationTexture = texture;
                    this.quad.material.uniforms.segmentationMap.value = texture;
                    if (this.gridQuads.segmentation) this.gridQuads.segmentation.material.uniforms.map.value = texture;
                    break;
            }
            
            // Actualizar panel de información para dar feedback
            document.getElementById('resolution').textContent = `${image.width}x${image.height}`;
            document.getElementById('lastUpdate').textContent = new Date().toLocaleTimeString();
            // --- FIN DE LA MODIFICACIÓN ---
        };
        image.src = 'data:image/png;base64,' + base64Data;
    }
    
    setupControls() {
        document.addEventListener('keydown', (event) => {
            switch(event.key) {
                case '1':
                    this.quad.material.uniforms.displayMode.value = 0; // Depth
                    break;
                case '2':
                    this.quad.material.uniforms.displayMode.value = 1; // Normal
                    break;
                case '3':
                    this.quad.material.uniforms.displayMode.value = 2; // Lines
                    break;
                case '4':
                    this.quad.material.uniforms.displayMode.value = 3; // Segmentation
                    break;
            }
        });
    }
    
    setDisplayMode(mode) {
        if (this.quad && this.quad.material.uniforms.displayMode) {
            this.quad.material.uniforms.displayMode.value = mode;
        }
    }
    
    animate() {
        requestAnimationFrame(() => this.animate());
        this.renderer.render(this.scene, this.camera);
    }
    createGridView() {
        const createQuad = (textureUniformName, texture) => {
            const material = new THREE.ShaderMaterial({
                uniforms: { map: { value: texture } },
                vertexShader: `
                    varying vec2 vUv;
                    void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }
                `,
                fragmentShader: `
                    uniform sampler2D map;
                    varying vec2 vUv;
                    void main() { gl_FragColor = texture2D(map, vUv); }
                `
            });
            const quad = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), material);
            quad.visible = false;
            this.scene.add(quad);
            return quad;
        };

        this.gridQuads.depth = createQuad('depthMap', this.depthTexture);
        this.gridQuads.normal = createQuad('normalMap', this.normalTexture);
        this.gridQuads.lines = createQuad('linesMap', this.linesTexture);
        this.gridQuads.segmentation = createQuad('segmentationMap', this.segmentationTexture);

        // Posicionar quads en una grilla 2x2
        this.gridQuads.depth.position.set(-0.5, 0.5, 0);
        this.gridQuads.normal.position.set(0.5, 0.5, 0);
        this.gridQuads.lines.position.set(-0.5, -0.5, 0);
        this.gridQuads.segmentation.position.set(0.5, -0.5, 0);
    }

    // IMPLEMENTACIÓN DE LAS FUNCIONES QUE FALTABAN
    setViewMode(mode) {
        this.viewMode = mode;
        if (mode === 'single') {
            this.quad.visible = true;
            Object.values(this.gridQuads).forEach(q => q.visible = false);
            // Restaurar cámara de perspectiva
            this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        } else { // grid
            this.quad.visible = false;
            Object.values(this.gridQuads).forEach(q => q.visible = true);
            // Usar cámara ortográfica para la grilla
            this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 1000);
            this.camera.position.z = 1;
        }
    }

    setDisplayMode(mode) {
        if (this.quad && this.quad.material.uniforms.displayMode) {
            this.quad.material.uniforms.displayMode.value = mode;
        }
    }
    
    setAutoUpdate(enabled) {
        this.autoUpdate = enabled;
        document.getElementById('autoUpdate').checked = enabled;
    }
    
    toggleViewMode() {
        const newMode = this.viewMode === 'single' ? 'grid' : 'single';
        this.setViewMode(newMode);
        document.querySelectorAll('[data-view]').forEach(b => b.classList.remove('active'));
        document.querySelector(`[data-view="${newMode}"]`).classList.add('active');
    }

    toggleAutoUpdate() {
        this.setAutoUpdate(!this.autoUpdate);
    }
    // --- FIN DE LA MODIFICACIÓN ---
}