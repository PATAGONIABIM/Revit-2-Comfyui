// WabiSabiViewer.js
class WabiSabiViewer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.renderer = new THREE.WebGLRenderer();
        
        // Texturas para los 4 mapas
        this.depthTexture = null;
        this.normalTexture = null;
        this.linesTexture = null;
        this.segmentationTexture = null;
        
        // WebSocket para recibir actualizaciones
        this.ws = null;
        
        this.init();
    }
    
    init() {
        this.renderer.setSize(window.innerWidth, window.innerHeight);
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
        const image = new Image();
        image.src = 'data:image/png;base64,' + base64Data;
        
        image.onload = () => {
            const texture = new THREE.Texture(image);
            texture.needsUpdate = true;
            
            switch(mapType) {
                case 'depth':
                    this.depthTexture = texture;
                    this.quad.material.uniforms.depthMap.value = texture;
                    break;
                case 'normal':
                    this.normalTexture = texture;
                    this.quad.material.uniforms.normalMap.value = texture;
                    break;
                case 'lines':
                    this.linesTexture = texture;
                    this.quad.material.uniforms.linesMap.value = texture;
                    break;
                case 'segmentation':
                    this.segmentationTexture = texture;
                    this.quad.material.uniforms.segmentationMap.value = texture;
                    break;
            }
        };
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
    
    animate() {
        requestAnimationFrame(() => this.animate());
        this.renderer.render(this.scene, this.camera);
    }
}