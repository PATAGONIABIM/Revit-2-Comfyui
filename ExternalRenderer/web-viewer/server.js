// ExternalRenderer/web-viewer/server.js
const express = require('express');
const WebSocket = require('ws');
const cors = require('cors');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = process.env.PORT || 3000;
const WS_PORT = process.env.WS_PORT || 9001;

// Middleware
app.use(cors());
app.use(express.static(__dirname));
app.use(express.json());

// Ruta principal
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

// API para obtener información del renderer
app.get('/api/renderer/status', (req, res) => {
    res.json({
        status: 'running',
        maps: {
            depth: true,
            normal: true,
            lines: true,
            segmentation: true
        },
        performance: {
            fps: getCurrentFPS(),
            frameTime: getLastFrameTime()
        }
    });
});

// WebSocket Server
const wss = new WebSocket.Server({ port: WS_PORT });

const clients = new Set();
let rendererConnection = null;

wss.on('connection', (ws, req) => {
    console.log('Nueva conexión WebSocket desde:', req.socket.remoteAddress);
    
    // Identificar tipo de cliente
    ws.on('message', (message) => {
        try {
            const data = JSON.parse(message);
            
            if (data.type === 'identify') {
                if (data.client === 'renderer') {
                    rendererConnection = ws;
                    console.log('Renderer conectado');
                    
                    // Notificar a todos los viewers
                    broadcastToViewers({
                        type: 'renderer_status',
                        connected: true
                    });
                } else if (data.client === 'viewer') {
                    clients.add(ws);
                    console.log('Viewer conectado. Total viewers:', clients.size);
                    
                    // Enviar estado inicial
                    ws.send(JSON.stringify({
                        type: 'welcome',
                        rendererConnected: rendererConnection !== null
                    }));
                }
            } else if (data.type === 'texture_update' && ws === rendererConnection) {
                // Reenviar actualizaciones de textura a todos los viewers
                broadcastToViewers(data);
            }
        } catch (error) {
            console.error('Error procesando mensaje:', error);
        }
    });
    
    ws.on('close', () => {
        if (ws === rendererConnection) {
            rendererConnection = null;
            console.log('Renderer desconectado');
            broadcastToViewers({
                type: 'renderer_status',
                connected: false
            });
        } else {
            clients.delete(ws);
            console.log('Viewer desconectado. Total viewers:', clients.size);
        }
    });
    
    ws.on('error', (error) => {
        console.error('WebSocket error:', error);
    });
});

function broadcastToViewers(data) {
    const message = JSON.stringify(data);
    clients.forEach(client => {
        if (client.readyState === WebSocket.OPEN) {
            client.send(message);
        }
    });
}

// Variables para tracking de rendimiento
let frameCount = 0;
let lastFPSUpdate = Date.now();
let currentFPS = 0;
let lastFrameTime = 0;

function getCurrentFPS() {
    return currentFPS;
}

function getLastFrameTime() {
    return lastFrameTime;
}

// Actualizar FPS cada segundo
setInterval(() => {
    const now = Date.now();
    const delta = now - lastFPSUpdate;
    currentFPS = Math.round((frameCount * 1000) / delta);
    frameCount = 0;
    lastFPSUpdate = now;
}, 1000);

// Iniciar servidor HTTP
app.listen(PORT, () => {
    console.log(`Servidor HTTP corriendo en http://localhost:${PORT}`);
    console.log(`WebSocket Server corriendo en ws://localhost:${WS_PORT}`);
});

// Manejo de cierre graceful
process.on('SIGINT', () => {
    console.log('\nCerrando servidor...');
    
    // Cerrar conexiones WebSocket
    clients.forEach(client => client.close());
    if (rendererConnection) rendererConnection.close();
    wss.close();
    
    process.exit(0);
});