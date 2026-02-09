// frontend/service-worker.js
const CACHE_NAME = 'driver-drowsiness-pwa-v6';
const API_CACHE_NAME = 'api-cache-v2';
const OFFLINE_CACHE_NAME = 'offline-cache-v1';

// ==================== CORE ASSETS ====================
// Essential files that must be available offline
const CORE_ASSETS = [
  '/',
  '/login.html',
  '/manifest.json',
  '/icon-192.png',
  '/icon-512.png',
  '/icon-144.png',
  '/icon-128.png',
   '/icon-96.png',
  '/icon-72.png',
  '/offline.html'
];

// ==================== INSTALL EVENT ====================
self.addEventListener('install', event => {
  console.log('🚀 [Service Worker] Installing v6...');
  
  // Skip waiting so the new service worker takes control immediately
  event.waitUntil(self.skipWaiting());
  
  // Cache core assets
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => {
        console.log('📦 [Service Worker] Caching core assets...');
        
        // Cache each core asset
        const cachePromises = CORE_ASSETS.map(asset => {
          return fetch(asset, { cache: 'reload', credentials: 'same-origin' })
            .then(response => {
              if (response.ok) {
                console.log(`✅ [Service Worker] Cached: ${asset}`);
                return cache.put(asset, response);
              }
              console.warn(`⚠️ [Service Worker] Failed to cache: ${asset} (${response.status})`);
              return Promise.resolve();
            })
            .catch(error => {
              console.warn(`⚠️ [Service Worker] Network error caching ${asset}:`, error.message);
              return Promise.resolve();
            });
        });
        
        return Promise.allSettled(cachePromises)
          .then(() => {
            console.log('✅ [Service Worker] Core assets cached');
          });
      })
      .catch(error => {
        console.error('❌ [Service Worker] Cache opening failed:', error);
      })
  );
});

// ==================== ACTIVATE EVENT ====================
self.addEventListener('activate', event => {
  console.log('🔄 [Service Worker] Activating v6...');
  
  event.waitUntil(
    Promise.all([
      // Clean up old caches
      caches.keys().then(cacheNames => {
        console.log('🗑️ [Service Worker] Checking old caches...');
        const cachesToDelete = cacheNames.filter(
          cacheName => cacheName !== CACHE_NAME && 
                      cacheName !== API_CACHE_NAME && 
                      cacheName !== OFFLINE_CACHE_NAME
        );
        
        const deletePromises = cachesToDelete.map(cacheName => {
          console.log(`🗑️ [Service Worker] Deleting old cache: ${cacheName}`);
          return caches.delete(cacheName);
        });
        
        return Promise.all(deletePromises);
      }),
      
      // Take control of all clients immediately
      self.clients.claim(),
      
      // Pre-cache common assets in background
      preCacheCommonAssets()
      
    ]).then(() => {
      console.log('✅ [Service Worker] Activation complete');
      
      // Notify all clients about the new service worker
      self.clients.matchAll().then(clients => {
        clients.forEach(client => {
          client.postMessage({
            type: 'SW_ACTIVATED',
            version: 'v6',
            timestamp: new Date().toISOString()
          });
        });
      });
    })
    .catch(error => {
      console.error('❌ [Service Worker] Activation failed:', error);
    })
  );
});

// ==================== FETCH EVENT ====================
self.addEventListener('fetch', event => {
  const { request } = event;
  const url = new URL(request.url);
  
  // Skip non-GET requests and browser extensions
  if (request.method !== 'GET') {
    return;
  }
  
  // Skip Chrome extensions
  if (url.protocol === 'chrome-extension:') {
    return;
  }
  
  // Handle different types of requests
  if (url.pathname.startsWith('/socket.io/') || 
      url.pathname.startsWith('/api/ws') ||
      url.href.includes('ws://') || 
      url.href.includes('wss://')) {
    // WebSocket/Socket.IO requests - pass through
    return;
  }
  
  // API requests - Network First with cache fallback
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(handleApiRequest(request));
    return;
  }
  
  // Static assets - Cache First with network fallback
  event.respondWith(handleStaticRequest(request));
});

// ==================== API REQUEST HANDLER ====================
function handleApiRequest(request) {
  const url = new URL(request.url);
  
  // Don't cache POST, PUT, DELETE requests
  if (request.method !== 'GET') {
    return fetch(request)
      .catch(() => {
        return new Response(
          JSON.stringify({ 
            success: false, 
            error: 'Network error. Please check your connection.',
            offline: true 
          }),
          {
            status: 503,
            headers: { 
              'Content-Type': 'application/json',
              'Cache-Control': 'no-cache'
            }
          }
        );
      });
  }
  
  // For GET API requests, try network first, then cache
  return fetch(request)
    .then(response => {
      // Cache successful responses
      if (response.ok) {
        const responseClone = response.clone();
        caches.open(API_CACHE_NAME)
          .then(cache => {
            cache.put(request, responseClone)
              .catch(err => console.warn('[Service Worker] API cache error:', err));
          })
          .catch(err => console.warn('[Service Worker] Cache open error:', err));
      }
      return response;
    })
    .catch(() => {
      // Network failed, try cache
      return caches.match(request)
        .then(cachedResponse => {
          if (cachedResponse) {
            console.log(`📦 [Service Worker] Serving API from cache: ${url.pathname}`);
            return cachedResponse;
          }
          
          // No cache available, return offline response
          return new Response(
            JSON.stringify({ 
              success: false, 
              error: 'You are offline. Please check your connection.',
              offline: true 
            }),
            {
              status: 503,
              headers: { 
                'Content-Type': 'application/json',
                'Cache-Control': 'no-cache'
              }
            }
          );
        });
    });
}

// ==================== STATIC REQUEST HANDLER ====================
function handleStaticRequest(request) {
  const url = new URL(request.url);
  
  // Try cache first
  return caches.match(request)
    .then(cachedResponse => {
      if (cachedResponse) {
        console.log(`📦 [Service Worker] Serving from cache: ${url.pathname}`);
        
        // Update cache in background for next time
        fetchAndCache(request);
        
        return cachedResponse;
      }
      
      // Not in cache, fetch from network
      return fetch(request)
        .then(networkResponse => {
          // Cache successful responses
          if (networkResponse.ok && 
              networkResponse.status !== 206 && // Don't cache partial content
              !networkResponse.headers.get('content-type')?.includes('text/event-stream')) {
            
            const responseClone = networkResponse.clone();
            caches.open(CACHE_NAME)
              .then(cache => {
                cache.put(request, responseClone)
                  .catch(err => console.warn('[Service Worker] Static cache error:', err));
              })
              .catch(err => console.warn('[Service Worker] Cache open error:', err));
          }
          return networkResponse;
        })
        .catch(() => {
          // Network failed, return offline page for HTML requests
          if (request.headers.get('accept')?.includes('text/html')) {
            return caches.match('/offline.html')
              .then(offlinePage => offlinePage || createOfflineResponse());
          }
          
          // For other file types, return error
          if (request.url.match(/\.(js|css|png|jpg|jpeg|gif|ico|svg)$/)) {
            console.warn(`[Service Worker] Failed to load static file: ${url.pathname}`);
          }
          
          return createOfflineResponse();
        });
    });
}

// ==================== BACKGROUND CACHING ====================
function fetchAndCache(request) {
  // Only fetch if it's not already being fetched
  if (self.fetching && self.fetching.has(request.url)) {
    return;
  }
  
  if (!self.fetching) {
    self.fetching = new Set();
  }
  
  self.fetching.add(request.url);
  
  fetch(request)
    .then(response => {
      if (response.ok) {
        const responseClone = response.clone();
        caches.open(CACHE_NAME)
          .then(cache => {
            cache.put(request, responseClone);
            console.log(`🔄 [Service Worker] Background updated: ${new URL(request.url).pathname}`);
          });
      }
    })
    .catch(() => { /* Ignore background fetch errors */ })
    .finally(() => {
      self.fetching.delete(request.url);
    });
}

// ==================== PRE-CACHE COMMON ASSETS ====================
function preCacheCommonAssets() {
  return caches.open(CACHE_NAME)
    .then(cache => {
      const commonAssets = [
        '/admin_login.html',
        '/guardian-dashboard.html',
        '/guardian-register.html',
        '/register-driver.html',
        '/styles.css',
        '/icon-72.png',
        '/icon-96.png',
        '/icon-128.png',
        '/icon-144.png',
        '/icon-152.png',
        '/icon-384.png',
        '/icon-512.png'
      ];
      
      console.log('📦 [Service Worker] Pre-caching common assets...');
      
      const cachePromises = commonAssets.map(asset => {
        return fetch(asset, { cache: 'reload', credentials: 'same-origin' })
          .then(response => {
            if (response.ok) {
              return cache.put(asset, response);
            }
            return Promise.resolve();
          })
          .catch(() => Promise.resolve());
      });
      
      return Promise.allSettled(cachePromises)
        .then(results => {
          const successful = results.filter(r => r.status === 'fulfilled').length;
          console.log(`✅ [Service Worker] Pre-cached ${successful}/${commonAssets.length} common assets`);
        });
    });
}

// ==================== CREATE OFFLINE RESPONSE ====================
function createOfflineResponse() {
  const offlineHtml = `
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Offline - Driver Alert System</title>
        <style>
            body {
                font-family: 'Poppins', sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                height: 100vh;
                margin: 0;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                padding: 20px;
                text-align: center;
            }
            .container {
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                padding: 40px;
                border-radius: 20px;
                max-width: 400px;
                width: 100%;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }
            .icon {
                font-size: 60px;
                margin-bottom: 20px;
                animation: pulse 2s infinite;
            }
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.7; }
            }
            h1 {
                margin-bottom: 15px;
                font-weight: 600;
                font-size: 28px;
            }
            p {
                margin-bottom: 25px;
                opacity: 0.9;
                line-height: 1.5;
            }
            button {
                background: white;
                color: #667eea;
                border: none;
                padding: 14px 35px;
                border-radius: 50px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }
            button:hover {
                transform: translateY(-3px);
                box-shadow: 0 8px 20px rgba(0,0,0,0.15);
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="icon">📶</div>
            <h1>You're Offline</h1>
            <p>Please check your internet connection and try again.</p>
            <button onclick="location.reload()">Retry Connection</button>
        </div>
    </body>
    </html>
  `;
  
  return new Response(offlineHtml, {
    status: 503,
    statusText: 'Service Unavailable',
    headers: {
      'Content-Type': 'text/html; charset=utf-8',
      'Cache-Control': 'no-cache'
    }
  });
}

// ==================== PUSH NOTIFICATIONS ====================
self.addEventListener('push', event => {
  console.log('📢 [Service Worker] Push notification received');
  
  let data = {};
  try {
    data = event.data ? event.data.json() : {};
  } catch (e) {
    data = {
      title: 'Driver Alert',
      body: 'New drowsiness alert detected!',
      icon: '/icon-192.png'
    };
  }
  
  const options = {
    body: data.body || 'Driver drowsiness detected!',
    icon: data.icon || '/icon-192.png',
    badge: '/icon-72.png',
    vibrate: [200, 100, 200, 100, 200],
    data: {
      url: data.url || '/',
      driverId: data.driverId,
      alertId: data.alertId,
      timestamp: new Date().toISOString()
    },
    tag: data.tag || 'driver-alert',
    renotify: true,
    requireInteraction: true,
    actions: [
      {
        action: 'view-alert',
        title: 'View Alert',
        icon: '/icon-72.png'
      },
      {
        action: 'dismiss',
        title: 'Dismiss',
        icon: '/icon-72.png'
      }
    ]
  };
  
  event.waitUntil(
    self.registration.showNotification(
      data.title || '🚨 Driver Alert',
      options
    )
    .catch(error => {
      console.error('❌ [Service Worker] Push notification error:', error);
    })
  );
});

// ==================== NOTIFICATION CLICK HANDLER ====================
self.addEventListener('notificationclick', event => {
  console.log('🖱️ [Service Worker] Notification clicked:', event.action);
  
  event.notification.close();
  
  const notificationData = event.notification.data || {};
  
  // Handle different actions
  switch (event.action) {
    case 'view-alert':
      event.waitUntil(
        handleViewAlert(notificationData)
      );
      break;
      
    case 'dismiss':
      // Just close the notification
      break;
      
    default:
      // Default: open dashboard
      event.waitUntil(
        clients.openWindow('/guardian-dashboard')
      );
      break;
  }
});

function handleViewAlert(notificationData) {
  return clients.matchAll({
    type: 'window',
    includeUncontrolled: true
  })
  .then(windowClients => {
    // Check if there's already a window open
    for (const client of windowClients) {
      if (client.url.includes('/guardian-dashboard') && 'focus' in client) {
        // Focus the existing window
        return client.focus().then(() => {
          // Send message to the client to show the alert
          client.postMessage({
            type: 'SHOW_ALERT',
            alertId: notificationData.alertId,
            driverId: notificationData.driverId
          });
        });
      }
    }
    
    // No window found, open a new one
    return clients.openWindow('/guardian-dashboard')
      .then(newClient => {
        if (newClient) {
          // Wait a bit for the page to load, then send message
          setTimeout(() => {
            newClient.postMessage({
              type: 'SHOW_ALERT',
              alertId: notificationData.alertId,
              driverId: notificationData.driverId
            });
          }, 1000);
        }
      });
  });
}

// ==================== MESSAGE HANDLER ====================
self.addEventListener('message', event => {
  console.log('📨 [Service Worker] Message from client:', event.data);
  
  if (!event.data || !event.data.type) return;
  
  switch (event.data.type) {
    case 'SKIP_WAITING':
      console.log('⏩ [Service Worker] Skipping waiting...');
      self.skipWaiting();
      break;
      
    case 'CLAIM_CLIENTS':
      console.log('👥 [Service Worker] Claiming clients...');
      self.clients.claim();
      break;
      
    case 'GET_CACHE_INFO':
      caches.keys().then(cacheNames => {
        const cacheInfo = cacheNames.map(name => ({
          name,
          size: 'unknown'
        }));
        
        event.ports[0].postMessage({
          type: 'CACHE_INFO',
          caches: cacheInfo,
          version: 'v6',
          timestamp: new Date().toISOString()
        });
      });
      break;
      
    case 'CLEAR_CACHE':
      caches.keys().then(cacheNames => {
        const deletePromises = cacheNames.map(name => caches.delete(name));
        Promise.all(deletePromises)
          .then(() => {
            console.log('🧹 [Service Worker] All caches cleared');
            if (event.ports[0]) {
              event.ports[0].postMessage({ 
                type: 'CACHE_CLEARED',
                success: true 
              });
            }
          })
          .catch(error => {
            console.error('❌ [Service Worker] Error clearing caches:', error);
            if (event.ports[0]) {
              event.ports[0].postMessage({ 
                type: 'CACHE_CLEARED',
                success: false,
                error: error.message 
              });
            }
          });
      });
      break;
      
    case 'UPDATE_CACHE':
      const asset = event.data.asset;
      if (asset) {
        fetch(asset, { cache: 'reload' })
          .then(response => {
            if (response.ok) {
              caches.open(CACHE_NAME)
                .then(cache => cache.put(asset, response))
                .then(() => {
                  console.log(`🔄 [Service Worker] Updated cache for: ${asset}`);
                  if (event.ports[0]) {
                    event.ports[0].postMessage({
                      type: 'CACHE_UPDATED',
                      asset,
                      success: true
                    });
                  }
                });
            }
          })
          .catch(error => {
            console.error(`❌ [Service Worker] Failed to update cache for ${asset}:`, error);
            if (event.ports[0]) {
              event.ports[0].postMessage({
                type: 'CACHE_UPDATED',
                asset,
                success: false,
                error: error.message
              });
            }
          });
      }
      break;
      
    case 'CHECK_UPDATE':
      // Check for updates by fetching the service worker file
      fetch('/service-worker.js', { cache: 'no-store' })
        .then(response => response.text())
        .then(text => {
          // Compare with current version (simplified check)
          const hasUpdate = !text.includes('v6');
          if (event.ports[0]) {
            event.ports[0].postMessage({
              type: 'UPDATE_CHECK',
              hasUpdate,
              currentVersion: 'v6'
            });
          }
        })
        .catch(() => {
          if (event.ports[0]) {
            event.ports[0].postMessage({
              type: 'UPDATE_CHECK',
              hasUpdate: false,
              error: 'Network error'
            });
          }
        });
      break;
  }
});

// ==================== PERIODIC SYNC ====================
self.addEventListener('periodicsync', event => {
  if (event.tag === 'update-check') {
    console.log('⏰ [Service Worker] Periodic update check');
    event.waitUntil(
      checkForUpdates()
    );
  }
});

function checkForUpdates() {
  return Promise.allSettled([
    fetch('/manifest.json', { cache: 'no-store' }).then(r => r.ok ? r.json() : null),
    fetch('/service-worker.js', { cache: 'no-store' }).then(r => r.ok ? r.text() : null)
  ])
  .then(results => {
    console.log('🔍 [Service Worker] Update check complete');
    return results;
  });
}

// ==================== BACKGROUND SYNC ====================
self.addEventListener('sync', event => {
  console.log('🔄 [Service Worker] Background sync:', event.tag);
  
  if (event.tag === 'sync-alerts') {
    event.waitUntil(syncPendingAlerts());
  }
});

function syncPendingAlerts() {
  // This would sync pending alerts when back online
  console.log('📡 [Service Worker] Syncing pending alerts...');
  return Promise.resolve();
}

// ==================== ERROR HANDLING ====================
self.addEventListener('error', event => {
  console.error('❌ [Service Worker] Error:', event.error);
});

self.addEventListener('unhandledrejection', event => {
  console.error('❌ [Service Worker] Unhandled rejection:', event.reason);
});

// ==================== INITIALIZATION ====================
console.log('🧠 [Service Worker] v6 loaded and ready!');
console.log('📅 Loaded at:', new Date().toISOString());
console.log('📍 Scope:', self.registration?.scope || 'unknown');

// Broadcast that service worker is ready
if (self.clients) {
  self.clients.matchAll().then(clients => {
    clients.forEach(client => {
      client.postMessage({
        type: 'SW_READY',
        version: 'v6',
        timestamp: new Date().toISOString()
      });
    });
  });
}