// frontend/service-worker.js
const CACHE_NAME = 'driver-drowsiness-pwa-v5';
const API_CACHE_NAME = 'api-cache-v1';
const OFFLINE_URL = '/offline.html';

// Core assets that MUST exist (only these)
const CORE_ASSETS = [
  '/',
  '/login.html',
  '/manifest.json',
  '/icon-192.png'
];

// Optional assets that might exist
const OPTIONAL_ASSETS = [
  '/index.html',
  '/guardian-dashboard.html',
  '/guardian-register.html',
  '/register-driver.html',
  '/icon-72.png',
  '/icon-144.png',
  '/icon-512.png',
  '/styles.css'
];

// External resources
const EXTERNAL_RESOURCES = [
  'https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css',
  'https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js',
  'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css'
];

// ==================== INSTALL ====================
self.addEventListener('install', event => {
  console.log('🚀 Service Worker installing...');
  
  // Skip waiting immediately - important for updates
  event.waitUntil(self.skipWaiting());
  
  // Cache core assets
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => {
      console.log('📦 Caching core assets...');
      
      // Cache only core assets that definitely exist
      return Promise.allSettled(
        CORE_ASSETS.map(asset => 
          fetch(asset, { cache: 'reload' })
            .then(response => {
              if (response.ok) {
                console.log(`✅ Cached: ${asset}`);
                return cache.put(asset, response);
              }
              console.warn(`⚠️ Skipping: ${asset} (status: ${response.status})`);
              return Promise.resolve();
            })
            .catch(err => {
              console.warn(`⚠️ Failed to cache ${asset}:`, err.message);
              return Promise.resolve();
            })
        )
      ).then(() => {
        console.log('✅ Core caching complete');
      });
    })
  );
});

// ==================== ACTIVATE ====================
self.addEventListener('activate', event => {
  console.log('🔄 Service Worker activating...');
  
  event.waitUntil(
    Promise.all([
      // Clean up old caches
      caches.keys().then(cacheNames => {
        return Promise.all(
          cacheNames.map(cacheName => {
            if (cacheName !== CACHE_NAME && cacheName !== API_CACHE_NAME) {
              console.log(`🗑️ Deleting old cache: ${cacheName}`);
              return caches.delete(cacheName);
            }
          })
        );
      }),
      
      // Claim clients immediately
      self.clients.claim(),
      
      // Cache optional assets in background
      cacheOptionalAssets(),
      
      // Cache external resources in background
      cacheExternalResources()
    ]).then(() => {
      console.log('✅ Service Worker ready!');
      
      // Notify clients
      self.clients.matchAll().then(clients => {
        clients.forEach(client => {
          client.postMessage({
            type: 'SW_READY',
            version: 'v5'
          });
        });
      });
    }).catch(err => {
      console.error('❌ Activation error:', err);
    })
  );
});

// ==================== FETCH STRATEGIES ====================
const fetchStrategies = {
  // Static assets: Cache First
  static: request => 
    caches.match(request)
      .then(cached => {
        if (cached) {
          // Return cached version immediately
          // Also fetch fresh version in background
          fetch(request).then(response => {
            if (response.ok) {
              caches.open(CACHE_NAME).then(cache => cache.put(request, response));
            }
          }).catch(() => {}); // Ignore fetch errors
          return cached;
        }
        
        // Not in cache, fetch from network
        return fetch(request)
          .then(response => {
            // Cache successful responses
            if (response.ok) {
              const responseClone = response.clone();
              caches.open(CACHE_NAME)
                .then(cache => cache.put(request, responseClone))
                .catch(err => console.warn('Cache put error:', err));
            }
            return response;
          })
          .catch(() => serveOffline(request));
      }),

  // API calls: Network First
  api: request =>
    fetch(request)
      .then(response => {
        // Cache successful GET API responses
        if (request.method === 'GET' && response.ok) {
          const responseClone = response.clone();
          caches.open(API_CACHE_NAME)
            .then(cache => cache.put(request, responseClone))
            .catch(err => console.warn('API cache error:', err));
        }
        return response;
      })
      .catch(() => 
        caches.match(request).then(cached => {
          if (cached) {
            return cached;
          }
          // Return offline response for API
          return new Response(JSON.stringify({ 
            success: false, 
            error: 'You are offline. Please check your connection.',
            offline: true 
          }), {
            status: 503,
            headers: { 
              'Content-Type': 'application/json',
              'Cache-Control': 'no-cache'
            }
          });
        })
      ),

  // Images: Cache First with background refresh
  image: request =>
    caches.match(request)
      .then(cached => {
        const fetchPromise = fetch(request)
          .then(response => {
            if (response.ok) {
              const responseClone = response.clone();
              caches.open(CACHE_NAME)
                .then(cache => cache.put(request, responseClone))
                .catch(err => console.warn('Image cache error:', err));
            }
            return response;
          })
          .catch(() => {});
        
        // Return cached if available, otherwise wait for network
        return cached || fetchPromise;
      }),

  // WebSocket: Direct fetch only
  websocket: request => fetch(request)
};

// ==================== MAIN FETCH HANDLER ====================
self.addEventListener('fetch', event => {
  const { request } = event;
  const url = new URL(request.url);
  
  // Skip non-GET requests
  if (request.method !== 'GET') {
    event.respondWith(fetch(request));
    return;
  }
  
  // Determine strategy
  let strategy;
  
  if (url.pathname.startsWith('/socket.io/')) {
    strategy = fetchStrategies.websocket;
  } else if (url.pathname.startsWith('/api/')) {
    strategy = fetchStrategies.api;
  } else if (request.headers.get('accept')?.includes('image/')) {
    strategy = fetchStrategies.image;
  } else if (url.origin === location.origin) {
    strategy = fetchStrategies.static;
  } else {
    strategy = fetchStrategies.static;
  }
  
  event.respondWith(strategy(request));
});

// ==================== HELPER FUNCTIONS ====================
function cacheOptionalAssets() {
  return caches.open(CACHE_NAME).then(cache => {
    console.log('📦 Caching optional assets in background...');
    
    return Promise.allSettled(
      OPTIONAL_ASSETS.map(asset => 
        fetch(asset)
          .then(response => {
            if (response.ok) {
              console.log(`✅ Cached optional: ${asset}`);
              return cache.put(asset, response);
            }
            return Promise.resolve();
          })
          .catch(() => Promise.resolve())
      )
    );
  });
}

function cacheExternalResources() {
  return caches.open(CACHE_NAME).then(cache => {
    console.log('🌐 Caching external resources in background...');
    
    return Promise.allSettled(
      EXTERNAL_RESOURCES.map(url => 
        fetch(url)
          .then(response => {
            if (response.ok) {
              console.log(`✅ Cached external: ${url}`);
              return cache.put(url, response);
            }
            return Promise.resolve();
          })
          .catch(() => Promise.resolve())
      )
    );
  });
}

function serveOffline(request) {
  if (request.headers.get('accept')?.includes('text/html')) {
    return caches.match('/offline.html')
      .then(response => response || new Response(
        `
        <!DOCTYPE html>
        <html>
        <head>
          <meta charset="UTF-8">
          <title>Offline - Driver Alert System</title>
          <style>
            body { font-family: Arial, sans-serif; padding: 20px; text-align: center; }
            .offline-icon { font-size: 48px; color: #666; margin: 20px 0; }
          </style>
        </head>
        <body>
          <div class="offline-icon">📶</div>
          <h2>You're Offline</h2>
          <p>Please check your internet connection.</p>
          <button onclick="location.reload()">Retry</button>
        </body>
        </html>
        `,
        {
          status: 503,
          headers: { 'Content-Type': 'text/html' }
        }
      ));
  }
  
  return new Response(
    'You are offline. Please check your internet connection.',
    {
      status: 503,
      headers: { 'Content-Type': 'text/plain' }
    }
  );
}

// ==================== PUSH NOTIFICATIONS ====================
self.addEventListener('push', event => {
  console.log('📢 Push notification received');
  
  let data = {};
  try {
    data = event.data ? event.data.json() : {};
  } catch (e) {
    data = {
      title: 'Driver Alert',
      body: 'New drowsiness alert detected!'
    };
  }
  
  const options = {
    body: data.body || 'Driver drowsiness detected!',
    icon: '/icon-192.png',
    badge: '/icon-72.png',
    vibrate: [200, 100, 200, 100, 200],
    data: {
      url: data.url || '/',
      driverId: data.driverId,
      alertId: data.alertId
    },
    actions: [
      {
        action: 'view-details',
        title: 'View Details',
        icon: '/icon-72.png'
      }
    ]
  };
  
  event.waitUntil(
    self.registration.showNotification(
      data.title || '🚨 Driver Alert',
      options
    )
  );
});

// ==================== NOTIFICATION CLICK HANDLER ====================
self.addEventListener('notificationclick', event => {
  console.log('🖱️ Notification clicked:', event.action);
  
  event.notification.close();
  
  const notificationData = event.notification.data || {};
  
  if (event.action === 'view-details') {
    event.waitUntil(
      clients.matchAll({ type: 'window' }).then(windowClients => {
        for (const client of windowClients) {
          if (client.url.includes('/guardian-dashboard') && 'focus' in client) {
            return client.focus();
          }
        }
        return clients.openWindow(notificationData.url || '/');
      })
    );
  } else {
    // Default: open dashboard
    event.waitUntil(
      clients.openWindow('/guardian-dashboard')
    );
  }
});

// ==================== MESSAGE HANDLER ====================
self.addEventListener('message', event => {
  console.log('📨 Message from client:', event.data);
  
  switch (event.data.type) {
    case 'SKIP_WAITING':
      self.skipWaiting();
      break;
      
    case 'GET_CACHE_INFO':
      caches.keys().then(cacheNames => {
        event.ports[0].postMessage({
          caches: cacheNames,
          version: 'v5'
        });
      });
      break;
      
    case 'CLEAR_CACHE':
      caches.keys().then(cacheNames => {
        Promise.all(cacheNames.map(name => caches.delete(name)))
          .then(() => {
            event.ports[0].postMessage({ success: true });
          });
      });
      break;
  }
});

// ==================== PERIODIC SYNC ====================
self.addEventListener('periodicsync', event => {
  if (event.tag === 'check-updates') {
    console.log('⏰ Periodic update check');
    event.waitUntil(
      fetch('/manifest.json', { cache: 'no-store' })
        .then(response => response.ok ? response.json() : null)
        .catch(() => null)
    );
  }
});

console.log('🧠 Service Worker v5 loaded successfully!');