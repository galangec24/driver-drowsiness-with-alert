// PWA Installation Handler
class PWAInstaller {
  constructor() {
    this.deferredPrompt = null;
    this.installButton = document.getElementById('installButton');
    this.installContainer = document.getElementById('installContainer');
    
    this.init();
  }
  
  init() {
    // Check if PWA is already installed
    if (window.matchMedia('(display-mode: standalone)').matches) {
      console.log('Running as installed PWA');
      return;
    }
    
    // Listen for beforeinstallprompt event
    window.addEventListener('beforeinstallprompt', (e) => {
      console.log('beforeinstallprompt event fired');
      
      // Prevent Chrome 67 and earlier from automatically showing the prompt
      e.preventDefault();
      
      // Stash the event so it can be triggered later
      this.deferredPrompt = e;
      
      // Show install button
      this.showInstallButton();
    });
    
    // Handle install button click
    if (this.installButton) {
      this.installButton.addEventListener('click', () => {
        this.installApp();
      });
    }
    
    // Track app installation
    window.addEventListener('appinstalled', (evt) => {
      console.log('App was successfully installed!');
      this.hideInstallButton();
      
      // Send analytics if you have them
      if (typeof gtag !== 'undefined') {
        gtag('event', 'app_installed');
      }
    });
    
    // Check if app is already installed
    if (navigator.standalone || window.matchMedia('(display-mode: standalone)').matches) {
      console.log('App is already installed');
      this.hideInstallButton();
    }
  }
  
  showInstallButton() {
    if (this.installContainer) {
      this.installContainer.style.display = 'block';
      
      // Add animation
      this.installContainer.classList.add('pwa-show');
    }
  }
  
  hideInstallButton() {
    if (this.installContainer) {
      this.installContainer.style.display = 'none';
    }
  }
  
  async installApp() {
    if (!this.deferredPrompt) {
      alert('Installation not available on this device/browser');
      return;
    }
    
    // Show the install prompt
    this.deferredPrompt.prompt();
    
    // Wait for the user to respond to the prompt
    const { outcome } = await this.deferredPrompt.userChoice;
    
    console.log(`User response to install prompt: ${outcome}`);
    
    // Clear the saved prompt since it can only be used once
    this.deferredPrompt = null;
    
    // Hide the install button
    this.hideInstallButton();
  }
  
  // Check if PWA is installable
  isInstallable() {
    return this.deferredPrompt !== null;
  }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  window.pwaInstaller = new PWAInstaller();
});

// Service Worker Registration
if ('serviceWorker' in navigator) {
  window.addEventListener('load', function() {
    navigator.serviceWorker.register('/service-worker.js')
      .then(registration => {
        console.log('ServiceWorker registration successful with scope: ', registration.scope);
        
        // Check for updates
        registration.addEventListener('updatefound', () => {
          const newWorker = registration.installing;
          console.log('ServiceWorker update found!');
          
          newWorker.addEventListener('statechange', () => {
            if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
              // New content is available, show update notification
              console.log('New content is available; please refresh.');
              showUpdateNotification();
            }
          });
        });
      })
      .catch(err => {
        console.log('ServiceWorker registration failed: ', err);
      });
  });
}

function showUpdateNotification() {
  // You can implement a refresh notification here
  if (confirm('New version available! Refresh to update?')) {
    window.location.reload();
  }
}