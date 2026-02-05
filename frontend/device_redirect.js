// frontend/device-redirect.js
(function() {
    // Check if we should redirect based on device type
    function shouldRedirectToMobile() {
        // Check if we're already on login.html
        if (window.location.pathname.includes('login.html')) {
            return false;
        }
        
        // Check if we're on admin.html (explicit admin access)
        if (window.location.pathname.includes('admin.html') || 
            window.location.search.includes('force_admin=true')) {
            return false;
        }
        
        // Check localStorage override
        if (localStorage.getItem('prefer_desktop_view') === 'true') {
            return false;
        }
        
        // Check query parameter
        if (window.location.search.includes('force_mobile=true')) {
            return true;
        }
        
        // Mobile detection
        const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
        const isTablet = /iPad|Tablet|PlayBook|Silk/i.test(navigator.userAgent);
        
        // Check screen size
        const isSmallScreen = window.innerWidth < 768;
        
        // Touch support
        const hasTouch = 'ontouchstart' in window || navigator.maxTouchPoints > 0;
        
        // Return true if mobile/tablet
        return (isMobile || isTablet || (isSmallScreen && hasTouch));
    }
    
    // Function to redirect to appropriate page
    function redirectToAppropriatePage() {
        const shouldRedirect = shouldRedirectToMobile();
        
        if (shouldRedirect && !window.location.pathname.includes('login.html')) {
            // Save original URL for desktop users
            if (!localStorage.getItem('original_desktop_url')) {
                localStorage.setItem('original_desktop_url', window.location.href);
            }
            
            // Redirect to login page for mobile
            window.location.href = '/login.html';
        } else if (!shouldRedirect && window.location.pathname.includes('login.html') && 
                   !window.location.search.includes('force_mobile=true')) {
            // If desktop user on login page, redirect to index
            window.location.href = '/';
        }
    }
    
    // Add a button to switch views
    function addViewSwitchButton() {
        const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
        
        if (isMobile) {
            // Add mobile-to-desktop switch button
            const switchBtn = document.createElement('button');
            switchBtn.id = 'viewSwitchBtn';
            switchBtn.innerHTML = '<i class="fas fa-desktop"></i> Desktop View';
            switchBtn.style.cssText = `
                position: fixed;
                bottom: 20px;
                right: 20px;
                z-index: 9999;
                background: #4e73df;
                color: white;
                border: none;
                border-radius: 50px;
                padding: 10px 20px;
                font-size: 14px;
                cursor: pointer;
                box-shadow: 0 4px 12px rgba(0,0,0,0.2);
                display: flex;
                align-items: center;
                gap: 8px;
            `;
            
            switchBtn.onclick = function() {
                localStorage.setItem('prefer_desktop_view', 'true');
                window.location.href = '/?force_admin=true';
            };
            
            document.body.appendChild(switchBtn);
        } else {
            // Add desktop-to-mobile switch button
            const switchBtn = document.createElement('button');
            switchBtn.id = 'viewSwitchBtn';
            switchBtn.innerHTML = '<i class="fas fa-mobile-alt"></i> Mobile View';
            switchBtn.style.cssText = `
                position: fixed;
                bottom: 20px;
                right: 20px;
                z-index: 9999;
                background: #4e73df;
                color: white;
                border: none;
                border-radius: 50px;
                padding: 10px 20px;
                font-size: 14px;
                cursor: pointer;
                box-shadow: 0 4px 12px rgba(0,0,0,0.2);
                display: flex;
                align-items: center;
                gap: 8px;
            `;
            
            switchBtn.onclick = function() {
                window.location.href = '/login.html?force_mobile=true';
            };
            
            document.body.appendChild(switchBtn);
        }
    }
    
    // Run on page load
    document.addEventListener('DOMContentLoaded', function() {
        redirectToAppropriatePage();
        
        // Only add switch button on specific pages
        if (window.location.pathname === '/' || 
            window.location.pathname.includes('index.html') ||
            window.location.pathname.includes('login.html')) {
            addViewSwitchButton();
        }
    });
    
    // Also run on window resize
    window.addEventListener('resize', function() {
        // Only check on small delays to avoid too many redirects
        clearTimeout(window.resizeTimer);
        window.resizeTimer = setTimeout(redirectToAppropriatePage, 250);
    });
})();