@echo off
echo Setting up Firebase deployment...

REM Create directories
mkdir firebase-deploy 2>nul
mkdir firebase-deploy\static 2>nul

REM Copy files
echo Copying static files...
copy frontend\*.png firebase-deploy\static\ >nul
copy frontend\styles.css firebase-deploy\static\ >nul
copy frontend\manifest.json firebase-deploy\ >nul
copy frontend\service-worker.js firebase-deploy\static\ >nul
copy frontend\device_redirect.js firebase-deploy\static\ >nul

echo Creating Firebase configuration files...

REM Create firebase.json
echo {
echo   "hosting": {
echo     "public": "firebase-deploy",
echo     "ignore": [
echo       "firebase-debug.log",
echo       "firestore-debug.log",
echo       "**/node_modules/**",
echo       "**/backend/**",
echo       "**/frontend/**",
echo       "**/.*"
echo     ],
echo     "rewrites": [
echo       {
echo         "source": "/",
echo         "destination": "/index.html"
echo       },
echo       {
echo         "source": "/manifest.json",
echo         "destination": "/manifest.json"
echo       },
echo       {
echo         "source": "/service-worker.js",
echo         "destination": "/static/service-worker.js"
echo       },
echo       {
echo         "source": "/icon-*",
echo         "destination": "/static/icon-192.png"
echo       },
echo       {
echo         "source": "**",
echo         "destination": "/index.html"
echo       }
echo     ]
echo   }
echo } > firebase.json

REM Create .firebaserc
echo {
echo   "projects": {
echo     "default": "guardian-drive"
echo   }
echo } > .firebaserc

echo.
echo Firebase setup complete!
echo.
echo Next steps:
echo 1. Run: firebase login
echo 2. Run: firebase init hosting
echo 3. Run: firebase deploy --only hosting
echo.
pause