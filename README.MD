# Driver Drowsiness Detection System

A capstone project that detects driver drowsiness and sends alerts to registered guardians. 

Architecture:
The Driver Drowsiness Detection System employs a modular, cloud-native architecture composed of four main components: a Firebase-hosted PWA frontend, a Render-deployed Flask backend API, a PostgreSQL database, and a standalone Dockerized AI inference service, all integrated through Cloudinary for image storage and REST APIs for communication.

The frontend is a Progressive Web Application (PWA) hosted on Firebase Hosting at guardian-drive-app.web.app. Built with vanilla JavaScript, Bootstrap 5, and WebSocket clients, it provides a mobile-responsive interface for guardians to register drivers, capture face images via device camera, monitor real-time alerts, and manage their accounts. The PWA can be installed on mobile home screens and includes a service worker for offline asset caching. All frontend communication with the backend occurs through authenticated REST API calls and WebSocket connections for instant alert delivery.

The backend is a Flask-based REST API server deployed on Render.com that serves as the system's central orchestrator. It handles guardian authentication (both phone/password and Google OAuth), session management using JWT tokens, driver registration with Cloudinary image uploads, and alert management. The backend maintains persistent connections via Flask-SocketIO to push real-time drowsiness alerts to authenticated guardian dashboards. All critical data—guardians, drivers, alerts, face image references, and session tokens—is stored in a PostgreSQL database hosted on Neon.tech, with bcrypt password hashing and parameterized queries for security. The backend exposes REST endpoints for the frontend, admin dashboard, and AI service, all protected by session validation or admin authentication decorators.

Cloudinary serves as the dedicated image management platform. When a guardian registers a driver, the frontend captures three or more face images from different angles, sends them as base64 strings to the backend, which then uploads them to Cloudinary. Critically, each upload is tagged with the unique driver_id, enabling efficient retrieval later. Cloudinary returns secure URLs that are stored in the database, while the original images are served via Cloudinary's global CDN for fast access. This offloads all image storage and bandwidth from the main backend.

The AI Drowsiness Detection Service is a completely independent, self-contained Python application running in a Docker container with Python 3.10.11. It operates autonomously on a separate Linux server (or any Docker-capable machine) and communicates solely via HTTP with the main backend. The service runs an infinite loop that at regular intervals: (1) fetches the list of all active drivers from the backend's /api/guardian/active-drivers endpoint, (2) for each driver, retrieves their latest 5-10 face images from Cloudinary using the driver ID as a search tag, (3) downloads and decodes each image into OpenCV format, (4) applies dlib's pre-trained facial landmark detector (shape_predictor_68_face_landmarks.dat) to locate the eyes and compute the Eye Aspect Ratio (EAR), (5) if the EAR falls below 0.25 for 20 consecutive frames (simulated across the batch of images), classifies the driver as drowsy, and (6) sends an alert payload to the backend's /api/send-alert endpoint. The service implements per-driver cooldown logic to prevent alert spam and logs all activities to both console and persistent log files mounted via Docker volumes. It is configured with restart: unless-stopped policy, ensuring automatic recovery from crashes and system reboots.

Data flow begins when a guardian registers a driver through the PWA. The face images are uploaded to Cloudinary with the driver ID as a tag, and the driver record is created in PostgreSQL. The AI service, running independently, periodically queries the backend for all active drivers. For each driver, it fetches their tagged images from Cloudinary, performs inference, and upon drowsiness detection, sends an alert back to the backend. The backend stores this alert in the database and simultaneously pushes it via WebSocket to all authenticated guardian clients associated with that driver. The guardian dashboard instantly displays the alert with visual indicators and sound notifications.

This architecture provides complete separation of concerns: the main backend remains lightweight and responsive, focused on authentication, data management, and real-time communication; the AI service handles computationally intensive image processing in isolation and can be scaled horizontally by running multiple container instances; Cloudinary eliminates the need for local image storage and provides global CDN acceleration; and the PWA frontend delivers a native-app-like experience without requiring App Store deployment. The system is designed for high availability, with each component independently deployable and scalable, communicating through well-defined, versioned APIs.


Features:

👤 Guardian Features

✅ Account Registration - Register with phone number & password

✅ Google OAuth - One-click login with Google account

✅ Driver Management - Add, edit, delete drivers

✅ Real-time Dashboard - Monitor driver status and alerts

✅ Instant Notifications - Receive WebSocket alerts when drowsiness detected

✅ Activity Logs - View complete history of alerts

✅ Profile Management - Update personal information


🚗 Driver Features

✅ Face Registration - Capture 3+ face images from different angles

✅ Cloudinary Storage - Secure face image storage in cloud

✅ Drowsiness Detection - AI-powered EAR (Eye Aspect Ratio) analysis

✅ Alert Generation - Automatic alerts when drowsiness detected


🛠 Admin Features

✅ System Overview - View all guardians, drivers, and alerts

✅ Database Management - Monitor system health and statistics

✅ User Management - Admin dashboard for oversight


📱 PWA Features

✅ Installable - Works as a native mobile app

✅ Offline Support - Basic functionality without internet

✅ Push Notifications - Alert notifications on mobile

✅ Responsive Design - Works on all screen sizes
