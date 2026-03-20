🚗 Driver Drowsiness Detection System














A cloud-based driver safety system that detects drowsiness using computer vision and delivers real-time alerts, SMS notifications, and live location tracking to guardians.

🧠 Overview

This system integrates:

👁️ Computer Vision (EAR-based detection)

☁️ Cloud-native architecture

⚡ Real-time WebSocket communication

🔐 Multi-provider authentication

📍 Live location + 3D distance monitoring

📩 SMS alert notifications

Designed for real-time monitoring, scalability, and rapid emergency response.

🏗️ System Architecture

The system follows a modular, distributed architecture:

Frontend (PWA) – Firebase Hosting

Backend API – Flask on Render

Database – PostgreSQL (Neon)

AI Service – Dockerized microservice

Image Storage – Cloudinary

Authentication – Supabase

🔄 Architecture Flow
Diagram
flowchart TD
A[User Login] --> B[Auth System]
B --> C[Session Created]
C --> D[Driver Monitoring Starts]
D --> E[AI Detection Service]
E -->|Not Drowsy| D
E -->|Drowsy Detected| F[Send Alert]
F --> G[Backend API]
G --> H[Database]
G --> I[WebSocket Alert]
G --> J[SMS Notification]
I --> K[Guardian Dashboard]
J --> K
K --> L[Live Map + Distance View]

🔐 Authentication System

Multi-provider authentication:

✅ Facebook Login

✅ Email + Password + OTP

🔄 (Planned) Microsoft Login

🔒 Security

OTP verification

JWT session management

bcrypt password hashing

Persistent sessions (auto-login)

📡 Core Components
📱 Frontend (PWA)

Vanilla JS + Bootstrap 5

Installable mobile experience

WebSocket-based real-time updates

Features:

Driver registration (camera capture)

Live monitoring dashboard

Alert visualization

Location tracking interface

⚙️ Backend API

Flask REST API (Render)

Handles:

Authentication & sessions

Driver & guardian management

Alert processing

WebSocket communication

SMS notification triggers

🧠 AI Detection Service

Dockerized Python microservice

Pipeline:

Fetch drivers

Retrieve images from Cloudinary

Detect facial landmarks (dlib)

Compute EAR

Detect drowsiness

Send alert

Detection Logic:

EAR Threshold: 0.25

Frames: 20 consecutive

☁️ Image Storage

Cloudinary (tagged by driver_id)

CDN-optimized delivery

No backend storage overhead

🗺️ Live Monitoring & 3D Mapping

Integrated with Google Maps Platform

Real-time driver location tracking

3D map visualization

Distance calculation between:

Driver ↔ Guardian

Driver ↔ Alert location

👉 Enables spatial awareness and faster response decisions

📩 SMS Alert System

Sends instant alerts to guardians

Works even without internet access on client side

Acts as fail-safe notification channel

Alert Includes:

Driver status

Timestamp

Location info

🔄 Data Flow

Guardian logs in

Registers driver + uploads images

Images stored in Cloudinary

AI service processes images

Drowsiness detected

Backend:

Stores alert

Sends WebSocket update

Sends SMS notification

Dashboard updates with:

Alert status

Live map

Distance tracking

🚀 Features
👤 Guardian

OTP-based registration

Facebook login

Real-time monitoring dashboard

WebSocket alerts

SMS notifications

Location & distance tracking

Activity logs

🚗 Driver

Face registration (multi-angle)

AI drowsiness detection

Automatic alert triggering

🛠 Admin

System monitoring

User & driver management

Alert analytics

📱 PWA

Installable

Offline support

Responsive UI

Real-time updates

🧠 Key Design Strengths

Microservice Architecture

Real-Time Communication (WebSocket)

Cloud Scalability

Fail-safe Alerts (SMS + Web)

Spatial Intelligence (3D Mapping + Distance)

🔮 Future Enhancements

Microsoft OAuth integration

Face recognition login

Mobile app version

Edge AI (on-device detection)

Predictive fatigue analytics

📌 Summary

A production-ready driver safety system combining:

AI-based drowsiness detection

Cloud-native infrastructure

Real-time + SMS alerting

Advanced location intelligence

Built for scalability, reliability, and real-world deployment.