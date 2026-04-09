# Kubernetes Deployment Guide: Smartphone Predictor

## Overview

This document provides a step-by-step guide to deploy and test the **Smartphone Predictor** application using Kubernetes (Minikube). It includes environment setup, deployment, monitoring, and stress testing.

---

## Phase A: Environment Setup

### 1. Start Minikube

```powershell
minikube start --driver=docker
```

### 2. Enable Metrics Server (Required for HPA)

```powershell
minikube addons enable metrics-server
```

### 3. Configure Docker to Use Minikube Engine

```powershell
& minikube -p minikube docker-env | Invoke-Expression
```

### 4. Build Docker Image

Ensure you are in the root directory containing your Dockerfile.

```powershell
docker build -t smartphone-predictor:latest .
```

---

## Phase B: Deployment

### 1. Apply Kubernetes Manifests

Make sure all YAML files (Deployment, Service, HPA) are inside the `k8s/` folder.

```powershell
kubectl apply -f k8s/
```

### 2. Expose Service via Minikube

```powershell
minikube service smartphone-service
```

* This opens a browser window.
* Copy the generated URL (e.g., `http://127.0.0.1:XXXXX`).

---

## Phase C: Verification

### 1. Check Pods

```powershell
kubectl get pods
```

Expected: Minimum 2 running pods.

### 2. Check HPA Status

```powershell
kubectl get hpa
```

* Initial state may show `<unknown>/50%`
* Wait 60–90 seconds for metrics collection.

---

## Phase D: Stress Testing

### PowerShell Stress Test Script

```powershell
# Update the URL from Minikube service output
$BASE_URL = "http://127.0.0.1:XXXXX"
$ENDPOINT = "$BASE_URL/ratings/predict"

$body = @{
    price = 7999
    brand_name = "motorola"
    has_5g = $false
    has_nfc = $false
    has_ir_blaster = $false
    num_cores = 8.0
    processor_speed = 1.8
    processor_brand = "tiger"
    ram_capacity = 4.0
    internal_memory = 64.0
    fast_charging = $null
    screen_size = 6.5
    resolution = "720x1600"
    refresh_rate = 90
    num_rear_cameras = 3
    num_front_cameras = 1
    primary_camera_rear = "48"
    primary_camera_front = 8.0
    fast_charging_available = 0
    extended_memory_available = 1
    extended_upto = 1024.0
} | ConvertTo-Json

Write-Host "Starting Stress Test on $ENDPOINT..." -ForegroundColor Yellow

while($true) {
    try {
        Invoke-RestMethod -Method Post -Uri $ENDPOINT -Body $body -ContentType "application/json"
        Write-Host "Prediction successful! Sending next..." -ForegroundColor Green
    }
    catch {
        Write-Host "Request failed. Pods might be restarting or overloaded." -ForegroundColor Red
    }
}
```

---

## Phase E: Monitoring Autoscaling

Run the following in a separate terminal:

```powershell
kubectl get hpa -w
```

### What to Observe:

* CPU utilization increases
* Number of replicas increases automatically
* Pods scale up under load and scale down when idle

---

## Troubleshooting

### 1. Metrics Not Showing

* Ensure metrics-server is enabled
* Wait at least 1–2 minutes

### 2. Pods Not Starting

```powershell
kubectl describe pod <pod-name>
kubectl logs <pod-name>
```

### 3. Service Not Accessible

* Ensure Minikube tunnel is running
* Re-run:

```powershell
minikube service smartphone-service
```

### 4. Image Not Found

* Ensure Docker environment is set correctly:

```powershell
& minikube docker-env
```

---

## Folder Structure

```
project-root/
│
├── k8s/
│   ├── deployment.yaml
│   ├── service.yaml
│   └── hpa.yaml
│
├── Dockerfile
├── app/
└── README.md
```

---

## Best Practices

* Use versioned image tags instead of `latest`
* Set proper resource requests and limits
* Use readiness and liveness probes
* Monitor logs and metrics regularly
* Keep manifests modular and reusable

---

## Conclusion

This setup simulates a production-like Kubernetes environment locally using Minikube. It allows you to test deployment, scaling, and performance of ML-powered API efficiently.

---

## Future Improvements

* Integrate CI/CD pipeline (GitHub Actions, Jenkins)
* Deploy on cloud Kubernetes (EKS, GKE, AKS)
* Add monitoring tools (Prometheus, Grafana)
* Implement logging (ELK stack)

---

**Author:** Pramit De
**Date:** 2026-03-26