# Resilience Brain: ML-Based Kubernetes Resilience Monitoring System

## Overview

Resilience Brain is an AIOps-inspired monitoring system designed for Kubernetes environments. The system collects real-time cluster metrics from Prometheus, applies Machine Learning-based anomaly detection using Isolation Forest, classifies failures, tracks recovery events, and provides remediation recommendations.

The goal of the project is to improve observability and resilience of Kubernetes workloads by automatically identifying abnormal cluster behavior and measuring recovery performance.

## Features

* Real-time Kubernetes cluster monitoring
* Prometheus integration using PromQL queries
* Isolation Forest-based anomaly detection
* Feature engineering using CPU-Memory interaction metrics
* Automated failure classification
* Recovery tracking and MTTR calculation
* Availability degradation measurement
* Suggested remediation actions for detected failures
* SQLite-based experiment and metrics storage

## Architecture

Kubernetes Cluster
→ Prometheus
→ Metrics Collector
→ Resilience Brain Engine
→ Isolation Forest Model
→ Failure Classification
→ Recovery Tracking
→ MTTR & Availability Analysis

## Technologies Used

* Python
* Kubernetes
* Prometheus
* PromQL
* Scikit-Learn
* Isolation Forest
* SQLite
* Pandas
* NumPy

## Project Structure

resilience-brain/

* resilience_brain.py
* prometheus_collector.py
* recovery_tracker.py
* db.py
* anomaly_detection_model.py
* requirements.txt

## Metrics Collected

* CPU Usage
* Memory Usage
* Ready Pods
* Total Pods
* Pod Restart Counts

## Failure Types Detected

* CPU Saturation
* Memory Collapse
* Crash Loop
* Resource Exhaustion
* Transient Anomalies

## Recovery Metrics

* Mean Time To Recovery (MTTR)
* Availability Drop
* Restart Increase

## Team

Developed as a collaborative B.Tech Major Project.

Team Members:

* Tanmay Sharma
* Ansh Verma

## Future Improvements

* Automatic Kubernetes YAML remediation
* Alerting integration
* Grafana dashboards
* Predictive failure forecasting
* Kubernetes Admission Controller integration
