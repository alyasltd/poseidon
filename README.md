# 🌊 POSEIDON  
**POSe estimation with Explicit/Implicit Differentiable OptimizatioN**
**POSe Estimation with Implicit Differentiable OptimizatioN**

> **POSEIDON** brings differentiable pose estimation to deep keypoint models by integrating Perspective-n-Point (PnP) solving with implicit gradients — diving deep into spatial reasoning for robust 3D localization.

---

## Industrial Project Overview

This project enhances the [YOLO-NAS](https://github.com/Deci-AI/super-gradients) architecture to support **vision-based landing** (VBL) by injecting **differentiable pose estimation** into the learning process. Instead of only optimizing bounding box and keypoint quality, we directly **supervise the 3D pose** using a differentiable PnP solver in the loss function.

---

## Motivation

Traditional object detection models are not designed to meet **aerospace-grade accuracy tolerances** like those defined by the **Instrument Landing System (ILS)**. This project aims to bridge that gap by:

- Embedding **camera pose estimation** directly into the learning objective  
- Enabling models to learn **keypoint configurations** that are optimal for 6-DoF localization  
- Leveraging **implicit gradients** for backpropagation through the PnP optimization  

---

## Key Features

- ✅ YOLO-NAS based backbone for real-time inference  
- ✅ Predict 2D keypoints of known 3D landmarks (e.g., runway corners)  
- ✅ Differentiable P3P (Perspective-Three-Point) solver using:  
  - 🌀 **Implicit differentiation** (Bo Chen et al. [2])  
- ✅ Pose-aware loss combining PnP and OKS  
- ✅ Configurable ILS-based error tolerances in the training objective 
