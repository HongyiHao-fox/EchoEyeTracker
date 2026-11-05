# Echo Eye Tracker

## Overview

Echo Eye Tracker is a reproduction project of the Cornell University paper:

**["GazeTrak: Exploring Acoustic-based Eye Tracking on a Glass Frame"](https://arxiv.org/abs/2402.14634)**

The goal of this project is to reproduce and implement an acoustic-based eye-tracking system using the methods proposed in the paper. This is also my first deep learning project, covering the complete pipeline from acoustic signal processing to deep learning-based gaze estimation.

The original GazeTrak system introduces a novel wearable eye-tracking approach based on **active acoustic sensing**. Instead of relying on cameras, the system uses a speaker and multiple microphones mounted on a glasses frame to emit inaudible FMCW acoustic signals and capture the reflected signals from the eye region. These signals are transformed into echo profiles and processed by a customized deep learning model to continuously estimate gaze positions. :contentReference[oaicite:0]{index=0}

The paper demonstrates that acoustic-based eye tracking can achieve competitive accuracy while significantly reducing power consumption compared with camera-based solutions. The proposed system achieves a gaze tracking accuracy of **3.6° within the same session** and **4.9° across different sessions**, with a refresh rate of up to **83.3 Hz**. :contentReference[oaicite:1]{index=1}

---

## Motivation

This project was inspired by a collaboration with a friend who is developing an electronic eye system for a fursuit studio.

Initially, we considered using computer vision approaches such as the **YOLO11** object detection model for gaze tracking because of its simplicity and strong performance. However, camera-based eye-tracking solutions, especially those using infrared cameras, introduce several limitations:

- High power consumption, which is unsuitable for lightweight wearable devices.
- Potential concerns regarding long-term infrared exposure near the eyes.
- Dependence on lighting conditions and camera placement.

While exploring alternative approaches, my collaborator **Meco_Joyang** discovered a demo video in April 2025:

["可自动进行眼动追踪的智能眼镜"](https://www.bilibili.com/video/BV1Mx421f7Q7e/)

After learning about acoustic-based eye tracking, we realized that this approach could provide a promising alternative. Compared with camera-based systems, acoustic sensing offers advantages such as:

- Lower power consumption
- Better privacy preservation
- Reduced dependence on lighting conditions
- Potential suitability for long-term wearable applications

These advantages make acoustic eye tracking an attractive direction for future smart glasses and physical AGI-related interfaces.

---

## Project Goal

This project aims to reproduce the main components of GazeTrak:

1. **Acoustic sensing pipeline**
   - Generate FMCW acoustic signals.
   - Capture reflected signals from the eye region.
   - Extract echo profiles through signal processing.

2. **Deep learning gaze estimation**
   - Build a neural network model inspired by the paper.
   - Train the model to map acoustic patterns to gaze coordinates.

3. **Hardware implementation**
   - Explore a glasses-mounted acoustic sensing prototype.
   - Investigate low-power embedded deployment possibilities.

The original paper uses two speakers and eight microphones mounted on a glasses frame. The collected acoustic signals are processed into echo profiles, which are then fed into a ResNet-18-based deep learning pipeline for gaze prediction. :contentReference[oaicite:2]{index=2}

---

## Reference

Li, Ke, et al.  
**"GazeTrak: Exploring Acoustic-based Eye Tracking on a Glass Frame."**  
ACM MobiCom 2024. :contentReference[oaicite:3]{index=3}