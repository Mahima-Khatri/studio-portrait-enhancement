# Studio Portrait Enhancement (Identity Preserving)

## Candidate Name
Mahima Khatri

## Role
Machine Learning Engineer – Round 1 (FOG)

---

## Problem Statement
Raw human portrait images captured in uncontrolled environments (low light, cluttered background, noise, motion blur) need to be converted into **studio-quality portraits** while preserving the **original facial identity** and maintaining **natural skin texture**.

The goal is to simulate a professional studio look using **computer vision + AI models**, with fast inference and minimal visual artifacts.

---

## Input
- Raw human portrait image (.jpg / .png)
- Image conditions may include:
  - Low light / uneven lighting
  - Noisy or cluttered background
  - Low contrast
  - Minor face artifacts

---

## Output
A studio-quality portrait image with:
- Background bokeh / blur
- Improved face clarity
- Enhanced sharpness and contrast
- Preserved skin texture
- Maintained original facial identity

---

## Approach & Pipeline

This solution is implemented in **Google Colab** using Python and the following AI / CV techniques:

### 1. Tool Installation
Required libraries are installed dynamically:
- `GFPGAN` → Face restoration (identity-preserving)
- `rembg` → Background segmentation
- `OpenCV` → Image processing
- `BasicSR` → Model utilities

A compatibility patch is applied to ensure smooth execution in Colab.

---

### 2. Gentle Denoising
```python
cv2.fastNlMeansDenoisingColored
