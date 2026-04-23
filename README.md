# Adversarial-Attack-on-Object-Detection in Duckietown


Adversarial attack research repository for Duckietown perception.

This project focuses on generating and evaluating adversarial perturbations against a YOLOv8-based detection pipeline running in the Duckietown GPU server. The current integrated attack is **FGSM** (Fast Gradient Sign Method), applied to the **YOLO input image only** before inference.

## Threat Model Primer

### White-Box Attacks

A white-box attack assumes access to the target model internals:
- architecture
- weights
- gradients/loss

Because gradients are available, the attacker can compute directed perturbations that efficiently maximize model loss.

### Black-Box Attacks

A black-box attack assumes no direct gradient/model access.

Typical black-box settings:
- query-only API access
- transfer attacks from surrogate models
- score-based or decision-based search

Black-box attacks are usually less sample-efficient than white-box methods but often more realistic in deployment settings.

## FGSM (Fast Gradient Sign Method)

FGSM is a one-step white-box attack:

```text
x_adv = clip(x + epsilon * sign(grad_x L(theta, x, y)))
```


Where:
- `x` is the clean input image
- `epsilon` controls perturbation strength
- `L` is the attack loss

In this repo, FGSM is integrated with YOLOv8 loss heads (`box`, `cls`, `dfl`) and uses pseudo-labels from YOLO self-detections.

## Current Pipeline Integration

Server flow per frame:
1. Receive JPEG frame from client
2. Build FGSM adversarial image (YOLO branch only)
3. Run YOLO on adversarial image
4. Run lane segmentation on clean image
5. Run tracking and control
6. Return `v, omega` command

So lane segmentation is currently **not** adversarially perturbed.

## Repository Layout

- `gpu_server.py` - TCP GPU server loop and frame handling
- `lane_pipeline.py` - inference + control pipeline, FGSM integration point
- `fgsm_yolov8.py` - FGSM attack module with YOLOv8 loss access
- `packages/my_package/src/vehicle_client.py` - remote Duckiebot client
- `packages/my_package/src/vehicle_local_inference.py` - local on-board inference node
- `segmentation/` - lane segmentation models and training code
- `ByteTrack/` - tracker implementation
- `weight/` - runtime model weights

## Setup

Install dependencies in your Python environment:

```bash
pip install -r requirements.txt
```

## Run Server With FGSM

From repo root:

```bash
FGSM_ENABLE=1 \
FGSM_EPSILON=0.015686 \
FGSM_BOX_GAIN=7.5 \
FGSM_CLS_GAIN=0.5 \
FGSM_DFL_GAIN=1.5 \
FGSM_PSEUDO_CONF=0.25 \
FGSM_PSEUDO_MAX_DET=30 \
LANE_VERBOSE=1 \
./scripts/run_gpu_server.sh
```

Expected FGSM logs:

```text
[fgsm] enabled eps=...
[fgsm] loss(total=..., box=..., cls=..., dfl=..., targets=...)
```

If `targets=0`, pseudo-label detection found no objects for that frame.

## Key Environment Variables

### Server

```bash
GPU_SERVER_HOST=0.0.0.0
GPU_SERVER_PORT=5001
SHOW_GUI=1
LANE_VERBOSE=0
YOLO_WEIGHTS=weight/yolo.pt
SEG_WEIGHTS=weight/segment_depthwise_se.pth
```

### FGSM

```bash
FGSM_ENABLE=0
FGSM_EPSILON=0.015686      # ~= 4/255
FGSM_BOX_GAIN=7.5
FGSM_CLS_GAIN=0.5
FGSM_DFL_GAIN=1.5
FGSM_PSEUDO_CONF=0.25
FGSM_PSEUDO_MAX_DET=30
```

Per-channel max pixel change for one FGSM step is approximately `epsilon * 255` intensity levels (before clipping).

Examples:
- `epsilon=4/255` -> max ~4 levels
- `epsilon=8/255` -> max ~8 levels
- `epsilon=16/255` -> max ~16 levels



## Scope and Intended Use

This repository is intended for adversarial robustness research, testing, and controlled experiments in Duckietown-style perception pipelines.

Use responsibly and only in authorized environments.
