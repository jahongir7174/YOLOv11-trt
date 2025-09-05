# YOLOv11-trt

YOLOv11-trt is an advanced, high-performance object detection framework based on the YOLOv11 architecture, optimized for NVIDIA TensorRT. It supports ONNX export, TensorRT engine building, and fast inference on GPU, making it suitable for real-time applications.

## Features
- **YOLOv11 Model Family**: Includes various model sizes (n, s, m, l, x) for different accuracy/speed trade-offs.
- **TensorRT Integration**: Export models to ONNX and build highly optimized TensorRT engines for deployment.
- **Flexible Inference**: Modular code for inference.
- **Docker Support**: Ready-to-use Dockerfile and docker-compose for reproducible GPU-accelerated environments.

## Project Structure
```
YOLOv11-trt/
├── main.py              # Main entry point: export, engine build, benchmark, inference
├── nets/nn.py           # Model architectures (YOLOv11 variants, backbone, FPN, head)
├── utils/util.py        # Utility functions
├── utils/args.yaml      # Hyperparameters and class names
├── weights/             # Pretrained weights, ONNX, TensorRT engines, logs
├── data/                # Example images for inference
├── Dockerfile           # Docker build for NVIDIA TensorRT
├── docker-compose.yml   # Compose file for easy deployment
├── requirements.txt     # Python dependencies
```

## Setup
### 1. Docker (Recommended)
Ensure you have [NVIDIA Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) and a compatible GPU driver installed.

```bash
git clone https://github.com/jahongir7174/YOLOv11-trt.git
cd YOLOv11-trt
docker-compose up --build
```
This will build the image, mount your code, and run inference on sample images in `data/`.

### 2. Manual (Native)
- Install Python 3.8+ and CUDA/cuDNN/TensorRT compatible with your GPU.
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
  ```

## Usage
Run the main script with different options:

- **Export to ONNX:**
  ```bash
  python3 main.py --onnx
  ```
- **Build TensorRT Engine:**
  ```bash
  python3 main.py --engine
  ```
- **Benchmark Inference Speed:**
  ```bash
  python3 main.py --benchmark
  ```
- **Run Inference on Images:**
  ```bash
  python3 main.py --run
  ```

Arguments:
- `--input-size`: Input image size (default: 640)
- `--batch-size`: Batch size (default: 1)

## Model Zoo
Pretrained weights and engines are provided in the `weights/` directory:
- `v11_n.pt`, `v11_s.pt`, `v11_m.pt`, `v11_l.pt`, `v11_x.pt`: PyTorch weights
- `v11_n.onnx`: ONNX export
- `v11_n.engine`: TensorRT engine

## Dataset & Classes
- Example images are in `data/`.
- Class names and hyperparameters are defined in `utils/args.yaml` (COCO 80 classes by default).

## Code Overview
- **main.py**: Handles argument parsing, ONNX export, engine build, benchmarking, and inference.
- **nets/nn.py**: Defines YOLOv11 model variants, backbone, FPN, head, and export/fuse utilities.
- **utils/util.py**: Contains other utilities.

## References
- [YOLOv11](https://github.com/jahongir7174/YOLOv11-pt)
- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt)
- [ONNX](https://onnx.ai/)

---
For questions or contributions, please open an issue or pull request.
