import torch
from ultralytics import YOLO

# 1. Load your 6-channel hacked model
# We use the internal .model because we only need the computation graph
model_wrapper = YOLO('model/artifacts/yolov8n_6ch.pt')
model = model_wrapper.model.fuse().eval()  # Fuse layers for satellite efficiency

# 2. Create a 6-channel dummy tensor (B2, B3, B4, B8, B11, B12)
# Shape: (Batch, Channels, Height, Width)
dummy_input = torch.randn(1, 6, 640, 640)

# 3. Export to ONNX
output_path = "model/artifacts/osp_yolov8n_int8.onnx"
print(f"🚀 Exporting 6-channel OSP model to {output_path}...")

torch.onnx.export(
    model,
    dummy_input,
    output_path,
    export_params=True,
    opset_version=12,
    do_constant_folding=True,
    input_names=['images'],
    output_names=['output'],
    dynamic_axes={'images': {0: 'batch'}, 'output': {0: 'batch'}}
)

print("✅ Mission-Ready: 6-channel ONNX exported successfully.")
