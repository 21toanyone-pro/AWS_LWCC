import torch
import os
from pathlib import Path

# Set up environment to load the model correctly
os.environ['HOME'] = os.path.expanduser('~')
home_dir = os.path.expanduser('~')
lwcc_cache_dir = os.path.join(home_dir, '.lwcc')
os.environ['LWCC_CACHE_DIR'] = lwcc_cache_dir
os.environ['TORCH_HOME'] = lwcc_cache_dir
os.environ['XDG_CACHE_HOME'] = lwcc_cache_dir

from lwcc import LWCC

def export_model_to_onnx():
    """
    Loads the PyTorch LWCC model and exports it to ONNX format.
    """
    print("📥 LWCC PyTorch 모델 로딩 중...")
    # Load the original PyTorch model and set it to evaluation mode on CPU
    model = LWCC.load_model(model_name="DM-Count", model_weights="SHA").cpu().eval()
    print("✅ PyTorch 모델 로딩 완료.")

    # Define a dummy input tensor with a representative size.
    # The model preprocesses images to have a longest side of 1000.
    # Let's use a 1280x720 -> 1000x562 example.
    # Shape: (batch_size, channels, height, width)
    dummy_input = torch.randn(1, 3, 562, 1000, requires_grad=False)

    # Define the output path
    onnx_model_dir = Path("onnx_models")
    onnx_model_dir.mkdir(exist_ok=True)
    onnx_model_path = onnx_model_dir / "lwcc_dm_count.onnx"

    print(f"🚀 모델을 ONNX 형식으로 변환 시작... -> {onnx_model_path}")

    # Export the model
    torch.onnx.export(
        model,                          # The model to export
        dummy_input,                    # A dummy input for tracing the model's graph
        str(onnx_model_path),           # Where to save the model
        export_params=True,             # Store the trained parameter weights inside the model file
        opset_version=11,               # The ONNX version to export the model to
        do_constant_folding=True,       # Whether to execute constant folding for optimization
        input_names=['input'],          # The model's input names
        output_names=['output'],        # The model's output names
        dynamic_axes={                  # Allow for variable-sized inputs
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    )

    print(f"✅ ONNX 모델 변환 성공! ({onnx_model_path})")

if __name__ == "__main__":
    export_model_to_onnx()