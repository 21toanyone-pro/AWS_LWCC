# convert_fp16.py
import onnx
from onnxconverter_common import float16

# 1) FP32 모델 로드
model_fp32 = onnx.load("onnx_models/lwcc_dm_count.onnx")
# 2) FP16으로 변환
model_fp16 = float16.convert_float_to_float16(model_fp32)
# 3) 저장
onnx.save(model_fp16, "onnx_models/lwcc_dm_count_fp16.onnx")
print("✅ FP16 ONNX 모델 생성:", "onnx_models/lwcc_dm_count_fp16.onnx")
