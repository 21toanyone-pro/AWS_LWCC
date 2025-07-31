import os
import uuid
import shutil
from pathlib import Path
import time

import cv2
import numpy as np
import torch                                  # ← 추가
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import List

# video_any_hd.py에서 필요한 함수와 변수들을 가져옵니다.
from video_any_hd import (
    initialize_onnx_model,
    analyze_batch_onnx,
    analyze_frame_onnx,
    analyze_video_hd,
    HD_RESOLUTION
)

app = FastAPI(
    title="LWCC Video Analysis API",
    description="비디오를 업로드하여 인원수를 분석하는 API",
    version="1.0.0"
)

@app.on_event("startup")
def startup_event():
    """
    API 서버가 시작될 때 모델을 미리 로드하여
    첫 요청 시 지연을 방지합니다.
    GPU 가능 여부에 따라 FP16/FP32 모델을 선택합니다.
    """
    # 1) GPU 사용 가능 여부 판단
    use_gpu = torch.cuda.is_available()
    print(f"▶ GPU 사용 가능: {use_gpu}")

    # 2) onnx_models 폴더 내 FP16/FP32 파일 경로
    p16 = Path("onnx_models/lwcc_dm_count_fp16.onnx")
    p32 = Path("onnx_models/lwcc_dm_count.onnx")

    # 3) 모델 선택 로직
    if use_gpu and p16.exists():
        target = p16
    elif not use_gpu and p16.exists():
        print("⚠️ GPU가 없으므로 FP16 모델 대신 FP32 모델을 사용합니다.")
        target = p32
    else:
        target = p32

    print(f"▶ ONNX 모델 초기화 시작 (use_gpu={use_gpu}) → {target.name}")
    initialize_onnx_model(use_gpu=use_gpu, model_path=str(target))

    # 워밍업: 첫 추론으로 지연 제거
    print("모델 워밍업을 위해 더미 추론을 실행합니다...")
    try:
        dummy_frame = np.zeros((HD_RESOLUTION[1], HD_RESOLUTION[0], 3), dtype=np.uint8)
        analyze_frame_onnx(dummy_frame, already_hd=True)
        print("모델 워밍업 완료. 이제 실제 요청을 빠르게 처리할 수 있습니다.")
    except Exception as e:
        print(f"모델 워밍업 중 오류 발생: {e}")


@app.get("/", tags=["Health Check"])
async def health_check():
    return {"status": "ok", "message": "LWCC Analysis Server is running."}


@app.post("/analyze_batch/", tags=["Batch Image Analysis"])
async def analyze_batch_endpoint(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="이미지 파일이 없습니다.")
    start_time = time.time()

    frames, names = [], []
    for file in files:
        data = await file.read()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            frames.append(img)
            names.append(file.filename)

    counts = analyze_batch_onnx(frames)
    results = []
    for fn, cnt in zip(names, counts):
        results.append({
            "filename": fn,
            "status": "success" if cnt is not None else "error",
            "count": round(cnt, 1) if cnt is not None else None,
            "count_int": int(round(cnt)) if cnt is not None else None,
        })

    total_time = time.time() - start_time
    print(f"배치 분석 완료: {len(results)}개, {total_time:.3f}초 소요")
    return results


@app.post("/analyze_image/", tags=["Image Analysis"])
async def analyze_image_endpoint(file: UploadFile = File(...)):
    try:
        data = await file.read()
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="유효하지 않은 이미지 파일입니다.")

        print(f"이미지 분석 시작: {file.filename}")
        count, proc = analyze_frame_onnx(frame, already_hd=True)
        if count is None:
            raise HTTPException(status_code=500, detail="프레임 분석 실패")

        resp = {
            "count": round(count, 1),
            "count_int": int(round(count)),
            "server_process_time_seconds": round(proc, 3)
        }
        print(f"이미지 분석 결과: {resp}")
        return resp

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분석 중 오류 발생: {e}")


@app.post("/analyze/", tags=["Video Analysis"])
async def analyze_video_endpoint(file: UploadFile = File(...)):
    session_id = str(uuid.uuid4())
    tmp = Path(f"/tmp/{session_id}")
    vdir = tmp / "video"
    rdir = tmp / "results"
    vdir.mkdir(parents=True, exist_ok=True)
    rdir.mkdir(parents=True, exist_ok=True)

    video_path = vdir / file.filename
    try:
        with open(video_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)

        json_result = analyze_video_hd(video_path, rdir, tmp)
        return JSONResponse(content=json_result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분석 중 오류 발생: {e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
