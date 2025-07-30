import os
import uuid
import shutil
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, List
from fastapi.responses import JSONResponse

# video_any_hd.py에서 필요한 함수와 변수들을 가져옵니다.
from video_any_hd import (
    initialize_model,
    analyze_frame_hd, # 워밍업을 위해 analyze_frame_hd 직접 사용
    analyze_video_hd, # 기존 기능 유지를 위해 그대로 둠
    HD_RESOLUTION
)

# ─── FastAPI 앱 설정 ──────────────────────────────────────────────────

app = FastAPI(
    title="LWCC Video Analysis API",
    description="비디오를 업로드하여 인원수를 분석하는 API",
    version="1.0.0"
)

# ─── 모델 로딩 ────────────────────────────────────────────────────────

@app.on_event("startup")
def startup_event():
    """
    API 서버가 시작될 때 모델을 미리 로드하여
    첫 요청 시 지연을 방지합니다.
    """
    print("🚀 API 서버 시작... 모델을 초기화합니다.")
    # GPU 사용을 원하시면 use_gpu=True로 설정하세요.
    # AWS에 GPU 인스턴스를 사용하는 경우 True로 설정하는 것이 좋습니다.
    initialize_model(use_gpu=True)
    print("✅ 모델 로딩 완료.")

    print("🔥 모델 워밍업(Warm-up)을 위해 더미 추론을 실행합니다...")
    try:
        # 모델이 기대하는 입력과 유사한 더미 데이터 생성 (HD 해상도의 검은색 이미지)
        dummy_frame = np.zeros((HD_RESOLUTION[1], HD_RESOLUTION[0], 3), dtype=np.uint8)
        # 첫 추론을 미리 실행하여 모델을 예열
        analyze_frame_hd(dummy_frame, already_hd=True)
        print("✅ 모델 워밍업 완료. 이제 실제 요청을 빠르게 처리할 수 있습니다.")
    except Exception as e:
        print(f"⚠️ 모델 워밍업 중 오류 발생: {e}")


# ─── API 엔드포인트 ──────────────────────────────────────────────────

@app.get("/", tags=["Health Check"])
async def health_check():
    """서버가 정상적으로 실행 중인지 확인하는 간단한 엔드포인트입니다."""
    return {"status": "ok", "message": "LWCC Analysis Server is running."}


@app.post("/analyze_batch/", tags=["Batch Image Analysis"])
async def analyze_batch_endpoint(files: List[UploadFile] = File(...)):
    """
    이미지 배치(목록)를 업로드하면 각각을 분석하고 결과 목록을 JSON으로 반환합니다.
    """
    if not files:
        raise HTTPException(status_code=400, detail="이미지 파일이 없습니다.")

    results = []
    print(f"➡️  이미지 배치 수신: {len(files)}개")

    for file in files:
        try:
            # 1. 업로드된 이미지 파일을 메모리로 읽기
            contents = await file.read()
            
            # 2. 이미지 데이터를 OpenCV 프레임으로 디코딩
            nparr = np.frombuffer(contents, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                results.append({"filename": file.filename, "status": "error", "detail": "유효하지 않은 이미지 파일"})
                continue

            # 3. 프레임 분석 실행
            count, process_time = analyze_frame_hd(frame, already_hd=True)

            if count is None:
                results.append({"filename": file.filename, "status": "error", "detail": "서버에서 프레임 분석 중 오류 발생"})
                continue
            
            # 성공 결과 추가
            results.append({
                "filename": file.filename,
                "status": "success",
                "count": round(count, 1),
                "count_int": int(round(count)),
                "server_process_time_seconds": round(process_time, 3)
            })

        except Exception as e:
            results.append({"filename": file.filename, "status": "error", "detail": f"처리 중 예외 발생: {str(e)}"})
    
    print(f"⬅️  배치 분석 결과 전송: {len(results)}개")
    return results


@app.post("/analyze_image/", tags=["Image Analysis"])
async def analyze_image_endpoint(file: UploadFile = File(...)):
    """
    단일 이미지를 업로드하면 분석을 수행하고 결과를 JSON으로 반환합니다.
    클라이언트가 HD 해상도로 리사이징한 이미지를 보내는 것을 가정합니다.
    """
    try:
        print(f"➡️  이미지 수신: {file.filename} ({file.content_type})")
        # 1. 업로드된 이미지 파일을 메모리로 읽기
        contents = await file.read()
        
        # 2. 이미지 데이터를 OpenCV 프레임으로 디코딩
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        print("...이미지 디코딩 완료.")

        if frame is None:
            raise HTTPException(status_code=400, detail="유효하지 않은 이미지 파일입니다.")

        # 3. 프레임 분석 실행 (video_any_hd.py의 함수 재사용)
        # 클라이언트가 이미 HD로 보냈으므로, 중복 리사이징을 건너뜁니다.
        print("...모델 분석 시작 (시간이 소요될 수 있습니다)...")
        count, process_time = analyze_frame_hd(frame, already_hd=True)
        print(f"...모델 분석 완료. (소요 시간: {process_time:.3f}초)")

        if count is None:
             raise HTTPException(status_code=500, detail="서버에서 프레임 분석 중 오류 발생")

        response_data = {
            "count": round(count, 1),
            "count_int": int(round(count)),
            "server_process_time_seconds": round(process_time, 3)
        }
        print(f"⬅️  응답 전송: {response_data}")
        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분석 중 오류 발생: {str(e)}")


@app.post("/analyze/", tags=["Video Analysis"])
async def analyze_video_endpoint(file: UploadFile = File(...)):
    """
    비디오 파일을 업로드하면 분석을 수행하고 결과를 JSON으로 반환합니다.
    """
    # 임시 디렉토리 생성 (고유한 ID 사용)
    session_id = str(uuid.uuid4())
    temp_dir = Path(f"/tmp/{session_id}")
    temp_video_dir = temp_dir / "video"
    temp_results_dir = temp_dir / "results"
    temp_video_dir.mkdir(parents=True, exist_ok=True)
    temp_results_dir.mkdir(parents=True, exist_ok=True)

    video_path = temp_video_dir / file.filename
    
    try:
        # 1. 업로드된 비디오 파일을 임시 저장
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. 비디오 분석 실행
        # analyze_video_hd 함수는 결과를 파일로 저장하므로, 해당 파일을 읽어와야 합니다.
        json_result = analyze_video_hd(video_path, temp_results_dir, temp_dir)

        return JSONResponse(content=json_result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분석 중 오류 발생: {str(e)}")
    finally:
        # 3. 임시 디렉토리 및 파일 정리
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    # uvicorn api:app --host 0.0.0.0 --port 8000 --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)