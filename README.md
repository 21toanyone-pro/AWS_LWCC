# LWCC 실시간 인원수 분석 API 서버

인원수 계수 모델(LWCC)을 활용하여 이미지 및 비디오 속 인원을 분석하는 API 서버. 

FastAPI를 기반으로 구축되었으며, ONNX Runtime을 통해 추론 성능을 최적화하여 CPU 환경에서 효율적으로 동작합니다.

여러 CCTV로 부터 이미지를 주기적(1분당 1장)으로 수집하여 배치로 처리, 그 결과를 JSON 파일로 저장.

## 주요 기능

- **API 서버**: FastAPI를 사용하여 비동기 방식으로 다중 요청을 처리.
- **성능 최적화**: PyTorch 모델을 ONNX 형식으로 변환후 사용하여 기존보다 빠른 추론 속도를 제공(4.5s -> 3.3s).
- **자동 모델 선택**: 서버 실행 환경(GPU/CPU)에 따라 최적화된 ONNX 모델(FP16/FP32)을 자동으로 선택하여 로드.
- **다양한 분석 엔드포인트**:
  - `/analyze_image/`: 단일 이미지 분석
  - `/analyze_batch/`: 여러 이미지를 한 번에 배치 처리
  - `/`: 서버 상태 확인 (Health Check)
- **실시간 배치 클라이언트**:
  - 다중 CCTV 소스(비디오 파일, RTSP 등) 관리
  - 주기적으로(예: 1분마다) 모든 소스에서 프레임을 캡처
  - 캡처된 이미지들을 하나의 배치로 묶어 서버에 전송
  - 분석 결과를 타임스탬프가 포함된 JSON 파일로 로컬에 저장

## 프로젝트 구조

```
AWS_LWCC/
├── onnx_models/                # 변환된 ONNX 모델 저장 폴더
│   ├── lwcc_dm_count.onnx
│   └── lwcc_dm_count_fp16.onnx
├── analysis_results/           # 클라이언트가 생성한 JSON 결과 저장 폴더
├── lwcc/                       # LWCC 원본 라이브러리
├── api.py                      # FastAPI 메인 애플리케이션
├── client.py                   # 실시간 배치 분석 클라이언트
├── video_any_hd.py             # 핵심 분석 로직 (ONNX/PyTorch)
├── export_onnx.py              # PyTorch 모델을 ONNX로 변환하는 스크립트
├── convert_fp16.py             # ONNX(FP32) 모델을 FP16으로 변환하는 스크립트
├── requirements.txt            # Python 의존성 목록
└── README.md                   # 프로젝트 설명서
```

## 설치 및 설정

### 1. 저장소 복제

```bash
git clone <repository-url>
cd AWS_LWCC
```

### 2. 의존성 설치

프로젝트에 필요한 모든 Python 라이브러리를 설치합니다.

```bash
pip install -r requirements.txt
```

### 3. 로컬 `lwcc` 패키지 설치

프로젝트에 포함된 `lwcc` 라이브러리를 시스템에 설치합니다.

```bash
pip install .
```

### 4. ONNX 모델 변환 (최초 1회)

추론 성능을 최적화하기 위해 PyTorch 모델을 ONNX 형식으로 변환합니다.

**1) FP32 모델 생성 (CPU/GPU 공용)**
```bash
python export_onnx.py
```
이 명령어를 실행하면 `onnx_models/lwcc_dm_count.onnx` 파일이 생성됩니다.

**2) FP16 모델 생성 (GPU 전용, 선택 사항)**
GPU 환경에서 더 높은 성능을 원할 경우, FP16 형식으로 추가 변환합니다.
```bash
python convert_fp16.py
```
이 명령어를 실행하면 `onnx_models/lwcc_dm_count_fp16.onnx` 파일이 생성됩니다.

---

## 실행 방법

### 서버 실행

아래 명령어를 실행하여 API 서버를 시작합니다. 서버는 시작 시 GPU 사용 가능 여부를 자동으로 감지하고 최적의 ONNX 모델을 로드합니다.

```bash
python3 -m uvicorn api:app --host 0.0.0.0 --port 8000
```

서버가 정상적으로 시작되면 `http://<서버_IP>:8000` 주소로 API 요청을 받을 준비가 됩니다.

### 클라이언트 실행

클라이언트는 설정된 CCTV 소스에서 주기적으로 이미지를 캡처하여 서버로 전송하고, 분석 결과를 JSON 파일로 저장합니다.

**1) 클라이언트 설정 (`client.py`)**

`client.py` 파일 하단의 `if __name__ == "__main__":` 블록에서 다음 변수들을 사용 환경에 맞게 수정합니다.

- `SERVER_IP`: 실행 중인 API 서버의 공인 IP 주소.
- `CCTV_SOURCES`: 분석할 CCTV 소스를 딕셔너리 형태로 정의합니다. (예: `{"CCTV_01": "video/test_1.avi"}`)
- `INTERVAL`: 이미지 수집 및 전송 주기 (초 단위).
- `OUTPUT_JSON_DIR`: 분석 결과가 저장될 폴더 이름.

**2) 클라이언트 실행**

```bash
python client.py
```

클라이언트를 실행하면 1분마다 모든 CCTV의 현재 프레임을 서버로 전송하고, `analysis_results` 폴더에 `analysis_YYYYMMDD_HHMMSS.json` 형식으로 결과가 저장됩니다.

## API 엔드포인트

- **`GET /`**: **Health Check**
  - 서버가 정상적으로 실행 중인지 확인합니다.
  - 응답: `{"status": "ok", "message": "..."}`

- **`POST /analyze_batch/`**: **이미지 배치 분석**
  - 여러 이미지 파일을 한 번의 요청으로 전송하여 분석합니다.
  - 요청: `files` 필드에 이미지 파일 목록을 `multipart/form-data` 형식으로 전송합니다.
  - 응답: 각 파일에 대한 분석 결과 목록 (JSON).

- **`POST /analyze_image/`**: **단일 이미지 분석**
  - 하나의 이미지 파일을 전송하여 분석합니다.
  - 요청: `file` 필드에 이미지 파일을 `multipart/form-data` 형식으로 전송합니다.
  - 응답: 단일 분석 결과 (JSON).
