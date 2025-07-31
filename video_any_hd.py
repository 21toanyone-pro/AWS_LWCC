#!/usr/bin/env python
# video_analyzer_hd.py
# HD 해상도로 비디오 파일을 초당 1프레임 분석하여 결과 저장

import os
import cv2
import time
import inspect
import argparse
import json
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
import onnxruntime as ort

# LWCC 캐시 디렉토리를 홈 디렉토리로 설정
os.environ['HOME'] = os.path.expanduser('~')
home_dir = os.path.expanduser('~')
lwcc_cache_dir = os.path.join(home_dir, '.lwcc')
os.makedirs(lwcc_cache_dir, exist_ok=True)
os.makedirs(os.path.join(lwcc_cache_dir, 'weights'), exist_ok=True)

# LWCC 관련 환경 변수 설정
os.environ['LWCC_CACHE_DIR'] = lwcc_cache_dir
os.environ['TORCH_HOME'] = lwcc_cache_dir
os.environ['XDG_CACHE_HOME'] = lwcc_cache_dir

from lwcc import LWCC

# ─── 글로벌 변수 ──────────────────────────────────────────────────
HD_RESOLUTION = (1280, 720)  # HD 해상도
model = None
device = None
ort_session = None  # ONNX Runtime 세션을 위한 전역 변수

def initialize_model(use_gpu=False):
    """LWCC 모델 초기화 (PyTorch 버전)"""
    global model, device

    # 디바이스 설정
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print("🔧 GPU 사용 (HD 최적화)")
    else:
        device = torch.device("cpu")
        print("🔧 CPU 사용 (HD 해상도)")

    # 모델 로드
    print(f"📥 LWCC PyTorch 모델 로딩 중... (HD {HD_RESOLUTION[0]}x{HD_RESOLUTION[1]})")
    mdl_kwargs = dict(model_name="DM-Count", model_weights="SHA")
    if 'device' in inspect.signature(LWCC.load_model).parameters:
        mdl_kwargs['device'] = device

    model = LWCC.load_model(**mdl_kwargs).to(device).eval()
    print("✅ LWCC 모델 로딩 완료!")


def initialize_onnx_model(use_gpu: bool = False, model_path: str = None):
    """
    ONNX Runtime 세션을 초기화합니다.
    - use_gpu: CUDA provider를 쓸지 여부
    - model_path: 로드할 ONNX 파일 경로 (기본: onnx_models/lwcc_dm_count.onnx)
    """
    global ort_session
    default_path = Path("onnx_models/lwcc_dm_count.onnx")
    model_path = Path(model_path) if model_path else default_path

    if not model_path.exists():
        raise FileNotFoundError(f"ONNX 모델 파일을 찾을 수 없습니다: {model_path}")

    available = ort.get_available_providers()
    if use_gpu and "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider"]
    else:
        if use_gpu:
            print("⚠️ CUDAExecutionProvider 미사용(사용 불가). CPUExecutionProvider로 대체합니다.")
        providers = ["CPUExecutionProvider"]

    print(f"🔖 ONNXRuntime providers 선택: {providers}")
    sess_options = ort.SessionOptions()
    ort_session = ort.InferenceSession(str(model_path), sess_options, providers=providers)
    print(f"✅ ort_session 초기화 완료: model={model_path.name}")


def analyze_frame_hd(frame, already_hd=False):
    """
    OpenCV 프레임을 HD 해상도로 리사이징 후
    PyTorch LWCC 모델로 인원수 분석
    """
    try:
        start_time = time.time()

        # HD 해상도로 리사이징
        hd_frame = frame if already_hd else cv2.resize(frame, HD_RESOLUTION)

        # BGR -> PIL(RGB)
        pil_img = Image.fromarray(cv2.cvtColor(hd_frame, cv2.COLOR_BGR2RGB))

        # 모델 전처리 (long side 1000 기준 리사이징)
        long = max(pil_img.size)
        factor = 1000 / long
        resized_img = pil_img.resize(
            (int(pil_img.size[0] * factor), int(pil_img.size[1] * factor)),
            Image.BILINEAR
        )

        # ToTensor & Normalize
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        img_tensor = trans(resized_img).unsqueeze(0).to(device)

        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        with torch.no_grad():
            output = model(img_tensor)

        density = output[0, 0].detach().cpu().numpy()
        count = float(density.sum())
        processing_time = time.time() - start_time
        return count, processing_time

    except Exception as e:
        print(f"❌ 프레임 분석 오류: {e}")
        return None, 0


def analyze_frame_onnx(frame, already_hd=False):
    """
    ONNX Runtime을 사용하여 프레임 인원수 분석
    """
    global ort_session
    if ort_session is None:
        raise RuntimeError("ONNX Runtime 세션이 초기화되지 않았습니다. initialize_onnx_model()을 먼저 호출하세요.")

    try:
        start_time = time.time()
        hd_frame = frame if already_hd else cv2.resize(frame, HD_RESOLUTION)
        pil_img = Image.fromarray(cv2.cvtColor(hd_frame, cv2.COLOR_BGR2RGB))

        long = max(pil_img.size)
        factor = 1000 / long
        resized_img = pil_img.resize(
            (int(pil_img.size[0] * factor), int(pil_img.size[1] * factor)),
            Image.BILINEAR
        )

        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        img_tensor = trans(resized_img).unsqueeze(0)

        ort_inputs = {ort_session.get_inputs()[0].name: img_tensor.numpy()}
        ort_outs = ort_session.run(None, ort_inputs)
        density = ort_outs[0][0, 0]
        count = float(density.sum())
        processing_time = time.time() - start_time
        return count, processing_time

    except Exception as e:
        print(f"❌ ONNX 프레임 분석 오류: {e}")
        return None, 0


def analyze_batch_onnx(frames):
    """
    ONNX Runtime을 사용하여 여러 프레임 배치 분석
    """
    global ort_session
    if ort_session is None:
        raise RuntimeError("ONNX Runtime 세션이 초기화되지 않았습니다.")

    if not frames:
        return []

    try:
        # 전처리
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        tensors = []
        for frame in frames:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            long = max(pil_img.size)
            factor = 1000 / long
            resized_img = pil_img.resize(
                (int(pil_img.size[0] * factor), int(pil_img.size[1] * factor)),
                Image.BILINEAR
            )
            tensors.append(trans(resized_img))

        batch = torch.stack(tensors).numpy()
        ort_inputs = {ort_session.get_inputs()[0].name: batch}
        ort_outs = ort_session.run(None, ort_inputs)

        counts = np.sum(ort_outs[0], axis=(2, 3)).flatten().tolist()
        return counts

    except Exception as e:
        print(f"❌ ONNX 배치 분석 오류: {e}")
        return [None] * len(frames)


def analyze_video_hd(video_path, output_dir, temp_dir):
    """HD 해상도로 비디오 분석 (초당 1프레임) + 결과 저장"""
    video_name = Path(video_path).stem
    print(f"\n🎬 [{video_name}] HD 분석 시작...")

    # 출력 파일
    txt_file  = Path(output_dir) / f"{video_name}_hd_analysis.txt"
    csv_file  = Path(output_dir) / f"{video_name}_hd_analysis.csv"
    json_file = Path(output_dir) / f"{video_name}_hd_analysis.json"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ [{video_name}] 비디오를 열 수 없습니다.")
        return

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration     = total_frames / fps

    print(f"📹 원본 FPS: {fps:.1f}, 총 프레임: {total_frames}, 길이: {duration:.1f}초")
    print(f"📺 HD 해상도: {HD_RESOLUTION[0]}x{HD_RESOLUTION[1]}")
    print("🎯 분석 간격: 1초당 1프레임")

    results = []
    frame_interval = int(fps)
    total_seconds  = int(duration)

    for second in range(total_seconds + 1):
        frame_number = second * frame_interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            break

        count, proc_time = analyze_frame_hd(frame)
        if count is not None:
            ts = str(timedelta(seconds=second))
            results.append({
                'second': second,
                'timestamp': ts,
                'frame_number': frame_number,
                'count': round(count, 1),
                'count_int': int(round(count)),
                'process_time': round(proc_time, 3),
                'resolution': f"{HD_RESOLUTION[0]}x{HD_RESOLUTION[1]}"
            })
            print(f"🎯 [{video_name}] {ts} | 👥 {count:.1f}명 | ⏱️ {proc_time:.3f}초")
        else:
            print(f"❌ [{video_name}] {second}초 분석 실패")

    cap.release()

    # 결과 저장
    print("\n💾 결과 저장 중...")
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write(f"HD LWCC 비디오 인원수 분석 결과\n파일: {video_name}\n")
        for r in results:
            f.write(f"{r['timestamp']} | {r['count']}명 | {r['process_time']}초\n")

    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write("초,타임스탬프,프레임번호,인원수,정수인원수,처리시간,해상도\n")
        for r in results:
            f.write(f"{r['second']},{r['timestamp']},{r['frame_number']},"
                    f"{r['count']},{r['count_int']},{r['process_time']},{r['resolution']}\n")

    analysis_data = {
        'video_info': {
            'filename': video_name, 'fps': fps,
            'total_frames': total_frames,
            'duration_seconds': duration,
            'resolution': f"{HD_RESOLUTION[0]}x{HD_RESOLUTION[1]}"
        },
        'analysis_info': {
            'analysis_time': datetime.now().isoformat(),
            'frame_interval': frame_interval,
            'total_analyzed_frames': len(results)
        },
        'results': results
    }
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_data, f, ensure_ascii=False, indent=2)

    # 요약 출력
    if results:
        avg = sum(r['count'] for r in results) / len(results)
        print(f"\n📊 [{video_name}] 분석 완료! 평균 인원수: {avg:.1f}명")
    return analysis_data


def main():
    parser = argparse.ArgumentParser("HD LWCC Video Analyzer")
    parser.add_argument("--video-dir",   default="video",     help="비디오 파일 폴더")
    parser.add_argument("--output-dir",  default="results_hd", help="결과 저장 폴더")
    parser.add_argument("--use-gpu",     action="store_true", help="GPU 사용")
    parser.add_argument("--temp-dir",    default="/tmp",       help="임시 파일 저장 경로")
    args = parser.parse_args()

    print("🎯 HD LWCC 비디오 인원수 분석기")
    initialize_model(args.use_gpu)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    videos = []
    for ext in ['*.mp4','*.avi','*.mov','*.mkv','*.flv','*.wmv']:
        videos += list(Path(args.video_dir).glob(ext))
    if not videos:
        print(f"❌ 비디오 파일을 찾을 수 없습니다: {args.video_dir}")
        return

    for i, vp in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] {vp.name}")
        analyze_video_hd(vp, out_dir, args.temp_dir)

    print(f"\n🎉 모든 비디오 분석 완료! 결과 폴더: {out_dir}")


if __name__ == "__main__":
    main()
