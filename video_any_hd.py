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
# from lwcc.util.functions import load_image # 메모리 내 처리를 위해 더 이상 사용 안 함

# ─── 글로벌 변수 ──────────────────────────────────────────────────
HD_RESOLUTION = (1280, 720)  # HD 해상도
model = None
device = None

def initialize_model(use_gpu=False):
    """LWCC 모델 초기화"""
    global model, device
    
    # 디바이스 설정
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print("🔧 GPU 사용 (HD 최적화)")
    else:
        device = torch.device("cpu")
        print("🔧 CPU 사용 (HD 해상도)")

    # 모델 로드
    print(f"📥 LWCC 모델 로딩 중... (HD {HD_RESOLUTION[0]}x{HD_RESOLUTION[1]})")
    mdl_kwargs = dict(model_name="DM-Count", model_weights="SHA")
    if 'device' in inspect.signature(LWCC.load_model).parameters:
        mdl_kwargs['device'] = device

    model = LWCC.load_model(**mdl_kwargs).to(device).eval()
    print("✅ LWCC 모델 로딩 완료!")

def analyze_frame_hd(frame, already_hd=False):
    """
    OpenCV 프레임을 HD 해상도로 리사이징 후 메모리에서 직접 인원수 분석
    :param frame: 분석할 OpenCV 프레임
    :param already_hd: 프레임이 이미 HD 해상도인지 여부
    """
    try:
        start_time = time.time()
        
        # 🎯 HD 해상도로 리사이징 (이미 HD가 아니면)
        if not already_hd:
            hd_frame = cv2.resize(frame, HD_RESOLUTION)
        else:
            hd_frame = frame
        
        # OpenCV 프레임(BGR)을 PIL 이미지(RGB)로 변환
        pil_img = Image.fromarray(cv2.cvtColor(hd_frame, cv2.COLOR_BGR2RGB))

        # lwcc.util.functions.load_image의 전처리 로직을 메모리에서 직접 수행
        # 1. 리사이징 (라이브러리 기본 동작)
        long = max(pil_img.size)
        factor = 1000 / long
        resized_img = pil_img.resize(
            (int(pil_img.size[0] * factor), int(pil_img.size[1] * factor)),
            Image.BILINEAR
        )

        # 2. 텐서 변환 및 정규화
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img_tensor = trans(resized_img).unsqueeze(0).to(device)
        
        # 추론
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        with torch.no_grad():
            output = model(img_tensor)
        
        density = output[0, 0].detach().cpu().numpy()
        count = float(density.sum())
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        return count, processing_time
        
    except Exception as e:
        print(f"❌ 프레임 분석 오류: {e}")
        return None, 0

def analyze_video_hd(video_path, output_dir, temp_dir):
    """HD 해상도로 비디오 분석 (초당 1프레임)"""
    
    video_name = Path(video_path).stem
    print(f"\n🎬 [{video_name}] HD 분석 시작...")
    
    # 출력 파일 경로
    txt_file = Path(output_dir) / f"{video_name}_hd_analysis.txt"
    csv_file = Path(output_dir) / f"{video_name}_hd_analysis.csv"
    json_file = Path(output_dir) / f"{video_name}_hd_analysis.json"
    
    # 비디오 캡처 초기화
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ [{video_name}] 비디오를 열 수 없습니다.")
        return
    
    # 비디오 정보
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"📹 원본 FPS: {fps:.1f}, 총 프레임: {total_frames}, 길이: {duration:.1f}초")
    print(f"📺 HD 해상도: {HD_RESOLUTION[0]}x{HD_RESOLUTION[1]}")
    print(f"🎯 분석 간격: 1초당 1프레임")
    
    # 결과 저장용 리스트
    results = []
    frame_interval = int(fps)  # 1초 간격
    total_seconds = int(duration)
    
    # 분석 시작
    for second in range(total_seconds + 1):
        frame_number = second * frame_interval
        
        # 프레임 위치 설정
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # HD 분석 (메모리에서 직접)
        count, process_time = analyze_frame_hd(frame)
        
        if count is not None:
            count_int = int(round(count))
            timestamp = str(timedelta(seconds=second))
            
            # 결과 저장
            result = {
                'second': second,
                'timestamp': timestamp,
                'frame_number': frame_number,
                'count': round(count, 1),
                'count_int': count_int,
                'process_time': round(process_time, 3),
                'resolution': f"{HD_RESOLUTION[0]}x{HD_RESOLUTION[1]}"
            }
            results.append(result)
            
            # 실시간 출력
            print(f"🎯 [{video_name}] {timestamp} | 👥 {count:.1f}명 ({count_int}명) | ⏱️  {process_time:.3f}초")
        
        else:
            print(f"❌ [{video_name}] {second}초 분석 실패")
    
    cap.release()
    
    # 결과 파일 저장
    print(f"\n💾 결과 저장 중...")
    
    # TXT 파일 저장
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write(f"HD LWCC 비디오 인원수 분석 결과\n")
        f.write(f"파일: {video_name}\n")
        f.write(f"해상도: {HD_RESOLUTION[0]}x{HD_RESOLUTION[1]} (HD)\n")
        f.write(f"분석 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"총 프레임: {total_frames}, FPS: {fps:.1f}, 길이: {duration:.1f}초\n")
        f.write(f"분석 간격: 1초당 1프레임\n")
        f.write("="*60 + "\n\n")
        
        for result in results:
            f.write(f"{result['timestamp']} | {result['count']}명 ({result['count_int']}명) | {result['process_time']}초\n")
    
    # CSV 파일 저장
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write("초,타임스탬프,프레임번호,인원수,정수인원수,처리시간,해상도\n")
        for result in results:
            f.write(f"{result['second']},{result['timestamp']},{result['frame_number']},{result['count']},{result['count_int']},{result['process_time']},{result['resolution']}\n")
    
    # JSON 파일 저장
    analysis_data = {
        'video_info': {
            'filename': video_name,
            'fps': fps,
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
    
    # 통계 출력
    print(f"\n📊 [{video_name}] 분석 완료!")
    print(f"✅ 분석된 프레임: {len(results)}개")
    print(f"📄 TXT 파일: {txt_file}")
    print(f"📊 CSV 파일: {csv_file}")
    print(f"💾 JSON 파일: {json_file}")
    
    if results:
        avg_count = sum(r['count'] for r in results) / len(results)
        max_count = max(r['count'] for r in results)
        min_count = min(r['count'] for r in results)
        avg_process_time = sum(r['process_time'] for r in results) / len(results)
        
        print(f"📈 평균 인원수: {avg_count:.1f}명")
        print(f"🔺 최대 인원수: {max_count:.1f}명")
        print(f"🔻 최소 인원수: {min_count:.1f}명")
        print(f"⏱️  평균 처리시간: {avg_process_time:.3f}초")
        
    return analysis_data # API에서 사용할 수 있도록 결과 반환

def main():
    # ─── CLI 설정 ──────────────────────────────────────────────────
    parser = argparse.ArgumentParser("HD LWCC Video Analyzer")
    parser.add_argument("--video-dir", default="video", help="비디오 파일이 있는 폴더")
    parser.add_argument("--output-dir", default="results_hd", help="결과 파일 저장 폴더")
    parser.add_argument("--use-gpu", action="store_true", help="GPU 사용")
    parser.add_argument("--temp-dir", default="/tmp", help="임시 파일 저장 경로")

    args = parser.parse_args()
    
    print("🎯 HD LWCC 비디오 인원수 분석기")
    print(f"📺 HD 해상도: {HD_RESOLUTION[0]}x{HD_RESOLUTION[1]}")
    print(f"🎬 분석 간격: 초당 1프레임")
    print("="*50)
    
    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 비디오 파일 찾기
    video_dir = Path(args.video_dir)
    if not video_dir.exists():
        print(f"❌ 비디오 폴더를 찾을 수 없습니다: {video_dir}")
        return
    
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
    video_files = []
    for ext in video_extensions:
        video_files.extend(video_dir.glob(f"*{ext}"))
    
    if not video_files:
        print(f"❌ {video_dir}에서 비디오 파일을 찾을 수 없습니다.")
        return
    
    print(f"📁 발견된 비디오: {len(video_files)}개")
    
    # 모델 초기화
    initialize_model(args.use_gpu)
    
    # 각 비디오 분석
    for i, video_path in enumerate(video_files, 1):
        print(f"\n🎬 [{i}/{len(video_files)}] {video_path.name}")
        analyze_video_hd(video_path, output_dir, args.temp_dir)
    
    print(f"\n🎉 모든 비디오 분석 완료!")
    print(f"📁 결과 폴더: {output_dir}")

if __name__ == "__main__":
    main() 