import requests
from pathlib import Path
import cv2
from tqdm import tqdm

# ─── 설정 ───────────────────────────────────────────────────────────
SERVER_IP = "3.38.253.26"
SERVER_PORT = 8000
API_ENDPOINT = f"http://{SERVER_IP}:{SERVER_PORT}/analyze_image/"
LOCAL_VIDEO_DIR = "/home/deepfine/video"
HD_RESOLUTION = (1280, 720)
# ────────────────────────────────────────────────────────────────────

def find_video_files(directory):
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
    video_dir = Path(directory)
    if not video_dir.is_dir():
        print(f"❌ 오류: 비디오 폴더를 찾을 수 없습니다 -> {directory}")
        return []
    
    video_files = []
    for ext in video_extensions:
        video_files.extend(video_dir.glob(f"*{ext}"))
    
    return video_files

def process_video_and_send_frames(video_path, session):
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"❌ 오류: '{video_path.name}' 비디오를 열 수 없습니다.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = int(fps)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_seconds = int(total_frames / fps)

    with tqdm(total=total_seconds, desc=f"🎬 {video_path.name}", unit="초") as pbar:
        for second in range(total_seconds):
            frame_pos = second * frame_interval
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()

            if not ret:
                tqdm.write(f"⚠️ '{video_path.name}'의 {second}초 프레임을 읽을 수 없습니다.")
                continue

            try:
                hd_frame = cv2.resize(frame, HD_RESOLUTION)
                is_success, buffer = cv2.imencode(".jpg", hd_frame)
                if not is_success:
                    tqdm.write(f"❌ {second}초 프레임 JPEG 인코딩 실패.")
                    continue

                files = {'file': ('frame.jpg', buffer.tobytes(), 'image/jpeg')}
                response = session.post(API_ENDPOINT, files=files, timeout=60)

                if response.status_code == 200:
                    result = response.json()
                    count_int = result.get('count_int', 'N/A')
                    pbar.set_postfix_str(f"👥 {count_int}명")
                else:
                    tqdm.write(f"❌ 서버 오류({response.status_code}): {response.text}")

            except requests.exceptions.RequestException as e:
                tqdm.write(f"❌ 서버 연결 오류: {e}")
                break
            except Exception as e:
                tqdm.write(f"❌ 프레임 처리 중 예외: {e}")
            
            pbar.update(1)

    cap.release()

def main():
    print("--- 로컬 비디오 분석 클라이언트 시작 ---")
    print(f"📡 서버 주소: {API_ENDPOINT}")
    print(f"📁 감시 폴더: {LOCAL_VIDEO_DIR}")
    print("------------------------------------")

    video_files = find_video_files(LOCAL_VIDEO_DIR)
    if not video_files:
        print(f"❌ 폴더 '{LOCAL_VIDEO_DIR}'에 비디오 파일이 없습니다.")
        return

    print(f"총 {len(video_files)}개의 비디오 파일을 순차적으로 처리합니다.")

    with requests.Session() as session:
        for video_path in video_files:
            process_video_and_send_frames(video_path, session)

    print("\n🎉 모든 비디오 처리가 완료되었습니다.")

if __name__ == "__main__":
    # pip install requests opencv-python tqdm 필요
    main()