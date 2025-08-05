import requests
from pathlib import Path
import cv2
import time
import threading
import json




def sec_to_hhmmss(sec: int) -> str:
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"          # 01:12:37 형태
class SynchronizedSequentialClient:
    """
    여러 CCTV 소스에서 매분 동기화된 프레임을 캡처하여,
    각 이미지를 순차적으로 서버에 전송/분석하고,
    해당 분의 전체 결과를 로컬에 JSON 파일로 저장하는 클라이언트.
    """
    def __init__(self, server_ip, server_port, cctv_sources, interval_seconds=60, output_dir="results_json"):
        self.api_endpoint = f"http://{server_ip}:{server_port}/analyze_image/"
        self.cctv_sources = {name: Path(path) for name, path in cctv_sources.items()}
        self.hd_resolution = (1280, 720)
        self.interval = interval_seconds
        self.session = requests.Session()
        self.video_metadata = {}
        self.stop_event = threading.Event()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        print("--- 동기화 순차 분석 클라이언트 시작 ---")
        print(f"📡 서버 주소: {self.api_endpoint}")
        print(f"📹 감시 CCTV 수: {len(self.cctv_sources)}개")
        print(f"⏳ 수집 주기: {self.interval}초")
        print(f"💾 결과 저장 폴더: {self.output_dir}")
        print("------------------------------------")

    def _initialize_captures(self):
        """CCTV 소스(비디오 파일)를 열고 캡처 객체와 메타데이터를 준비합니다."""
        print("📹 CCTV 소스를 초기화합니다...")
        for name, path in self.cctv_sources.items():
            if not path.exists():
                print(f"❌ 경고: CCTV '{name}'의 소스 파일을 찾을 수 없습니다: {path}")
                continue
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                print(f"❌ 경고: CCTV '{name}'의 소스를 열 수 없습니다.")
                continue
            
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.video_metadata[name] = {'cap': cap, 'fps': fps}
            print(f"✅ CCTV '{name}' 준비 완료 (FPS: {fps:.2f})")
        return bool(self.video_metadata)

    def _capture_frames_at_minute(self, minute_index):
        """지정된 '분'에 해당하는 프레임을 모든 소스에서 캡처하여 리스트로 반환합니다."""
        captured_frames = []
        target_second = minute_index * self.interval
        print(f"⏱️  {minute_index}번째 주기 ({target_second}초 지점) 프레임 캡처 중...")

        for name, meta in self.video_metadata.items():
            cap, fps = meta['cap'], meta['fps']
            frame_position = int(target_second * fps)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
            ret, frame = cap.read()
            
            if ret:
                hd_frame = cv2.resize(frame, self.hd_resolution)
                is_success, buffer = cv2.imencode(".jpg", hd_frame)
                if is_success:
                    captured_frames.append({'name': name, 'buffer': buffer.tobytes()})
            else:
                print(f"  - 정보: CCTV '{name}'는 {target_second}초 지점의 프레임이 없습니다 (영상 종료).")
        return captured_frames

    def _create_new_json_structure(self, cctv_name):
        """CCTV에 대한 기본 JSON 구조를 생성하는 헬퍼 함수입니다."""
        return {
            "cctv_name": cctv_name,
            "video_path": str(self.cctv_sources.get(cctv_name, "N/A")),
            "analysis_results": []
        }
    

    def _save_results_to_json(self, results, minute_index):
        """분석 결과를 각 CCTV별 JSON 파일에 누적하여 저장합니다."""
        successful_results = [res for res in results if res.get('status') == 'success']
        if not successful_results:
            print("저장할 유효한 분석 결과가 없습니다.")
            return

        iso_timestamp = time.strftime('%Y-%m-%dT%H:%M:%S')
        target_second = minute_index * self.interval

        for res in successful_results:
            cctv_name = Path(res['filename']).stem
            output_filepath = self.output_dir / f"{cctv_name}.json"

            # 새 분석 결과 항목 생성
            # new_entry = {
            #     "timestamp": iso_timestamp,
            #     "target_second": target_second,
            #     "count": res.get('count_int')
            # }
            new_entry = {
                "timestamp": iso_timestamp,              # 저장 시각(ISO)
                "target_second": target_second,          # 영상 절대 초
                "timecode": sec_to_hhmmss(target_second),# ⇒ 사람이 읽기 쉬운 mm:ss
                "count": res.get('count_int')            # 분석 결과
            }

            # 기존 파일이 있으면 읽고, 없으면 새로 생성
            if output_filepath.exists():
                try:
                    with open(output_filepath, 'r', encoding='utf-8') as f:
                        output_data = json.load(f)
                    if "analysis_results" not in output_data or not isinstance(output_data.get("analysis_results"), list):
                        print(f"⚠️ '{output_filepath}' 파일 형식이 올바르지 않아 새로 생성합니다.")
                        output_data = self._create_new_json_structure(cctv_name)
                except (json.JSONDecodeError, IOError) as e:
                    print(f"⚠️ '{output_filepath}' 파일 읽기 오류. 파일을 새로 생성합니다: {e}")
                    output_data = self._create_new_json_structure(cctv_name)
            else:
                output_data = self._create_new_json_structure(cctv_name)

            # 새 결과를 추가하고 파일에 저장
            output_data["analysis_results"].append(new_entry)
            try:
                with open(output_filepath, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=4)
                print(f"💾 결과가 '{output_filepath}'에 추가되었습니다.")
            except Exception as e:
                print(f"❌ '{output_filepath}' JSON 파일 저장 중 오류 발생: {e}")

    def _process_frames_sequentially(self, frames_to_process, minute_index):
        """캡처된 프레임들을 하나씩 서버로 전송하고, 모든 결과를 모아 저장합니다."""
        if not frames_to_process:
            print("처리할 이미지가 없습니다.")
            return False

        all_results = []
        total_frames = len(frames_to_process)
        print(f"🚀 총 {total_frames}개의 이미지를 순차적으로 분석합니다...")

        for i, frame_data in enumerate(frames_to_process):
            name = frame_data['name']
            buffer = frame_data['buffer']
            
            print(f"  - [{i+1}/{total_frames}] '{name}' 전송 중...")
            
            try:
                files = {'file': (f"{name}.jpg", buffer, 'image/jpeg')}
                response = self.session.post(self.api_endpoint, files=files, timeout=180)
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"    ✅ 결과 수신: 👥 {result.get('count_int')}명")
                    all_results.append({
                        "filename": f"{name}.jpg",
                        "status": "success",
                        "count_int": result.get('count_int')
                    })
                else:
                    print(f"    ❌ 서버 오류 (코드: {response.status_code}): {response.text}")
                    all_results.append({"filename": f"{name}.jpg", "status": "error"})

            except requests.exceptions.RequestException as e:
                print(f"    ❌ 서버 연결 오류: {e}")
                all_results.append({"filename": f"{name}.jpg", "status": "error"})

        print("\n📊 해당 주기의 모든 프레임 분석 완료. 결과를 저장합니다.")
        self._save_results_to_json(all_results, minute_index)
        return True

    def run(self):
        """주기적으로 이미지를 수집하고 서버로 전송하는 메인 루프"""
        if not self._initialize_captures():
            print("초기화할 수 있는 CCTV가 없어 클라이언트를 종료합니다.")
            return

        minute_counter = 1
        try:
            while not self.stop_event.is_set():
                start_time = time.time()
                
                captured_frames = self._capture_frames_at_minute(minute_counter)
                
                if not self._process_frames_sequentially(captured_frames, minute_counter):
                    print("모든 비디오의 처리가 완료되어 클라이언트를 종료합니다.")
                    break
                
                minute_counter += 1

                # 1분마다 대기하는 로직을 주석 처리하고, 처리 완료 후 바로 다음 작업을 수행합니다.
                print("✅ 처리 완료. 즉시 다음 주기를 시작합니다...")
                # elapsed_time = time.time() - start_time
                # wait_time = max(0, self.interval - elapsed_time)
                # print(f"⏳ 다음 수집까지 {wait_time:.1f}초 대기합니다...")
                # self.stop_event.wait(wait_time)
        except KeyboardInterrupt:
            print("\n🛑 사용자에 의해 클라이언트가 중지되었습니다.")
        finally:
            self.shutdown()

    def shutdown(self):
        """모든 비디오 캡처를 해제합니다."""
        print("자원을 정리합니다...")
        for meta in self.video_metadata.values(): # type: ignore
            meta['cap'].release()
        self.stop_event.set()

if __name__ == "__main__":
    # ─── 설정 (이 부분을 수정하여 사용하세요) ────────────────────────
    SERVER_IP = "3.38.253.26"
    SERVER_PORT = 8000
    
    # 분석할 CCTV 소스 목록 (이름: 파일 경로)
    # 22개의 CCTV를 시뮬레이션하기 위한 예시
    video_folder = Path("/home/deepfine/video")
    available_videos = list(video_folder.glob("*.avi"))
    if not available_videos:
        raise FileNotFoundError(f"비디오 폴더에 영상이 없습니다: {video_folder}")
    
    # 폴더에서 찾은 비디오 파일의 수만큼만 CCTV 소스를 생성합니다.
    CCTV_SOURCES = {f"CCTV_{i+1:02d}": path for i, path in enumerate(available_videos)}
    # 이미지 수집 및 전송 주기 (초)
    INTERVAL = 1

    # 결과 JSON 파일이 저장될 폴더
    OUTPUT_JSON_DIR = "analysis_results"
    # ────────────────────────────────────────────────────────────
    
    client = SynchronizedSequentialClient(
        server_ip=SERVER_IP,
        server_port=SERVER_PORT,
        cctv_sources=CCTV_SOURCES,
        interval_seconds=INTERVAL,
        output_dir=OUTPUT_JSON_DIR
    )
    client.run()