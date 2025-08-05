#!/usr/bin/env python3
# snapshot_batch_client.py
from __future__ import annotations
import os, time, json, requests
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv

# ──────────────────────────────────────────────────────────
# .env 로부터 설정 읽기
# ──────────────────────────────────────────────────────────
import re
def load_config() -> tuple[Dict[str, str], str, int, int]:
    load_dotenv()

    server_ip   = os.getenv("SERVER_IP", "127.0.0.1")
    server_port = int(os.getenv("SERVER_PORT", 8000))
    interval    = int(os.getenv("INTERVAL", 60))

    raw_env = os.getenv("CAMERAS", "")
    # ①  백슬래시 + 개행(\<newline>) 패턴 제거 → 한 줄로 합치기
    cameras_raw = re.sub(r"\\\s*\n", "", raw_env)

    snapshot_urls: Dict[str, str] = {}
    for cam in cameras_raw.split(";"):
        if "|" in cam:
            name, url = cam.split("|", 1)
            snapshot_urls[name.strip()] = url.strip()

    if not snapshot_urls:
        raise ValueError("⚠️ .env 의 CAMERAS 가 비어 있거나 형식 오류입니다.")
    return snapshot_urls, server_ip, server_port, interval

# ──────────────────────────────────────────────────────────
# 메인 클라이언트
# ──────────────────────────────────────────────────────────
class SnapshotBatchClient:
    def __init__(self,
                 snapshot_urls: Dict[str, str],
                 server_ip: str,
                 server_port: int,
                 interval_sec: int = 60,
                 output_dir: str = "analysis_results"):
        self.snapshot_urls = snapshot_urls
        self.api_endpoint  = f"http://{server_ip}:{server_port}/analyze_image/"
        self.interval      = interval_sec
        self.session       = requests.Session()
        self.output_dir    = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        print("──────── SnapshotBatchClient ────────")
        print(f"📡 분석 서버   : {self.api_endpoint}")
        print(f"📷 카메라 수   : {len(self.snapshot_urls)}")
        print(f"⏳ 호출 주기   : {self.interval} 초")
        print(f"💾 결과 폴더   : {self.output_dir}\n")

    # ── 스냅샷 한 장 받기
    def _get_snapshot(self, url: str) -> Optional[bytes]:
        try:
            r = self.session.get(url, timeout=10)
            if r.ok:
                return r.content
            print(f"❌ 스냅샷 실패 {r.status_code} → {url}")
        except requests.exceptions.RequestException as e:
            print(f"❌ 스냅샷 예외: {e}")
        return None

    # ── 서버 전송하여 count 받기
    def _send_to_server(self, cam: str, img: bytes) -> Optional[int]:
        try:
            r = self.session.post(
                self.api_endpoint,
                files={'file': (f"{cam}.jpg", img, "image/jpeg")},
                timeout=180
            )
            if r.ok:
                return r.json().get("count_int")
            print(f"❌ 서버 오류 {cam}: {r.status_code} {r.text}")
        except requests.exceptions.RequestException as e:
            print(f"❌ 서버 예외 {cam}: {e}")
        return None

    # ── 카메라별 JSON 파일에 append
    def _append_per_camera_json(self, batch: List[dict]) -> None:
        iso_ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        for rec in batch:                                 # {"camera": "..", "count": ..}
            cam, cnt = rec["camera"], rec["count"]
            fp = self.output_dir / f"{cam}.json"

            # 기존 파일 읽기
            if fp.exists():
                try:
                    data = json.loads(fp.read_text(encoding="utf-8"))
                    if not isinstance(data, list):
                        raise ValueError
                except Exception:
                    data = []
            else:
                data = []

            data.append({"timestamp": iso_ts, "count": cnt})
            fp.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                          encoding="utf-8")
            print(f"💾 {fp.name} ← {cnt}명 기록")

    # ── 메인 루프
    def run(self) -> None:
        try:
            while True:
                t0 = time.time()
                batch: List[dict] = []

                for cam, url in self.snapshot_urls.items():
                    img = self._get_snapshot(url)
                    if img:
                        cnt = self._send_to_server(cam, img)
                        if cnt is not None:
                            batch.append({"camera": cam, "count": cnt})
                            print(f"✅ {cam}: {cnt}명")

                if batch:
                    self._append_per_camera_json(batch)
                else:
                    print("⚠️ 결과 없음 (모든 스냅샷·전송 실패)")

                time.sleep(max(0, self.interval - (time.time() - t0)))
        except KeyboardInterrupt:
            print("\n🛑 사용자 중단")
        finally:
            self.session.close()

# ──────────────────────────────────────────────────────────
# 실행
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    urls, ip, port, interval = load_config()
    SnapshotBatchClient(urls, ip, port, interval).run()
