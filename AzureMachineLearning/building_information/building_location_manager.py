# =====================================================
# building_location_manager.py
# E-gle Eye 시스템 - Building Location Excel Manager
# 원준님 요청: buildings2.xlsx (8대 카메라) 기본 사용
# [변경됨 ★★★★★] 2026-03-04 factory_data (인천 북항 스마트 물류센터 8대) 적용
# 생성일: 2026-03-04
# =====================================================

import pandas as pd
import os
from datetime import datetime

# [사용자 변경 필요] 기본 파일 경로 (buildings2.xlsx로 고정)
EXCEL_PATH = "buildings2.xlsx"

# 기본 컬럼 정의 (요구사항 완전 반영)
DEFAULT_COLUMNS = [
    "Camera_ID", "Building_Name", "Floor", "Zone",
    "GPS_Lat", "GPS_Lon",
    "Fire_Dept_Phone", "Fire_Dept_Email", "Fire_Dept_Address",
    "Device_IP",          
    "Threshold_Override", 
    "Last_Event_Time"     
]

class BuildingLocationManager:
    def __init__(self, excel_path=EXCEL_PATH):
        self.excel_path = excel_path
        # [적용 완료] Virtual Camera Manager 연동 준비 완료
        self.df = self._load_or_create_excel()
        print(f"✅ BuildingLocationManager 초기화 완료 ({len(self.df)}개 카메라 등록됨)")

    def _load_or_create_excel(self):
        if os.path.exists(self.excel_path):
            df = pd.read_excel(self.excel_path)
            print(f"📂 기존 파일 로드: {self.excel_path}")
        else:
            print(f"🆕 새 파일 생성: {self.excel_path}")
            df = pd.DataFrame(columns=DEFAULT_COLUMNS)
            
            # [변경됨 ★★★★★] 원준님 제공 factory_data 완전 적용
            # 이전: example_data (3개 카메라) → 삭제
            # 추가: 인천 북항 스마트 물류센터 8대 카메라 factory_data
            factory_data = {
                "Camera_ID": [
                    "camera_01", "camera_02", "camera_03", "camera_04",
                    "camera_05", "camera_06", "camera_07", "camera_08"
                ],
                "Building_Name": ["인천 북항 스마트 물류센터"] * 8,
                "Floor": [1, 1, 2, 2, 1, 3, 1, 1],
                "Zone": [
                    "입고 게이트 A", "입고 게이트 B", "생산라인 1층", "생산라인 2층",
                    "출고 적재장", "사무동 3층", "야외 보안 주차장", "중앙 보안센터"
                ],
                "GPS_Lat": [
                    37.4782, 37.4785, 37.4791, 37.4793,
                    37.4778, 37.4802, 37.4765, 37.4789
                ],
                "GPS_Lon": [
                    126.6321, 126.6324, 126.6318, 126.6320,
                    126.6335, 126.6312, 126.6341, 126.6327
                ],
                "Fire_Dept_Phone": ["032-123-4567"] * 8,
                "Fire_Dept_Email": ["northport@fire.go.kr"] * 8,
                "Fire_Dept_Address": ["인천광역시 중구 북항로 123"] * 8,
                "Device_IP": [
                    "192.168.10.101", "192.168.10.102", "192.168.10.103", "192.168.10.104",
                    "192.168.10.105", "192.168.10.106", "192.168.10.107", "192.168.10.108"
                ],
                "Threshold_Override": [72, 75, 68, 80, 65, 78, 70, 85],
                "Last_Event_Time": [None] * 8
            }
            df = pd.DataFrame(factory_data)
            self._save_excel(df)
        return df

    def _save_excel(self, df=None):
        if df is None:
            df = self.df
        df.to_excel(self.excel_path, index=False)
        print(f"💾 저장 완료 → {self.excel_path}")

    def get_by_camera_id(self, camera_id):
        row = self.df[self.df["Camera_ID"] == camera_id]
        if row.empty:
            return None
        return row.iloc[0].to_dict()

    def update_last_event(self, camera_id):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if camera_id in self.df["Camera_ID"].values:
            self.df.loc[self.df["Camera_ID"] == camera_id, "Last_Event_Time"] = now
            self._save_excel()
            print(f"🕒 {camera_id} Last_Event_Time 업데이트 → {now}")

    def get_fire_dept_info(self, camera_id):
        info = self.get_by_camera_id(camera_id)
        if not info:
            return None
        return {
            "phone": info["Fire_Dept_Phone"],
            "email": info["Fire_Dept_Email"],
            "address": info["Fire_Dept_Address"],
            "building": info["Building_Name"],
            "gps": (info["GPS_Lat"], info["GPS_Lon"])
        }

    def add_camera(self, **kwargs):
        new_row = pd.DataFrame([kwargs])
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        self._save_excel()
        print(f"➕ 새 카메라 등록: {kwargs.get('Camera_ID')}")

    # [적용 완료 ★★★★★] Virtual Camera Manager와 연동 Full Info
    def get_full_camera_info(self, camera_id):
        excel_info = self.get_by_camera_id(camera_id)
        if not excel_info:
            return None
        full_info = excel_info.copy()
        full_info.update({
            "status": "Green",
            "recent_events": [],
            "rtsp_url": None,
            "clip_paths": []
        })
        return full_info


# ====================== 테스트 코드 (필요 시 실행) ======================
if __name__ == "__main__":
    manager = BuildingLocationManager()
    full_info = manager.get_full_camera_info("camera_01")
    print("📍 camera_01 FULL 정보:", full_info)
    manager.update_last_event("camera_01")
    dept = manager.get_fire_dept_info("camera_01")
    print("🚨 소방서 자동 신고 정보:", dept)
