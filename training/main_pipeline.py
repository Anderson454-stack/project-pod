import os
import time
import cv2
from datetime import datetime

# ==================== Azure Custom Vision ====================
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials

# ==================== SmartTemporalLogic (이전 버전 그대로 사용) ====================
# ← 여기 이전에 드린 SmartTemporalLogic 클래스 전체를 붙여넣으세요 (window_size=7 버전)
class SmartTemporalLogic:
    """AI 결과가 들어올 때마다 최근 5개 프레임을 기억하고 
    과잉경고를 막아주는 핵심 클래스"""

    def __init__(self, window_size=5, vote_threshold=4, min_conf=0.75, alpha=0.7):
        # 초기 설정하는 부분
        # - window_size: 최근 몇 프레임까지 기억할지 (기본 5개 = 5초)
        # - vote_threshold: 몇 개 이상 같아야 알림을 줄지 (기본 4개)
        self.window = deque(maxlen=window_size) # 자동으로 오래된 프레임 삭제
        self.vote_threshold = vote_threshold
        self.min_conf = min_conf
        self.alpha = alpha # EWMA 가중치


# ==================== 1. 순차 이미지 로더 (당신 요구사항 100% 반영) ====================
class SequentialImageLoader:
    def __init__(self, image_folder: str, interval_sec: float = 1.0):
        self.image_folder = image_folder
        self.interval = interval_sec
        self.image_files = sorted([
            f for f in os.listdir(image_folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ])
        self.index = 0
        print(f"✅ {len(self.image_files)}개 이미지 준비 완료 (1초 간격 시뮬레이션)")

    def get_next(self):
        """다음 이미지 반환 (cv2 프레임 + Azure용 bytes)"""
        if self.index >= len(self.image_files):
            return None, None

        filename = self.image_files[self.index]
        path = os.path.join(self.image_folder, filename)

        # 원본 (화면 표시용)
        frame = cv2.imread(path)
        if frame is None:
            self.index += 1
            return None, None

        # ==================== Preprocessing (Azure 규격 + 야간 보정) ====================
        proc = cv2.resize(frame, (800, 600))                    # ← Azure 학습 크기에 맞게 변경하세요
        # 야간 화재 식별을 위한 Grayscale 보정 (필요 없으면 주석 처리)
        gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
        proc = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)           # 3채널 유지

        # Normalization (선택)
        # proc = cv2.normalize(proc, None, 0, 255, cv2.NORM_MINMAX)

        # Azure로 보낼 bytes
        success, encoded = cv2.imencode('.jpg', proc)
        image_bytes = encoded.tobytes() if success else None

        self.index += 1
        return frame, image_bytes   # frame: 표시용, bytes: AI 입력용


# ==================== 2. Azure Custom Vision Detector ====================
class AzureCustomVisionDetector:
    def __init__(self, endpoint: str, prediction_key: str, project_id: str, published_name: str):
        credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
        self.predictor = CustomVisionPredictionClient(endpoint, credentials)
        self.project_id = project_id
        self.published_name = published_name
        print("✅ Azure Custom Vision Object Detection 준비 완료")

    def detect(self, image_bytes):
        if not image_bytes:
            return {'class': 'Normal', 'confidence': 0.0}

        results = self.predictor.detect_image(
            self.project_id,
            self.published_name,
            image_bytes
        )

        # Fire 또는 Smoke 중 가장 높은 confidence 선택
        best_class = 'Normal'
        best_conf = 0.0
        for pred in results.predictions:
            if pred.tag_name in ['Fire', 'Smoke'] and pred.probability > best_conf:
                best_conf = pred.probability
                best_class = pred.tag_name

        return {'class': best_class, 'confidence': float(best_conf)}


# ==================== 3. 메인 실행 ====================
def main():
    # ==================== Azure 정보 (여기 수정!) ====================
    ENDPOINT = "https://your-customvision.cognitiveservices.azure.com/"   # ← 수정
    PREDICTION_KEY = "your_prediction_key"                               # ← 수정
    PROJECT_ID = "your-project-guid-here"                                # ← 수정
    PUBLISHED_NAME = "Iteration1"   # 또는 Publish한 이름 (예: "detectModel")   # ← 수정

    IMAGE_FOLDER = r"C:\path\to\your\fire_smoke_images"   # ← 당신 이미지 폴더 경로

    # 클래스 초기화
    loader = SequentialImageLoader(IMAGE_FOLDER, interval_sec=1.0)
    detector = AzureCustomVisionDetector(ENDPOINT, PREDICTION_KEY, PROJECT_ID, PUBLISHED_NAME)
    logic = SmartTemporalLogic(window_size=7)   # ← 당신이 이미 가지고 있는 클래스

    print("\n🚀 화재/연기 시뮬레이션 시스템 시작! (q 키로 종료)\n")

    try:
        while True:
            cv_frame, image_bytes = loader.get_next()
            if cv_frame is None:
                print("모든 이미지 처리 완료!")
                break

            # AI 검출
            ai_result = detector.detect(image_bytes)

            # 신뢰도 검증 레이어
            logic.add_result(ai_result)
            decision = logic.get_decision()

            # 결과 출력
            print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                  f"상태: {decision['status']} | 알림: {decision['alert']} | "
                  f"이상률: {decision.get('anomaly_ratio', 0)}% | "
                  f"AI: {ai_result['class']}({ai_result['confidence']:.2f})")

            if decision['alert']:
                print("🚨 RED ALERT! 즉시 알림 발송!!!")

            # 화면 미리보기
            cv2.imshow("Fire/Smoke Simulation", cv_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(1.0)   # 1초 간격 시뮬레이션

    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()