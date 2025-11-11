import os
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Sequence, Optional
from insightface.app import FaceAnalysis
from torch.utils.data import Dataset
from torchvision import transforms

# --- (1) 동영상에서 가운데 이미지 추출해서 np.array로 반환 ---
def extract_middle_frame(video_path):
    if not os.path.exists(video_path):
        return None
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            return None
        
        middle_frame_index = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
        ret, frame = cap.read()

        if ret:
            return frame
        else:
            return None
    finally:
        cap.release()

# --- (2) 512 미만 이미지는 복제 패딩으로 보정 ---
def pad(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    pad_t = max(0, (512 - h) // 2)
    pad_b = max(0, 512 - h - pad_t)
    pad_l = max(0, (512 - w) // 2)
    pad_r = max(0, 512 - w - pad_l)
    if pad_t or pad_b or pad_l or pad_r:
        img = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r, borderType=cv2.BORDER_REPLICATE)
    return img


# --- (3) 얼굴 중심(코 좌표) ---
def find_center(app: FaceAnalysis, img: np.ndarray) -> Tuple[int, int]:
    faces = app.get(img)
    if not faces:
        # 얼굴이 없으면 이미지 중앙 사용
        h, w = img.shape[:2]
        return w // 2, h // 4

    # 가장 큰 얼굴 선택
    f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
    cx, cy = f.kps[2]
    return int(cx), int(cy)


# --- (4) 512×512 크롭, 밖으로 나가면 안쪽으로 이동 ---
def crop(img: np.ndarray, cx: int, cy: int) -> np.ndarray:
    H, W = img.shape[:2]
    half = 256
    x1 = min(max(cx - half, 0), max(W - 512, 0))
    y1 = min(max(cy - half, 0), max(H - 512, 0))
    x2, y2 = x1 + 512, y1 + 512
    return img[y1:y2, x1:x2]


# --- (5) 전체 파이프라인 ---
def process(img: np.ndarray, app: FaceAnalysis) -> Optional[np.ndarray]:
    img = pad(img)
    c = find_center(app, img)

    return crop(img, *c)


# --- (6) 크롭 이미지 저장 함수 (변경 없음) ---
def save_crops(
    in_dir="data",
    out_dir="data_cropped",
    providers: Sequence[str] = ("CUDAExecutionProvider", "CPUExecutionProvider"),
    det_size: Tuple[int, int] = (640, 640),
    model="buffalo_l"
):
    os.makedirs(out_dir, exist_ok=True)
    app = FaceAnalysis(name=model, providers=list(providers))
    app.prepare(ctx_id=0, det_size=det_size)

    for fname in os.listdir(in_dir):
        # --- [참고] 이 함수는 동영상을 처리하지 않습니다. ---
        # --- (FaceData 클래스와 별개) ---
        if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            continue
        ip = os.path.join(in_dir, fname)
        op = os.path.join(out_dir, fname)

        img = cv2.imread(ip)
        if img is None:
            continue
        
        out = process(img, app)
        if out is None:
            continue

        cv2.imwrite(op, out)
    print(f"✅ Cropped images saved in '{out_dir}'")


# --- (7) Dataset 클래스 (동영상 처리 기능 추가됨) ---
class FaceData(Dataset):

    def __init__(
        self,
        root="data",
        normalize=True,
        providers: Sequence[str] = ("CUDAExecutionProvider", "CPUExecutionProvider"),
        det_size: Tuple[int, int] = (640, 640),
        model="buffalo_l",
        fallback_center=True,
        transform=None
    ):
        # self.paths 대신 (경로, 레이블) 튜플을 저장할 self.samples 리스트 사용
        self.samples = []
        self.class_to_idx = {'real': 0, 'ai_images': 1} # 클래스와 인덱스 매핑

        IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        VIDEO_EXTS = {".avi", ".mp4"}
        supported_exts = IMAGE_EXTS | VIDEO_EXTS

        # --- [수정 1] 하위 폴더(real, ai_images)를 순회하며 레이블 할당 ---
        for target_class in self.class_to_idx.keys():
            class_index = self.class_to_idx[target_class]
            target_dir = os.path.join(root, target_class)

            if not os.path.isdir(target_dir):
                print(f"⚠️ Warning: Directory not found, skipping: {target_dir}")
                continue

            # os.walk로 target_dir의 모든 하위 폴더까지 탐색
            for r, _, fs in os.walk(target_dir):
                for f in fs:
                    ext = os.path.splitext(f)[1].lower()
                    if ext in supported_exts:
                        path = os.path.join(r, f)
                        item = (path, class_index) # (파일 경로, 클래스 인덱스) 저장
                        self.samples.append(item)
        # --- [수정 1 끝] ---

        if not self.samples:
            raise RuntimeError(f"No images or videos found in {root}/real or {root}/ai_images")
        
        print(f"✅ Found {len(self.samples)} files in {root}.")

        # 얼굴 탐지기 한 번만 초기화
        self.app = FaceAnalysis(name=model, providers=list(providers))
        self.app.prepare(ctx_id=0, det_size=det_size)
        
        # (이 부분은 원본 코드와 동일)
        if transform is None:
            self.transform = process
        else:
            self.transform = transform

        t = [transforms.ToTensor()]
        if normalize:
            t.append(transforms.Normalize([0.5]*3, [0.5]*3))
        self.tf = transforms.Compose(t)
        self.fallback_center = fallback_center

    def __len__(self):
        # --- [수정 2] self.paths 대신 self.samples 사용 ---
        return len(self.samples)

    def __getitem__(self, i):
        IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        VIDEO_EXTS = {".avi", ".mp4"}
        
        # --- [수정 3] 경로와 레이블을 self.samples에서 가져옴 ---
        path, label = self.samples[i]
        # --- [수정 3 끝] ---
        
        ext = os.path.splitext(path)[1].lower()
        img = None

        if ext in IMAGE_EXTS:
            img = cv2.imread(path)
        elif ext in VIDEO_EXTS:
            img = extract_middle_frame(path)

        if img is None:
            # 실패 시 다음 샘플을 로드하도록 None 대신 예외 발생 (또는 더미 데이터 반환)
            raise RuntimeError(f"Cannot read image or extract frame from: {path}")

        out = self.transform(img, self.app)

        if out is None:
            raise RuntimeError(f"Failed to process file (e.g., face not found): {path}")

        out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        x = self.tf(Image.fromarray(out_rgb))
        
        # --- [수정 4] 하드코딩된 0 대신 실제 레이블 반환 ---
        return x, label