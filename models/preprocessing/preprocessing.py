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
        fallback_center=True, # 원본 코드에 있던 인자
        transform = None
    ):
        self.paths = []
        
        # --- [수정 1] 이미지와 동영상을 모두 찾도록 변경 ---
        supported_exts = IMAGE_EXTS | VIDEO_EXTS
        for r, _, fs in os.walk(root):
            for f in fs:
                # 파일의 확장자를 소문자로 변경하여 확인
                ext = os.path.splitext(f)[1].lower()
                if ext in supported_exts:
                    self.paths.append(os.path.join(r, f))
        # --- [수정 1 끝] ---
                    
        self.paths.sort()
        if not self.paths:
            raise RuntimeError(f"No images or videos found in '{root}'")

        # 얼굴 탐지기 한 번만 초기화
        self.app = FaceAnalysis(name=model, providers=list(providers))
        self.app.prepare(ctx_id=0, det_size=det_size)
        
        # --- [수정 2] transform이 None이면 'process' 함수를 기본값으로 사용 ---
        if transform is None:
            self.transform = process
        else:
            self.transform = transform
        # --- [수정 2 끝] ---

        # 텐서 변환 설정
        t = [transforms.ToTensor()]
        if normalize:
            t.append(transforms.Normalize([0.5]*3, [0.5]*3))
        self.tf = transforms.Compose(t)
        self.fallback_center = fallback_center

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        VIDEO_EXTS = {".avi", ".mp4"}
        path = self.paths[i]
        
        # --- [수정 3] 파일 확장자에 따라 다르게 로드 ---
        ext = os.path.splitext(path)[1].lower()
        img = None

        if ext in IMAGE_EXTS:
            img = cv2.imread(path)
        elif ext in VIDEO_EXTS:
            img = extract_middle_frame(path)
        # --- [수정 3 끝] ---

        if img is None:
            raise RuntimeError(f"Cannot read image or extract frame from: {path}")

        # self.transform은 이제 'process' 함수 (또는 사용자가 지정한 함수)
        out = self.transform(img, self.app)

        if out is None:
            # process 함수가 (예: 얼굴 탐지 실패 시) None을 반환할 경우
            raise RuntimeError(f"Failed to process file (e.g., face not found): {path}")

        # BGR → RGB → PIL → Tensor
        out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        x = self.tf(Image.fromarray(out_rgb))
        return x, 0