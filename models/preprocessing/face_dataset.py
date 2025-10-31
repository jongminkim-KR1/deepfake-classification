import os
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Sequence, Optional
from insightface.app import FaceAnalysis
from torch.utils.data import Dataset
from torchvision import transforms


# --- (1) 512 미만 이미지는 복제 패딩으로 보정 ---
def pad(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    pad_t = max(0, (512 - h) // 2)
    pad_b = max(0, 512 - h - pad_t)
    pad_l = max(0, (512 - w) // 2)
    pad_r = max(0, 512 - w - pad_l)
    if pad_t or pad_b or pad_l or pad_r:
        img = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r, borderType=cv2.BORDER_REPLICATE)
    return img


# --- (2) 얼굴 중심(코 좌표) ---
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


# --- (3) 512×512 크롭, 밖으로 나가면 안쪽으로 이동 ---
def crop(img: np.ndarray, cx: int, cy: int) -> np.ndarray:
    H, W = img.shape[:2]
    half = 256
    x1 = min(max(cx - half, 0), max(W - 512, 0))
    y1 = min(max(cy - half, 0), max(H - 512, 0))
    x2, y2 = x1 + 512, y1 + 512
    return img[y1:y2, x1:x2]


# --- (4) 전체 파이프라인 ---
def process(img: np.ndarray, app: FaceAnalysis) -> Optional[np.ndarray]:
    img = pad(img)
    c = find_center(app, img)

    return crop(img, *c)


# --- (5) 크롭 이미지 저장 함수 ---
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
        if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            continue
        ip = os.path.join(in_dir, fname)
        op = os.path.join(out_dir, fname)

        img = cv2.imread(ip)
        if img is None:
            continue

        # --- 클래스와 동일한 방식으로 처리 ---
        out = process(img, app)
        if out is None:
            continue

        cv2.imwrite(op, out)
    print(f"✅ Cropped images saved in '{out_dir}'")


# --- (6) Dataset 클래스 (원본 → pad+탐지+크롭 → 텐서) ---
class FaceData(Dataset):

    def __init__(
        self,
        root="data",
        normalize=True,
        providers: Sequence[str] = ("CUDAExecutionProvider", "CPUExecutionProvider"),
        det_size: Tuple[int, int] = (640, 640),
        model="buffalo_l",
        fallback_center=True,
        transform = None
    ):
        self.paths = []
        for r, _, fs in os.walk(root):
            for f in fs:
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                    self.paths.append(os.path.join(r, f))
        self.paths.sort()
        if not self.paths:
            raise RuntimeError(f"No images found in '{root}'")

        # 얼굴 탐지기 한 번만 초기화
        self.app = FaceAnalysis(name=model, providers=list(providers))
        self.app.prepare(ctx_id=0, det_size=det_size)
        self.transform = transform

        # 텐서 변환 설정
        t = [transforms.ToTensor()]
        if normalize:
            t.append(transforms.Normalize([0.5]*3, [0.5]*3))
        self.tf = transforms.Compose(t)
        self.fallback_center = fallback_center

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"Cannot read image: {path}")

        out = self.transform(img, self.app)

        # BGR → RGB → PIL → Tensor
        out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        x = self.tf(Image.fromarray(out_rgb))
        return x, 0
