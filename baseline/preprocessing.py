import PIL
import cv2
import torch
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional
import os
import numpy as np
from typing import Tuple, Sequence, Optional


def detect_and_crop_face_optimized(
    image: Image.Image,
    target_size: Tuple[int, int] = (224, 224),
    detect_side: int = 512,
) -> Optional[Image.Image]:
    """
    OpenCV 얼굴 탐지 + 512→256→224 파이프라인
    얼굴 탐지 실패 시 상단부 중심으로 fallback crop
    """
    if image is None:
        return None
    if image.mode != "RGB":
        image = image.convert("RGB")

    original_np = np.array(image)
    H, W = original_np.shape[:2]

    # (1) 탐지용 리사이즈
    scale = detect_side / float(max(H, W))
    if scale < 1.0:
        det_w = int(W * scale)
        det_h = int(H * scale)
        det_img = cv2.resize(original_np, (det_w, det_h), interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
        det_img = original_np

    # (2) 얼굴 탐지
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    det_gray = cv2.cvtColor(det_img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(det_gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:

        cx, cy = W // 2, H // 4  # 살짝 위쪽 중심
    else:
        # 가장 큰 얼굴 선택
        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
        cx, cy = int((x + w / 2) / scale), int((y + h / 2) / scale)

    # (3) 원본에서 512 or 256 크롭
    crop_size = 512 if (H >= 512 and W >= 512) else 256
    half = crop_size // 2

    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x1 = min(x1, W - crop_size)
    y1 = min(y1, H - crop_size)

    cropped = original_np[y1:y1 + crop_size, x1:x1 + crop_size]
    if cropped.size == 0:
        return None

    # (4) 512/256 → 256
    resized_256 = cv2.resize(cropped, (256, 256), interpolation=cv2.INTER_AREA)

    # (5) 가운데 224
    start = (256 - 224) // 2
    end = start + 224
    final_np = resized_256[start:end, start:end]

    face_img = Image.fromarray(final_np).convert("RGB")
    return face_img

# 처리할 동영상 파일 확장자 목록
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTS = {".avi", ".mp4"}


def single_file_processing(
    file_path: Path,
) -> Tuple[str, Optional[List[Image.Image]], Optional[str]]:
    """
    # --- 2. 독스트링 설명 수정 ---
        하나의 파일(이미지 또는 비디오)을 받아 PIL Image 객체 리스트를 반환합니다.
        - 이미지: 원본 이미지를 로드합니다.
        - 비디오: 중간 프레임 1장만 추출하여 로드합니다.

        반환: (파일명, [PIL Image 리스트], 오류 메시지)
    """
    filename = file_path.name
    ext = file_path.suffix.lower()

    pil_image: Optional[Image.Image] = None

    try:
        if ext in IMAGE_EXTS:
            # --- 이미지 파일 처리 ---
            pil_image = Image.open(file_path).convert("RGB")

        elif ext in VIDEO_EXTS:
            # --- 동영상 파일 처리 (중간 프레임) ---
            cap = None
            try:
                cap = cv2.VideoCapture(str(file_path))
                if not cap.isOpened():
                    return filename, None, "Error: OpenCV_Failed to open video"

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames == 0:
                    return filename, None, "Error: Video has 0 frames"

                middle_frame_index = total_frames // 2
                cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
                ret, frame = cap.read()

                if not ret:
                    return filename, None, "Error: OpenCV_Failed to read middle frame"

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)

            finally:
                if cap:
                    cap.release()

        else:
            # --- 지원하지 않는 파일 ---
            return filename, None, f"Unsupported file type: {ext}"

        # --- 4. 공통 반환 (수정된 부분) ---
        if pil_image:
            # 'preprocessing' (data_transform) 함수를 호출하는 대신
            processed_image = detect_and_crop_face_optimized(pil_image)

            # 메인 루프가 리스트(List)를 기대하므로 리스트에 담아 반환
            # --- 3. PIL Image 객체를 직접 반환 ---
            return filename, [processed_image], None
        else:
            # (이론상 도달하지 않아야 함)
            return filename, None, "Error: Image object is None"

    except Exception as e:
        # 기타 모든 예외 처리 (파일 손상 등)
        return filename, None, f"Error: {str(e)}"
