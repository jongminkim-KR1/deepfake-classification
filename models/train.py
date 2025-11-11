# models/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import sys

# --- [수정] 경로 설정 및 import ---
# 현재 파일(train.py)의 디렉토리 ( .../project_root/models )
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ( .../project_root )
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) 

# 프로젝트 루트를 sys.path에 추가하여 모듈 임포트가 가능하도록 함
# (이미 models가 PYTHONPATH에 잡혀있다면 필요 없을 수 있음)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# 데이터 경로는 프로젝트 루트 기준 'data' 폴더
DATA_ROOT = os.path.join(PROJECT_ROOT, "data") 
# 모델 저장 경로는 프로젝트 루트에 저장
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "cnn_fft_detector.pth")

try:
    # models/preprocessing/preprocessing.py에서 FaceData 임포트
    from models.preprocessing.preprocessing import FaceData 
    # models/base_models.py에서 CNNClfWithFFT 임포트
    from models.base_models import CNNClfWithFFT
except ImportError as e:
    print(f"Error: 모듈 임포트에 실패했습니다. 경로를 확인하세요. (Error: {e})")
    print(f"SCRIPT_DIR: {SCRIPT_DIR}")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"sys.path: {sys.path}")
    exit()
# --- [수정 끝] ---


# --- 하이퍼파라미터 설정 ---
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 20
TEST_SPLIT_RATIO = 0.2
NUM_WORKERS = 4 
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] 
det_size = (640, 640) 

# (collate_fn_safe 함수는 이전과 동일)
def collate_fn_safe(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return torch.empty(0), torch.empty(0)
    try:
        return torch.utils.data.dataloader.default_collate(batch)
    except RuntimeError as e:
        print(f"Warning: Skipping batch due to error: {e}")
        return torch.empty(0), torch.empty(0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading data from: {DATA_ROOT}")

    # --- 1. 데이터셋 로드 및 분리 ---
    print("Loading dataset...")
    try:
        full_dataset = FaceData(
            root=DATA_ROOT,
            normalize=True,
            providers=providers,
            det_size=det_size
        )
    except RuntimeError as e:
        print(f"Error loading dataset: {e}")
        print(f"'{DATA_ROOT}' 경로에 'real' 및 'ai_images' 폴더가 있는지 확인하세요.")
        return

    if len(full_dataset) == 0:
        print("Error: No data found. Check DATA_ROOT.")
        return
        
    indices = list(range(len(full_dataset)))
    labels = [s[1] for s in full_dataset.samples]

    train_indices, val_indices = train_test_split(
        indices,
        test_size=TEST_SPLIT_RATIO,
        stratify=labels,
        random_state=42
    )

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(
        full_dataset, batch_size=BATCH_SIZE, sampler=train_sampler,
        num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn_safe
    )
    val_loader = DataLoader(
        full_dataset, batch_size=BATCH_SIZE, sampler=val_sampler,
        num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn_safe
    )
    
    print(f"Total samples: {len(full_dataset)}")
    print(f"Training samples: {len(train_indices)}")
    print(f"Validation samples: {len(val_indices)}")

    # --- 2. 모델, 손실 함수, 옵티마이저 정의 ---
    model = CNNClfWithFFT(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # --- 3. 학습 루프 ---
    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        
        # --- 학습 (Train) ---
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        pbar_train = tqdm(train_loader, desc="Train", unit="batch", leave=False)
        for inputs, labels in pbar_train:
            if inputs.nelement() == 0: continue 

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels.data)
            total_preds += inputs.size(0)
            
            pbar_train.set_postfix(loss=f"{loss.item():.4f}")
        
        if total_preds > 0:
            epoch_loss = running_loss / total_preds
            epoch_acc = correct_preds.double() / total_preds
            print(f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")
        else:
            print("Warning: No training data processed in this epoch.")

        # --- 검증 (Validation) ---
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_total = 0

        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc="Val", unit="batch", leave=False)
            for inputs, labels in pbar_val:
                if inputs.nelement() == 0: continue

                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)
                val_total += inputs.size(0)

        if val_total > 0:
            val_epoch_loss = val_loss / val_total
            val_epoch_acc = val_corrects.double() / val_total
            print(f"Val Loss:   {val_epoch_loss:.4f} | Val Acc:   {val_epoch_acc:.4f}")

            if val_epoch_acc > best_val_acc:
                print(f"🚀 New best model found! Saving to {MODEL_SAVE_PATH}")
                best_val_acc = val_epoch_acc
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            print("Warning: No validation data processed in this epoch.")

    print("\n✅ Training finished.")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()