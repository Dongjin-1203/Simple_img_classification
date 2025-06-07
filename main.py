# 라이브러리 불러오기
from torchvision import datasets, transforms
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import ssl
import certifi
ssl._create_default_https_context = ssl._create_unverified_context


# 모듈 불러오기
from module.data_set import data_split
from module.model import Simple_CNN
from module.train import train_model
from module.utils import evaluate, save_result, plot_confusion_matrix
from module.resNet_18 import get_resnet18_model

if __name__ == "__main__":
    print("모듈을 성공적으로 불러왔습니다.")
    
    # 데이터 증강
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(150, scale=(0.7, 1.0)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    # ImageFolder로 불러오기
    path = 'data/seg_train/seg_train'
    data = datasets.ImageFolder(root=path, transform=train_transform)
    class_names = data.classes

    # 훈련/검증 데이터 분리
    train_loader, val_loader = data_split(data, class_names, visualize=False)
    val_loader.dataset.dataset.transform = val_transform

    # 테스트셋 경로
    test_path = 'data/seg_test/seg_test'

    # 훈련할 모델 리스트
    model_list = {
        "SimpleCNN": Simple_CNN,
        "ResNet18": lambda: get_resnet18_model(num_classes=len(class_names), pretrained=True)
    }

    for model_name, model_fn in model_list.items():
        print(f"\n📌 Training: {model_name}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model_fn().to(device)

        # 손실함수, 옵티마이저, 스케줄러
        cce = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        # best model 저장 경로 지정
        save_path = f"{model_name}_best_model.pt"
        epoch = 30

        # 학습
        train_model(model, cce, optimizer, train_loader, val_loader, device, epoch, scheduler, save_path=save_path)

        # best model 로드
        model.load_state_dict(torch.load(save_path))

        # 평가
        val_loss, val_acc = evaluate(model, val_loader, device, cce)
        print(f"✅ {model_name} → Validation Loss: {val_loss:.4f} | Accuracy: {val_acc:.2f}%")

        # 테스트 결과 저장
        save_result(model, test_path, val_transform, device, epoch, prefix=f"{model_name}_pred")

        # 혼동 행렬 저장
        plot_confusion_matrix(model, val_loader, device, class_names,
                              save_path=f"{model_name}_confusion_matrix.png")
