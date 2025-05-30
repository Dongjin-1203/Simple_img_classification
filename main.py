# 라이브러리 불러오기
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.optim as optim

# 모듈 불러오기
from module.data_set import data_split
from module.model import Simple_CNN
from module.train import train_model
from module.utils import evaluate, test, save_result, plot_training

if __name__ == "__main__":
    print("모듈을 성공적으로 불러왔습니다.")
    
    # 이미지 전처리 정의
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
    ])

    # ImageFolder로 불러오기
    path='data/seg_train/seg_train'
    data = datasets.ImageFolder(root=path, transform=transform)
    """
    # 라벨 확인
    print(data.classes)   # ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
    print(data.class_to_idx)  # {'buildings': 0, 'forest': 1, ...}

    # DataLoader로 감싸기
    loader = DataLoader(data, batch_size=32, shuffle=True)

    # 데이터 확인
    images, labels = next(iter(loader))
    print(images.shape)  # torch.Size([32, 3, 150, 150])
    print(labels[:5])    # tensor([2, 2, 3, 3, 0])
    """ 
    # 클래스 이름
    class_names = data.classes

    # 훈련/검증 데이터 분류 및 확인
    train_loader, val_loader = data_split(data, class_names, visualize=False)

    # 모델 선언
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Simple_CNN().to(device)

    # 손실함수, 옵티마이져 선언
    cce = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 훈련 모델
    epoch = 8
    train_model(model, cce, optimizer, train_loader, val_loader, device, epoch)
    
    # 모델 평가
    val_loss, val_acc = evaluate(model, val_loader, device, cce)
    print(f"Validation Loss: {val_loss:.4f} | Accuracy: {val_acc:.2f}%")

    # 결과 저장
    test_path = 'data/seg_test/seg_test'
    save_result(model, test_path, transform, device, epoch)