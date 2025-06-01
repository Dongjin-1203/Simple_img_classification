# 라이브러리 불러오기
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.optim as optim

# 모듈 불러오기
from module.data_set import data_split
from module.model import Simple_CNN
from module.train import train_model
from module.utils import evaluate, test, save_result, plot_training, plot_confusion_matrix

if __name__ == "__main__":
    print("모듈을 성공적으로 불러왔습니다.")
    
    # 데이터 증강
    transform = transforms.Compose([
    #transforms.Resize((150, 150)),  #이미지를 픽셀단위로 resize(). RandomResizedCrop이 있으면 무의미
    transforms.RandomHorizontalFlip(),  # 50%확률로 좌우 반전
    transforms.RandomRotation(15),  # 이미지를 15도 사이 각도로 회전
    transforms.RandomResizedCrop(150, scale=(0.7, 1.0)),    # 이미지에서 무작위로 일정영역을 잘라냄
    transforms.ColorJitter(brightness=0.3, contrast=0.3),   # 이미지의 밝기, 대비를 무작위 조절. 조명, 날씨 조건 변화에 강인성 확보
    transforms.ToTensor(),  # 텐서로 변환
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
    epoch = 30
    train_model(model, cce, optimizer, train_loader, val_loader, device, epoch)
    
    # 모델 평가
    val_loss, val_acc = evaluate(model, val_loader, device, cce)
    print(f"Validation Loss: {val_loss:.4f} | Accuracy: {val_acc:.2f}%")

    # 결과 저장
    test_path = 'data/seg_test/seg_test'
    save_result(model, test_path, transform, device, epoch)

    # class_names는 ImageFolder로부터 추출된 클래스 리스트입니다.
    plot_confusion_matrix(model, val_loader, device, class_names, save_path="confusion_matrix.png")