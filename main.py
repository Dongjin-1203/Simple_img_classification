# 라이브러리 불러오기
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

# 모듈 불러오기
from module.data_set import data_split

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
    data_split(data, class_names)