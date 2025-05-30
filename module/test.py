from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch

def test_data(model, test_path, transform, device):
    test_data = datasets.ImageFolder(root=test_path, transform=transform)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    model.eval()
    predictions = []     # 분류 결과

    with torch.no_grad():
        for inputs, _ in test_loader:  # 테스트셋 라벨은 무시
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())