import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import os
import pandas as pd

def evaluate(model, data_loader, device, cce):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = cce(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    avg_loss = running_loss / len(data_loader)
    return avg_loss, acc

def test(model, test_path, transform, device):
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
    return test_data, predictions

def save_result(model, test_path, transform, device, epoch, output_dir=".", prefix="predictions"):
    # 필요 변수들
    test_data, predictions= test(model, test_path, transform, device)

    # 파일명과 예측 결과 매핑
    filenames = [os.path.basename(path[0]) for path in test_data.samples]
    df = pd.DataFrame({"filename": filenames, "predicted": predictions})
    # 클래스 명으로 저장
    idx_to_class = {v: k for k, v in test_data.class_to_idx.items()}
    df["label"] = df["predicted"].map(idx_to_class)

    output_filename = f"[Epoch_{epoch}]_{prefix}.csv"
    output_path = os.path.join(output_dir, output_filename)

    df.to_csv(output_path, index=False)
    print(f"예측 결과 저장 완료: {output_path}")


def plot_training(train_losses, val_losses, val_accuracies):
    epochs = range(1, len(train_losses)+1)

    plt.figure(figsize=(12, 5))

    # Loss 그래프
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()

    # Accuracy 그래프
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label='Val Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Validation Accuracy over Epochs")
    plt.legend()

    plt.tight_layout()
    plt.show()
