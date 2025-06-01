from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import os
import pandas as pd
import numpy as np

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


def plot_training(train_losses, val_losses, val_accuracies, save_path="img/"):
    import matplotlib.pyplot as plt

    epochs = range(1, len(train_losses)+1)
    plt.figure(figsize=(12, 5))

    # 손실 그래프
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()

    # 정확도 그래프
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label='Val Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Validation Accuracy over Epochs")
    plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"📈 학습 그래프 저장 완료: {save_path}")
    plt.show()

# 조기 종료
class EarlyStopping:
    def __init__(self, patience=5, verbose=True, save_path="best_model.pt"):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.save_path = save_path

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss >= self.best_loss:
            self.counter += 1
            if self.verbose:
                print(f"💤 EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.save_path)
        if self.verbose:
            print(f"✅ Best model saved to {self.save_path}")

# 혼동 행렬
def plot_confusion_matrix(model, data_loader, device, class_names, normalize='true', title="Confusion Matrix", save_path="img/"):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds, normalize=normalize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues', values_format=".2f")
    plt.title(title)

    if save_path:
        plt.savefig(save_path)
        print(f"✅ Confusion Matrix 저장 완료: {save_path}")
    plt.show()
