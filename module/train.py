import matplotlib.pyplot as plt
from module.utils import evaluate, plot_training

def train_model(model, cce, optimizer, train_loader, val_loader, device, epoch):
        train_losses = []
        val_losses = []
        val_accuracies = []
        # 학습 루프
        for epoch in range(epoch):
            model.train()
            running_loss = 0.0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = cce(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_train_loss = running_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # 검증
            val_loss, val_acc = evaluate(model, val_loader, device, cce)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # 훈련 종료 후 시각화
        plot_training(train_losses, val_losses, val_accuracies)