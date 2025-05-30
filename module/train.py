def train_model(model, cce, optimizer, train_loader, device, num_epochs = 10):
        # 학습 루프
        for epoch in range(num_epochs):
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

            avg_loss = running_loss / len(train_loader)
            print(f"[Epoch {epoch+1}] Avg Loss: {avg_loss:.4f}")