

def train_with_vim(model, train_loader, pca, criterion, optimizer, alpha=1.0, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()

            # Forward pass
            logits, features = model(images)

            # Compute virtual logits
            virtual_logits = []
            for i in range(features.shape[0]):
                feature_vector = features[i].cpu().detach().numpy().flatten()
                virtual_logit = compute_virtual_logit(feature_vector, pca, alpha)
                virtual_logits.append(virtual_logit)

            virtual_logits = torch.tensor(virtual_logits, device=logits.device).unsqueeze(1)

            # Augment logits with virtual logit
            augmented_logits = torch.cat([logits, virtual_logits], dim=1)

            # Compute loss and update
            loss = criterion(augmented_logits, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")