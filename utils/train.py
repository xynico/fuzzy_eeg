import torch
import torch.nn as nn
import torch.optim as optim

def train_and_test_model_binary(model, learning_rate, train_loader, test_loader, epochs=10):
    # Check for GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Transfer the model to GPU if available
    model = model.to(device)

    # Define the loss function and optimizer
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training the model
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0

        for data, labels in train_loader:
            # Transfer data and labels to GPU if available
            data, labels = data.to(device), labels.to(device)

            # Convert labels to float32
            labels = labels.float()

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(data)
            outputs = outputs.squeeze()  # Remove any additional dimensions
            outputs = outputs.float()  # Ensure outputs are float32

            # Calculate loss
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader)}')

    # Testing the model
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            labels = labels.float()

            outputs = model(data)
            outputs = outputs.squeeze()  # Remove any additional dimensions

            # Convert outputs to predicted class (0 or 1)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test images: {accuracy}%')

    return accuracy
