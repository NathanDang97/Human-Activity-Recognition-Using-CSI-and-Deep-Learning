import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_confusion_matrix(y_true, y_pred, classes):
    c_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(c_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()

def evaluate_model(model, test_loader, device, print_report=True, class_names=None):
    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(y_batch.numpy())
    if print_report:
        print(classification_report(y_true, y_pred, target_names=class_names if class_names is not None else None))
        if class_names is not None:
            plot_confusion_matrix(y_true, y_pred, class_names)

    correct_preds = np.sum(np.array(y_pred) == np.array(y_true))
    return correct_preds / len(y_true)

def train_model(model, train_loader, val_loader, device, learning_rate, epochs, checkpoint_path='best_model.pt'):
   # Train model using Adam optimizer and smoothed cross-entropy loss
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Apply learning rate scheduling based on validation accuracy
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    model.to(device)

    best_val_acc = 0
    patience_counter = 0
    early_stopping_patience = 10 # can change if we use different numbers of epochs

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_acc = evaluate_model(model, val_loader, device, print_report=False)
        scheduler.step(val_acc)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss:.4f}, Val Accuracy: {val_acc:.4f}")

        # Save best model checkpoint during training
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            patience_counter = 0
            print("Model checkpoint saved.")
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break