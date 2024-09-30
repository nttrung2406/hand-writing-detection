from prometheus_client import start_http_server, Gauge
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import OCRDataset  
from model import OCRModel 
import torch.nn as nn

start_http_server(8000)  
training_loss_gauge = Gauge('training_loss', 'Training Loss')
validation_loss_gauge = Gauge('validation_loss', 'Validation Loss')
epoch_gauge = Gauge('epoch', 'Current Epoch')

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, labels_padded, label_lengths in train_loader:  
        images = images.to(device)  
        labels_padded = labels_padded.to(device)
        label_lengths = label_lengths.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)  
        
        outputs = outputs.permute(1, 0, 2)  
        input_lengths = torch.full(size=(outputs.size(1),), fill_value=outputs.size(0), dtype=torch.long).to(device)
        loss = criterion(outputs, labels_padded, input_lengths, label_lengths)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    training_loss_gauge.set(avg_loss)
    
    return avg_loss


def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for images, labels_padded, label_lengths in val_loader: 
            images = images.to(device)
            labels_padded = labels_padded.to(device)
            label_lengths = label_lengths.to(device)

            outputs = model(images)
            outputs = outputs.permute(1, 0, 2) 
            input_lengths = torch.full(size=(outputs.size(1),), fill_value=outputs.size(0), dtype=torch.long).to(device)

            loss = criterion(outputs, labels_padded, input_lengths, label_lengths)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    validation_loss_gauge.set(avg_val_loss)
    
    return avg_val_loss


def train_epochs(device, num_epochs, model, train_loader, val_loader, criterion, optimizer):
    for epoch in range(num_epochs):
        epoch_gauge.set(epoch)
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate_model(model, val_loader, criterion, device)
        
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
