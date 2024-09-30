import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader import OCRDataset  
from model import OCRModel  
from training import train_epochs  
import os
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    num_epochs = 20
    learning_rate = 0.001
    batch_size = 8

    root = os.getcwd()
    train_csv = os.path.join(root, 'CSV', 'written_name_train.csv')
    val_csv = os.path.join(root, 'CSV', 'written_name_validation.csv')
    train_img_dir = os.path.join(root, 'train_v2', 'train')
    val_img_dir = os.path.join(root, 'validation_v2', 'validation')

    train_dataset = OCRDataset(csv_file=train_csv, img_dir=train_img_dir)
    val_dataset = OCRDataset(csv_file=val_csv, img_dir=val_img_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=OCRDataset.ocr_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=OCRDataset.ocr_collate_fn)
    num_classes = len(train_dataset.char_to_idx)
    model = OCRModel(num_classes=num_classes).to(device)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_epochs(device, num_epochs, model, train_loader, val_loader, criterion, optimizer)

    torch.save(model.state_dict(), 'ocr_model.pth')
    print('Model saved as ocr_model.pth')

if __name__ == '__main__':
    main()

