import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms

class OCRDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((50, 380)),  
                transforms.ToTensor()           
            ])
        else:
            self.transform = transform

        self.char_to_idx = self.create_char_to_idx()

    def create_char_to_idx(self):
        characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- .'
        char_to_idx = {char: idx for idx, char in enumerate(characters)}
        return char_to_idx

    def encode_label(self, label):
        try:
            label = str(label)
            return torch.tensor([self.char_to_idx[char] for char in label if char in self.char_to_idx], dtype=torch.long)
        except Exception as e:
            print("Error decoding label: {label}, Error: {e}")
            raise e
        
    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data_frame.iloc[idx, 0])
        image = Image.open(img_name).convert('L')
        if self.transform:
            image = self.transform(image)

        label = self.data_frame.iloc[idx, 1]
        if pd.isna(label) or not isinstance(label, str):
            label = ""

        label_encoded = self.encode_label(label)
        
        return image, label_encoded

    @staticmethod
    def ocr_collate_fn(batch):
        images, labels = zip(*batch)
        images = torch.stack(images, 0)  
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)
        label_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
        
        return images, labels_padded, label_lengths
