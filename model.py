import torch
import torch.nn as nn

class OCRModel(nn.Module):
    def __init__(self, num_classes):
        super(OCRModel, self).__init__()        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                         
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                       
            nn.BatchNorm2d(128)
        )
        
        self.lstm_input_size = 128 * 12  
        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=128, num_layers=2, bidirectional=True, batch_first=True)        
        self.fc = nn.Linear(128 * 2, num_classes)  

    def forward(self, x):
        conv_out = self.cnn(x)  
        b, c, h, w = conv_out.size()         
        lstm_in = conv_out.permute(0, 3, 1, 2).contiguous().view(b, w, c * h) 
        lstm_out, _ = self.lstm(lstm_in)          
        output = self.fc(lstm_out)          
        return output
