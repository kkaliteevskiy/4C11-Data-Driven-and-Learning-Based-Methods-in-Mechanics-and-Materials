import torch
import torch.nn as nn
from math import floor

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, 'same'),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, stride, 'same'),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, stride, 'same'),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, input_padding = 0, output_padding = 0):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels, floor(in_channels/2), kernel_size, stride, output_padding=output_padding),
            nn.ReLU(),
            nn.Conv1d(floor(in_channels/2), out_channels, kernel_size, 1, 'same'),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, 1, 'same'),
            nn.ReLU()
        )

    def forward(self, x):
        return self.decoder(x)
 

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.enc1 = 0
        self.enc2 = 0
        self.bottle = 0
        self.dec1 = 0
        self.dec2 = 0
        self.out = 0

        self.encoder1 = Encoder(6, 12, 3, 1, 'same')
        self.encoder2 = Encoder(12, 24, 3, 1, 'same')

        self.bottleneck = nn.Sequential(nn.Conv1d(24, 24, 3, 1, 'same'), nn.ReLU())

        self.decoder1 = Decoder(48, 12, 2, 2, 'same', 1)
        self.decoder2 = Decoder(24, 6, 2, 2, 'same')


    def forward(self, x):
        self.enc1 = self.encoder1(x)
        self.enc2 = self.encoder2(self.enc1)
        self.bottle = self.bottleneck(self.enc2)
        self.dec1 = self.decoder1(torch.cat((self.bottle, self.enc2), axis=1))
        self.dec2 = self.decoder2(torch.cat((self.dec1, self.enc1), axis = 1))

        return self.dec2
    
class DenceBottleneck(nn.Module):
    def __init__(self, in_channels, seq_length, hidden_channels, out_channels):
        super(DenceBottleneck, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels*seq_length, hidden_channels), 
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels * seq_length),
            nn.ReLU(),
            nn.Unflatten(1, (out_channels, seq_length))
        )

    def forward(self, x):
        return self.bottleneck(x)
    
class ConvBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBottleneck, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding),
        )

    def forward(self, x):
        return self.bottleneck(x)



class UNet_DenseBottleneck(nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super(UNet_DenseBottleneck, self).__init__()
        self.input = 0
        self.enc1 = 0
        self.enc2 = 0
        self.bottle = 0
        self.dec1 = 0
        self.dec2 = 0
        self.out = 0

        self.encoder1 = Encoder(input_channels, 16, 3, 1, 'same')
        self.encoder2 = Encoder(16, 32, 3, 1, 'same')
        self.bottleneck = DenceBottleneck(32, 12, hidden_channels, 32)
        self.decoder1 = Decoder(64, 16, 2, 2, 'same', 1)
        self.decoder2 = Decoder(32, 8, 2, 2, 'same')
        self.output = nn.Sequential(nn.Conv1d(8 + input_channels, 8 + input_channels, 3, 1, 'same'),
                                    nn.ReLU(),
                                    nn.Conv1d(8 + input_channels, input_channels, 1, 1, 'same'))
        

    def forward(self, x):
        # batch_size = x.shape[0]
        self.input = x
        self.enc1 = self.encoder1(self.input)
        self.enc2 = self.encoder2(self.enc1)
        self.bottle = self.bottleneck(self.enc2)
        self.dec1 = self.decoder1(torch.cat((self.bottle, self.enc2), axis=1))
        self.dec2 = self.decoder2(torch.cat((self.dec1, self.enc1), axis = 1))
        self.out = self.output(torch.cat((self.dec2, self.input), axis=1))

        return self.out




class MLP(nn.Module):
    def __init__(self, input_channels,hidden_channels, output_channels):
        super(MLP, self).__init__()
        self.n_layers = len(hidden_channels)

        self.layers = nn.ModuleList()

        for i in range(self.n_layers):
            if i == 0:
                self.layers.append(nn.Flatten())
                self.layers.append(nn.Linear(input_channels, hidden_channels[i]))
            else:
                self.layers.append(nn.Linear(hidden_channels[i-1], hidden_channels[i]))
            self.layers.append(nn.ReLU())
        
        self.layers.append(nn.Linear(hidden_channels[-1], output_channels))
        self.layers.append(nn.Unflatten(1, (1, output_channels)))


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    

class UNet_ConvBottleneck(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = 0
        self.enc1 = 0
        self.enc2 = 0
        self.bottle = 0
        self.dec1 = 0
        self.dec2 = 0
        self.out = 0

        self.encoder1 = Encoder(6, 16, 3, 1, 'same')
        self.encoder2 = Encoder(16, 32, 3, 1, 'same')
        self.bottleneck = nn.Sequential(nn.Conv1d(32, 32, 3, 1, 'same'), nn.ReLU())
    