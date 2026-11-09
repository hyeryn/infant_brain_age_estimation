import torch
import torch.nn as nn
import torch.nn.functional as F


class Inception3D(nn.Module):
    def __init__(self, in_channels, f1, f3_in, f3_out, f5_in, f5_out, pool_proj, stride):
        super(Inception3D, self).__init__()

        # 1x1x1 Conv + BatchNorm + ReLU
        self.branch1 = nn.Sequential(
            nn.Conv3d(in_channels, f1, kernel_size=1, stride=stride),
            nn.BatchNorm3d(f1),
            nn.ReLU(inplace=True)
        )

        # 1x1x1 → 3x3x3 Conv + BatchNorm + ReLU
        self.branch2 = nn.Sequential(
            nn.Conv3d(in_channels, f3_in, kernel_size=1),
            nn.BatchNorm3d(f3_in),
            nn.ReLU(inplace=True),
            nn.Conv3d(f3_in, f3_out, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm3d(f3_out),
            nn.ReLU(inplace=True)
        )

        # 1x1x1 → 5x5x5 Conv + BatchNorm + ReLU
        self.branch3 = nn.Sequential(
            nn.Conv3d(in_channels, f5_in, kernel_size=1),
            nn.BatchNorm3d(f5_in),
            nn.ReLU(inplace=True),
            nn.Conv3d(f5_in, f5_out, kernel_size=5, padding=2, stride=stride),
            nn.BatchNorm3d(f5_out),
            nn.ReLU(inplace=True)
        )

        # Max Pooling → 1x1x1 Conv + BatchNorm + ReLU
        self.branch4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=1),
            nn.Conv3d(in_channels, pool_proj, kernel_size=1, padding=1, stride=stride),
            nn.BatchNorm3d(pool_proj),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        return torch.cat([branch1, branch2, branch3, branch4], dim=1)


class FireModule(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super(FireModule, self).__init__()

        self.squeeze = nn.Conv3d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)

        # Expand
        self.expand1x1 = nn.Conv3d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand3x3 = nn.Conv3d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)

        self.expand_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))

        x1 = self.expand_activation(self.expand1x1(x))
        x3 = self.expand_activation(self.expand3x3(x))

        return torch.cat([x1, x3], dim=1)  

class AgeNet(nn.Module):
    def __init__(self, in_chans):
        super(AgeNet, self).__init__()

        # Stem Network
        self.stem = nn.Sequential(
            # 1st Conv Layer
            nn.Conv3d(in_chans, 32, kernel_size=3, stride=(2,2,1), padding=1),  
            nn.ReLU(inplace=True),
            
            # 1st MaxPool Layer
            nn.MaxPool3d(kernel_size=3, stride=(2,2,1), padding=1), 

            # 2nd Conv Layer
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(inplace=True),

            # 3rd Conv Layer
            nn.Conv3d(64, 192, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(inplace=True),

            # 2nd MaxPool Layer
            nn.MaxPool3d(kernel_size=3, stride=(2,2,1), padding=1), 
        )


        # Inception + Fire Block
        self.inception3b = Inception3D(in_channels=192, f1=64, f3_in=64, f3_out=64, f5_in=64, f5_out=64, pool_proj=64, stride=1)
        self.inception3c = Inception3D(256, 120, 120, 120, 120, 120, 120, 1)   # 16 ×19 ×6 ×256 
        self.fire3 = FireModule(480, 16, 64, 64)    # 16 ×19 ×6 ×480 

        self.inception4b = Inception3D(128, 128, 128, 128, 128, 128, 128, (2,2,1))
        self.inception4c = Inception3D(512, 128, 128, 128, 128, 128, 128, 1)
        self.fire4 = FireModule(512, 16, 64, 64)

        self.inception4d = Inception3D(128, 128, 128, 128, 128, 128, 128, 1)
        self.inception4e = Inception3D(512, 132, 132, 132, 132, 132, 132, 1)
        self.fire5 = FireModule(528, 16, 128, 128)

        self.inception4f = Inception3D(256, 208, 208, 208, 208, 208, 208, (2,2,1))
        self.inception5b = Inception3D(832, 208, 208, 208, 208, 208, 208, 1)
        self.fire6 = FireModule(832, 32, 256, 256)

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)

        # Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  
        )

    def forward(self, x):
           
        x = self.stem(x) 
        x = self.inception3b(x)  
        x = self.inception3c(x)  
        x = self.fire3(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.fire4(x)

        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.fire5(x)

        x = self.inception4f(x)  
        x = self.inception5b(x)
        x = self.fire6(x)

        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        
        x = self.fc(x)

        return x


class FiLMBlock(nn.Module):
    def __init__(self, feature_dim, condition_dim):
        super(FiLMBlock, self).__init__()
        self.gamma_fc = nn.Linear(condition_dim, feature_dim)
        self.beta_fc = nn.Linear(condition_dim, feature_dim)

    def forward(self, x, cond):
        gamma = self.gamma_fc(cond).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta_fc(cond).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta         # gamma = scale, bata = shift  (like FiLM)


class AgeNet_film(nn.Module):
    def __init__(self, in_chans):
        super(AgeNet_film, self).__init__()

        # Stem Network
        self.stem = nn.Sequential(
            # 1st Conv Layer
            nn.Conv3d(in_chans, 32, kernel_size=3, stride=(2,2,1), padding=1),  
            nn.ReLU(inplace=True),
            
            # 1st MaxPool Layer
            nn.MaxPool3d(kernel_size=3, stride=(2,2,1), padding=1), 

            # 2nd Conv Layer
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(inplace=True),

            # 3rd Conv Layer
            nn.Conv3d(64, 192, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(inplace=True),

            # 2nd MaxPool Layer
            nn.MaxPool3d(kernel_size=3, stride=(2,2,1), padding=1), 
        )


        # Inception + Fire Block
        self.inception3b = Inception3D(in_channels=192, f1=64, f3_in=64, f3_out=64, f5_in=64, f5_out=64, pool_proj=64, stride=1)
        self.inception3c = Inception3D(256, 120, 120, 120, 120, 120, 120, 1)   # 16 ×19 ×6 ×256 
        self.fire3 = FireModule(480, 16, 64, 64)    # 16 ×19 ×6 ×480 

        self.inception4b = Inception3D(128, 128, 128, 128, 128, 128, 128, (2,2,1))
        self.inception4c = Inception3D(512, 128, 128, 128, 128, 128, 128, 1)
        self.fire4 = FireModule(512, 16, 64, 64)
        self.film4 = FiLMBlock(128, 2)  # for layer2

        self.inception4d = Inception3D(128, 128, 128, 128, 128, 128, 128, 1)
        self.inception4e = Inception3D(512, 132, 132, 132, 132, 132, 132, 1)
        self.fire5 = FireModule(528, 16, 128, 128)
        self.film5 = FiLMBlock(256, 2)  # for layer2

        self.inception4f = Inception3D(256, 208, 208, 208, 208, 208, 208, (2,2,1))
        self.inception5b = Inception3D(832, 208, 208, 208, 208, 208, 208, 1)
        self.fire6 = FireModule(832, 32, 256, 256)

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)

        # Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  
        )

    def forward(self, x, info):

        wm, icv = info
        wm = wm.reshape(-1, 1).float()
        icv = icv.reshape(-1, 1).float()
        cond = torch.concat((wm, icv), dim=1)  # [B, 2]
           
        x = self.stem(x) 
        x = self.inception3b(x)  
        x = self.inception3c(x)  
        x = self.fire3(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.fire4(x)
        x = self.film4(x, cond)

        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.fire5(x)
        x = self.film5(x, cond)

        x = self.inception4f(x)  
        x = self.inception5b(x)
        x = self.fire6(x)

        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        
        x = self.fc(x)

        return x
