import torch
import torch.nn as nn
import torchvision.transforms as transforms

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.feature(x)
        x = self.classifier(x)
        return x

class OCTTransFormer(nn.Module):
    def __init__(self, 
                 img_size=28, 
                 patch_size=7,
                 num_classes=4,  # OCTMNIST has 4 classes
                 dim=64,        # Giảm dim từ 64 xuống 32
                 depth=6,       # số lớp
                 heads=8,       # Giảm heads từ 16 xuống 4
                 mlp_dim=128):   # Giảm mlp_dim từ 128 xuống 64
        super().__init__()
        
        self.patch_size = patch_size
        self.img_size = img_size
        
        num_patches = (img_size // patch_size) ** 2  # 16 patches
        patch_dim = 1 * patch_size * patch_size      # 49 (grayscale)
        self.patch_embedding = nn.Linear(patch_dim, dim)
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=mlp_dim,
                batch_first=True  # Không dùng dropout
            ),
            num_layers=depth
        )
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        
    def forward(self, x):
        B = x.shape[0]
        
        # Unfold the image into patches
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(B, -1, self.patch_size * self.patch_size * 1)  # Single-channel
        
        # Patch embedding
        x = self.patch_embedding(x)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embedding
        x = x + self.pos_embedding
        
        # Transformer encoder
        x = self.transformer(x)
        
        # Extract CLS token for classification
        x = x[:, 0]
        x = self.mlp_head(x)
        
        return x


# Define a BasicBlock for ResNet18 without BatchNorm, using Tanh

# Định nghĩa block cơ bản không có BatchNorm
class BasicBlockNoBatchNorm(nn.Module):
    expansion = 1  # Không thay đổi expansion vì đây là ResNet18

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlockNoBatchNorm, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.tanh = nn.Tanh()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=True)
            )

    def forward(self, x):
        out = self.tanh(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = self.tanh(out)
        return out
        
# Định nghĩa ResNet18NoBatchNorm với số nút giảm một nửa
class ResNet18NoBatchNorm(nn.Module):
    def __init__(self, num_classes=9):  # Default for PathMNIST
        super(ResNet18NoBatchNorm, self).__init__()
        self.in_planes = 8  # Giảm từ 32 xuống 16

        # Initial conv layer
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=True)  # Giảm từ 32 xuống 16
        self.tanh = nn.Tanh()
        self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers với số kênh giảm một nửa
        self.layer1 = self._make_layer(BasicBlockNoBatchNorm, 8, 2, stride=1)   # Giảm từ 32 xuống 16
        self.layer2 = self._make_layer(BasicBlockNoBatchNorm, 16, 2, stride=2)   # Giảm từ 64 xuống 32
        self.layer3 = self._make_layer(BasicBlockNoBatchNorm, 32, 2, stride=2)   # Giảm từ 128 xuống 64
        self.layer4 = self._make_layer(BasicBlockNoBatchNorm, 64, 2, stride=2)  # Giảm từ 256 xuống 128

        # Global average pooling và fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * BasicBlockNoBatchNorm.expansion, num_classes)  # Giảm từ 256 xuống 128

        # Initialize weights with Xavier initialization for Tanh
        self._initialize_weights()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.tanh(x)
        x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class DermaMNISTNet(nn.Module):
    def __init__(self):
        super(DermaMNISTNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            #nn.Dropout(0.3),
            nn.Linear(7 * 7 * 64, 128),
            nn.Tanh(),
            #nn.Dropout(0.3),
            nn.Linear(128, 7),
        )
 

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class BloodMNISTNet(nn.Module):
    def __init__(self):
        super(BloodMNISTNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            #nn.Dropout(0.3),
            nn.Linear(7 * 7 * 64, 128),
            nn.Tanh(),
            #nn.Dropout(0.3),
            nn.Linear(128, 8),
        )
 

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Định nghĩa ResNet18NoBatchNorm với số nút giảm một nửa
class ResNet18Blood(nn.Module):
    def __init__(self, num_classes=8):  # Default for BloodMNIST
        super(ResNet18Blood, self).__init__()
        self.in_planes = 8  # Giảm từ 32 xuống 16

        # Initial conv layer
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=True)  # Giảm từ 32 xuống 16
        self.tanh = nn.Tanh()
        self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers với số kênh giảm một nửa
        self.layer1 = self._make_layer(BasicBlockNoBatchNorm, 8, 2, stride=1)   # Giảm từ 32 xuống 16
        self.layer2 = self._make_layer(BasicBlockNoBatchNorm, 16, 2, stride=2)   # Giảm từ 64 xuống 32
        self.layer3 = self._make_layer(BasicBlockNoBatchNorm, 32, 2, stride=2)   # Giảm từ 128 xuống 64
        self.layer4 = self._make_layer(BasicBlockNoBatchNorm, 64, 2, stride=2)  # Giảm từ 256 xuống 128

        # Global average pooling và fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * BasicBlockNoBatchNorm.expansion, num_classes)  # Giảm từ 256 xuống 128

        # Initialize weights with Xavier initialization for Tanh
        self._initialize_weights()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.tanh(x)
        x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
        
# Kiểm tra số tham số
model = LeNet5()
total_params = sum(p.numel() for p in model.parameters())
print(f"Total LeNet5 parameters: {total_params}")     

model = OCTTransFormer()
total_params = sum(p.numel() for p in model.parameters())
print(f"Total OCTTranformer parameters: {total_params}")  

model = ResNet18NoBatchNorm()
total_params = sum(p.numel() for p in model.parameters())
print(f"Total ResNet18NoBatchNorm parameters: {total_params}")   

model = DermaMNISTNet()
total_params = sum(p.numel() for p in model.parameters())
print(f"Total DermaMNISTNet parameters: {total_params}")   