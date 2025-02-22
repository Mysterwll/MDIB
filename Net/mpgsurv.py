import torch
from torch import nn
# from models.ResNet3D_BraTS import generate_model_BraTS_LF
from torch.nn import init
import torch.nn.functional as F
from Net.basicArchs import *

class ResNet_Linformer(nn.Module):
    def __init__(self, vision_encoder_selection = 2):
        super(ResNet_Linformer,self).__init__()

        if vision_encoder_selection == 1:
            self.Resnet = get_pretrained_Vision_Encoder()
        elif vision_encoder_selection == 2:
            self.Resnet = M3D_ResNet_50()
        elif vision_encoder_selection == 0:
            self.Resnet = nn.Identity()

        self.r_encoder = nn.Linear(1781, 32)
        # self.resnet_linformer = generate_model_BraTS_LF(model_depth=50)
        self.conv_deepsurv = DeepSurv_1dcnn()
        # self.linear_projection = nn.Linear(759,1)
        self.linear_projection = nn.Linear(1387,2)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, img, radio, cli):
        res_linformer_output = self.Resnet(img)
        
        radio = self.r_encoder(radio)
        tabular_data = torch.concat((radio,cli),dim = 1)
        conv_deepsurv_output = self.conv_deepsurv(tabular_data)

        #one_d_conv_output = self.oned_conv(tabular_data)
        output = torch.concat((res_linformer_output,conv_deepsurv_output),dim=1)
        output = F.tanh(self.linear_projection(output))

        return output
    
class DeepSurv_1dcnn(nn.Module):
    def __init__(self):
        super(DeepSurv_1dcnn, self).__init__()
        self.deepsurv = nn.Sequential(
            # nn.Linear(54,256),
            nn.Linear(90,256),
            nn.BatchNorm1d(256),
            nn.ELU(alpha=0.5,inplace=False),
            nn.Dropout(p=0.22),
            nn.Linear(256,16),
            nn.BatchNorm1d(16),
            nn.SiLU(inplace=False),
            nn.Dropout(p=0.22),
            nn.Linear(16,8),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(inplace=False),
            nn.Dropout(p=0.22),
            nn.Linear(8,1)
        )

        self.conv_batch = nn.Sequential(
        nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1,padding=1),
        nn.BatchNorm1d(16),
        nn.AvgPool1d(2),
        nn.SELU(),
        nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2,padding=1),
        nn.BatchNorm1d(32),
        nn.AvgPool1d(2),
        nn.SELU(),
        nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=3,padding=1),
        nn.SELU(),
        nn.AvgPool1d(2),
        nn.Flatten()
    )

        self.dilation_conv_batch = nn.Sequential(
        nn.Conv1d(in_channels=1, out_channels=16, dilation=1, kernel_size=3, stride=1,padding=0),
        nn.BatchNorm1d(16),
        nn.LeakyReLU(),
        nn.Conv1d(in_channels=16, out_channels=32, dilation=2, kernel_size=3, stride=2,padding=0),
        nn.BatchNorm1d(32),
        nn.LeakyReLU(),
        nn.Conv1d(in_channels=32, out_channels=64, dilation=3, kernel_size=3, stride=3,padding=0),
        nn.BatchNorm1d(64),
        nn.LeakyReLU(),
        nn.Flatten()
    )
        for m in self.modules():
            if isinstance(m,nn.Linear):
                init.kaiming_uniform_(m.weight,mode='fan_out',nonlinearity='relu')  #best c-index:0.8492702821126428 in epcho 781

    def forward(self,clinical_features):
        out4 = clinical_features
        out3 = self.deepsurv(clinical_features)
        all_features = clinical_features#torch.concat((clinical_features,radiomics_features),dim=1)
        #out3 = all_features
        #out3 = self.ds(all_features)
        features_size_2 = all_features.size(1)
        all_features = all_features.reshape(-1,1,features_size_2)

        out1 = self.conv_batch(all_features)
        out2 = self.dilation_conv_batch(all_features)
        output = torch.concat((out1,out2,out3,out4),dim=1)
        return output
    
if __name__ == '__main__':
    model = ResNet_Linformer()
    r = torch.tensor(torch.rand(2, 1781))
    v = torch.tensor(torch.rand(2, 1, 32, 32, 32))
    t = torch.tensor(torch.rand(2, 58))
    label = torch.tensor([1, 0])
    output = model(v, r,  t)
    print(output.shape)
    # print(loss)