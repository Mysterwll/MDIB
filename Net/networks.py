import torch
from transformers import AutoModel

from Net.basicArchs import *

from Net.fusions import *
import torch.nn as nn
from Net.club import *
from utils.Constractive import ContrastiveLoss

class Vis_only(nn.Module):
    def __init__(self, use_pretrained=True):
        super(Vis_only, self).__init__()
        self.name = 'Vis_only'
        if use_pretrained:
            self.Resnet = get_pretrained_Vision_Encoder()
        else:
            self.Resnet = M3D_ResNet_50()
        self.output = nn.Linear(400, 2)


    def forward(self, x):
        '''
        :param x: torch.Size([B, 1, 64, 512, 512])
        :return: torch.Size([B, 2])
        '''
        x = self.Resnet(x)

        return self.output(x)

class Single_modal(nn.Module):
    def __init__(self, i_dim = 512):
        super(Single_modal, self).__init__()
        self.name = 'Single_modal'
        self.encoder = nn.Sequential(
            nn.Linear(i_dim, i_dim*4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(i_dim*4, i_dim*4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(i_dim*4, i_dim*2),
        )
        self.output = nn.Linear(i_dim*2, 2)


    def forward(self, x):
        '''
        :param x: torch.Size([B, i_dim])
        :return: torch.Size([B, 2])
        '''
        x = self.encoder(x)

        return self.output(x)
    
class CTEncoder(nn.Module):
    def __init__(self, use_pretrained=True, embedding_dim: int = 512, projection_head: bool = True):
        super(CTEncoder, self).__init__()
        if use_pretrained:
            self.encoder = get_pretrained_Vision_Encoder()
        else:
            self.encoder = M3D_ResNet_50()

        if projection_head:
            self.projection_head = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embedding_dim, int(embedding_dim / 2)),
            )

        if embedding_dim != 2048:
            self.encoder.fc = nn.Sequential(
                nn.Linear(2048, 1024), nn.LeakyReLU(), nn.Linear(1024, embedding_dim)
            )
        else:
            self.encoder.fc = nn.Identity()

    def forward(self, x):
        '''
        :param x: torch.Size([B, 1, 64, 512, 512])
        :return: torch.Size([B, 400])
        '''
        x = self.encoder(x)
        if hasattr(self, "projection_head"):
            return self.projection_head(x), x
        else:
            return x
    
class Resnet_2D(nn.Module):
    def __init__(self, use_pretrained=True):
        super(Resnet_2D, self).__init__()
        self.name = 'Resnet_2D'
        if use_pretrained:
            self.Resnet = get_pretrained_resnet_2D()

        self.output = nn.Sequential(
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.Linear(256, 2, bias=False),
        )


    def forward(self, x):
        '''
        :param x: torch.Size([B, 1, 64, 512, 512])
        :return: torch.Size([B, 2])
        '''
        x = x[:,:,32,:,:].squeeze(2)
        x = x.repeat(1, 3,  1, 1)
        x = self.Resnet(x)
        x = x.view(x.size(0), -1)

        return self.output(x)

'''KL divergence between two normal distributions'''
def KL_between_normals(q_distr, p_distr):
    mu_q, sigma_q = q_distr
    mu_p, sigma_p = p_distr
    k = mu_q.size(1)

    mu_diff = mu_p - mu_q
    mu_diff_sq = torch.mul(mu_diff, mu_diff)
    logdet_sigma_q = torch.sum(2 * torch.log(torch.clamp(sigma_q, min=1e-8)), dim=1)
    logdet_sigma_p = torch.sum(2 * torch.log(torch.clamp(sigma_p, min=1e-8)), dim=1)

    fs = torch.sum(torch.div(sigma_q ** 2, sigma_p ** 2), dim=1) + torch.sum(torch.div(mu_diff_sq, sigma_p ** 2), dim=1)
    two_kl = fs - k + logdet_sigma_p - logdet_sigma_q
    return two_kl * 0.5

class IB(nn.Module):
    def __init__(self, x_dim, z_dim = 128, peculiar_dim=128, common_dim=128, n_classes =2):
        super(IB, self).__init__()

        self.z_dim =128
        self.device = 'cuda'
        self.encoder = nn.Sequential(
            nn.Linear(x_dim, z_dim*4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(z_dim*4, z_dim*4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(z_dim*4, z_dim*2),
        )

        self.encoder_common = nn.Sequential(
            nn.Linear(z_dim, common_dim),
            nn.ReLU(inplace=True),
        )

        self.encoder_peculiar = nn.Sequential(
            nn.Linear(z_dim, peculiar_dim),
            nn.ReLU(inplace=True),
        )

        self.decoder_logits = nn.Linear(z_dim, n_classes)
    
    # def get_from_encode(self, x):
    #     mu = x[:, :self.z_dim]
    #     sigma = torch.nn.functional.softplus(x[:, self.z_dim:]) # Make sigma always positive

    #     return mu, sigma

    def get_from_encode(self, x):
        mu = x[:, :self.z_dim]
        logvar = x[:, self.z_dim:] # Make sigma always positive

        return mu, logvar
    
    def reparameterization(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn(std.shape).to(self.device)
            return mu + std * eps
        else:
            # Reconstruction mode
            return mu

    def forward(self, x):
        x = self.encoder(x)
        mu, logvar = self.get_from_encode(x)
        z = self.reparameterization(mu, logvar)
        peculiar_feature = self.encoder_common(z)
        common_feature = self.encoder_peculiar(z) 
        logits = self.decoder_logits(z)
        return mu, logvar, peculiar_feature, common_feature, logits
    
class MDIB(nn.Module):
    def __init__(self, vision_encoder_selection=1, vision_dim = 2048, z_dim = 128, peculiar_dim=128, common_dim=128, n_classes =2, w = 0.5, use_attention = False):
        super(MDIB, self).__init__()
        self.vision_encoder_selection = vision_encoder_selection
        self.use_attension = use_attention
        if vision_encoder_selection == 1:
            self.Resnet = get_pretrained_Vision_Encoder()
        elif vision_encoder_selection == 2:
            self.Resnet = pretrained_resnet_2D()
        elif vision_encoder_selection == 3:
            self.Resnet = CTEncoder(embedding_dim = vision_dim, projection_head= False)
        elif vision_encoder_selection == 0:
            self.Resnet = nn.Identity()

        if self.use_attension:
            self.attn =  MultiheadAttention(num_attention_heads = 16, input_size = 128, hidden_dropout_prob= 0.2)

        self.IB_vision = IB(x_dim = vision_dim, z_dim = 128, peculiar_dim=128, common_dim=128, n_classes =2)
        self.IB_radio = IB(x_dim = 1781, z_dim = 128, peculiar_dim=128, common_dim=128, n_classes =2)

        self.classify_head = nn.Linear(z_dim *3 , n_classes)
        self.w = w

        # self.alpha = 0.2
        # self.beta = 0.2

        self.alpha = 0.1
        self.beta = 0.01

        self.loss_classify = nn.CrossEntropyLoss()

    def forward(self, radio, img, label = None):
        '''
        :param radio: torch.Size([B, 1781])
        :param img: torch.Size([B, 1, 64, 512, 512])
        :return: torch.Size([B, 2])
        '''
        vision_feature = self.Resnet(img)
        radio_feature = radio

        mu_vison, logvar_vision, peculiar_feature_vision, common_feature_vision, logits_vision = self.IB_vision(vision_feature)
        mu_radio, logvar_radio, peculiar_feature_radio, common_feature_radio, logits_radio = self.IB_radio(radio_feature)

        common_feature = self.w * common_feature_vision + (1 - self.w)* common_feature_radio

        if self.use_attension:
            fusion = torch.stack((peculiar_feature_vision, common_feature, peculiar_feature_radio), dim = 1)
            fusion = self.attn(fusion)
            global_feature = fusion.reshape(fusion.shape[0],-1)
        else:
            global_feature = torch.cat((peculiar_feature_vision, common_feature, peculiar_feature_radio), dim= 1)

        logit = self.classify_head(global_feature)

        if label is None:
            return logit, global_feature
        else:
            IB_loss = self.alpha * self.loss_classify(logits_vision, label) \
                                + self.alpha * self.loss_classify(logits_radio, label) \
                                + self.beta * self.get_KL_loss(mu_vison, logvar_vision) \
                                + self.beta * self.get_KL_loss(mu_radio, logvar_radio)
            
            _, _, disen_loss = self.get_disen_loss(common_feature_vision, common_feature_radio, peculiar_feature_vision, peculiar_feature_radio)

            task_loss = self.loss_classify(logit, label)

            return logit, IB_loss, disen_loss, task_loss

    def get_KL_loss(self, mu, std):
        '''
        :param mu: [batch_size, dimZ]
        :param std: [batch_size, dimZ]
        :return:
        '''
        # KL divergence between prior and posterior
        prior_z_distr = torch.zeros_like(mu), torch.ones_like(std)
        encoder_z_distr = mu, std

        I_zx_bound = torch.mean(KL_between_normals(encoder_z_distr, prior_z_distr))

        return torch.mean(I_zx_bound)


    def get_disen_loss(self, common_vision, common_radio, spec_vision, spec_radio):
        # 计算 com_loss
        com_loss = torch.mean(torch.sqrt(F.mse_loss(common_vision, common_radio, reduction='mean')))

        # 计算 spec_loss
        spec_loss = torch.mean(torch.sqrt(F.mse_loss(spec_vision, spec_radio, reduction='mean')))
      
        # 计算 com_spec_loss
        com_spec_loss = com_loss / spec_loss

        return com_loss, spec_loss, com_spec_loss
    
        







