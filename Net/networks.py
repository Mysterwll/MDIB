import torch
from transformers import AutoModel

from Net.basicArchs import *
from Net.mamba_modules import *
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
    
class CLNet(nn.Module):
    def __init__(self, vision_encoder_selection=1, vision_dim = 2048, z_dim = 128, peculiar_dim=128, common_dim=128, n_classes =2, w = 0.5, use_attention = False):
        super(CLNet, self).__init__()
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
    
class CLNet_test(nn.Module):
    def __init__(self, vision_encoder_selection=1, vision_dim = 2048, z_dim = 128, peculiar_dim=128, common_dim=128, n_classes =2, w = 0.5, use_attention = False):
        super(CLNet_test, self).__init__()
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

        # self.mimin_glob = CLUBMean(self.dim*2, self.dim)
        # self.mimin = CLUBMean(self.dim, self.dim)

        self.CLUB = MIEstimator(z_dim)
        self.contrastiveLoss = ContrastiveLoss(temperature= 0.07)
        
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
            
            l_sh, _, _, _ = self.contrastiveLoss(common_feature_vision, common_feature_radio)
            disen_loss = self.CLUB(peculiar_feature_vision, peculiar_feature_radio, common_feature) \
                + self.CLUB.learning_loss(peculiar_feature_vision, peculiar_feature_radio, common_feature)\
                + l_sh

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

class CLNet_ablation(nn.Module):
    def __init__(self, vision_encoder_selection=1, vision_dim = 2048, z_dim = 128, peculiar_dim=128, common_dim=128, n_classes =2, w = 0.5, use_attention = False):
        super(CLNet_ablation, self).__init__()
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
            return logit
        else:
            IB_loss = self.alpha * self.loss_classify(logits_vision, label) \
                                + self.alpha * self.loss_classify(logits_radio, label) \
                                + self.beta * self.get_KL_loss(mu_vison, logvar_vision) \
                                + self.beta * self.get_KL_loss(mu_radio, logvar_radio)
            
            _, _, disen_loss = self.get_disen_loss(common_feature_vision, common_feature_radio, peculiar_feature_vision, peculiar_feature_radio)

            task_loss = self.loss_classify(logit, label)

            return logit, IB_loss, disen_loss, task_loss
        
class CLNet_1(nn.Module):
    def __init__(self, z_dim = 128, peculiar_dim=128, common_dim=128, n_classes =2, w = 0.5):
        super(CLNet, self).__init__()
        self.Resnet = get_pretrained_Vision_Encoder()
        self.IB_vision = IB(x_dim = 400, z_dim = 128, peculiar_dim=128, common_dim=128, n_classes =2)
        self.IB_radio = IB(x_dim = 1781, z_dim = 128, peculiar_dim=128, common_dim=128, n_classes =2)
        self.clinic_encoder = Radiomic_mamba_encoder(num_features=58)
        self.classify_head = nn.Linear(z_dim *3 , n_classes)
        self.w = w

        self.alpha = 0.1
        self.beta = 0.01

        self.club = CLUBMean(128*3, 128)
        self.loss_classify = nn.CrossEntropyLoss()

    def forward(self, cli, radio, img, label = None):
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
        vis_feature = torch.cat((peculiar_feature_vision, common_feature, peculiar_feature_radio), dim= 1)

        cli_feature = self.clinic_encoder(cli)

        global_feature = torch.cat((vis_feature, cli_feature), dim= 1)

        logit = self.classify_head(global_feature)

        if label is None:
            return logit
        else:
            IB_loss = self.alpha * self.loss_classify(logits_vision, label) \
                                + self.alpha * self.loss_classify(logits_radio, label) \
                                + self.beta * self.get_KL_loss(mu_vison, logvar_vision) \
                                + self.beta * self.get_KL_loss(mu_radio, logvar_radio)
            
            _, _, disen_loss = self.get_disen_loss(common_feature_vision,common_feature_radio, peculiar_feature_vision, peculiar_feature_radio)

            diff_loss = self.club(vis_feature, cli_feature)+ self.club.learning_loss(vis_feature, cli_feature)
            task_loss = self.loss_classify(logit, label)

            return logit, IB_loss, disen_loss, diff_loss, task_loss

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

        com_loss = torch.mean(torch.sqrt(F.mse_loss(common_vision, common_radio, reduction='mean')))
        spec_loss = torch.mean(torch.sqrt(F.mse_loss(spec_vision, spec_radio, reduction='mean')))
        com_spec_loss = com_loss / spec_loss

        return com_loss, spec_loss, com_spec_loss


class Multi_IBNet(nn.Module):
    def __init__(self, vision_encoder_selection=2, vision_dim = 2048, z_dim = 128, mid_dim = 20, n_classes =2, w = 0.5):
        super(Multi_IBNet, self).__init__()
        self.device = 'cuda'
        self.vision_encoder_selection = vision_encoder_selection
        self.mid_dim = 20
        if vision_encoder_selection == 1:
            self.Resnet = get_pretrained_Vision_Encoder()
        elif vision_encoder_selection == 2:
            self.Resnet = pretrained_resnet_2D()

        self.IB_vision = IB(x_dim = vision_dim, z_dim = 128, peculiar_dim=128, common_dim=128, n_classes =2)
        self.IB_radio = IB(x_dim = 1781, z_dim = 128, peculiar_dim=128, common_dim=128, n_classes =2)

        self.DNN = nn.Sequential(
            nn.Linear(z_dim *3, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, mid_dim *2),
        )

        self.classify_head = nn.Linear(mid_dim , n_classes)
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
        global_feature = torch.cat((peculiar_feature_vision, common_feature, peculiar_feature_radio), dim= 1)

        x = self.DNN(global_feature)
        mu, logvar = self.get_from_encode(x)
        out_feature = self.reparameterization(mu, logvar)

        logit = self.classify_head(out_feature)

        if label is None:
            return logit
        else:
            IB_loss = self.alpha * self.loss_classify(logits_vision, label) \
                                + self.alpha * self.loss_classify(logits_radio, label) \
                                + self.alpha * self.loss_classify(logit, label)\
                                + self.beta * self.get_KL_loss(mu_vison, logvar_vision) \
                                + self.beta * self.get_KL_loss(mu_radio, logvar_radio)\
                                + self.beta * self.get_KL_loss(mu, logvar)
            
            _, _, disen_loss = self.get_disen_loss(common_feature_vision, common_feature_radio, peculiar_feature_vision, peculiar_feature_radio)

            task_loss = self.loss_classify(logit, label)

            return logit, IB_loss, disen_loss, task_loss

    def get_from_encode(self, x):
        mu = x[:, :self.mid_dim]
        logvar = x[:, self.mid_dim:] # Make sigma always positive

        return mu, logvar
    
    def reparameterization(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn(std.shape).to(self.device)
            return mu + std * eps
        else:
            # Reconstruction mode
            return mu

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

        com_loss = torch.mean(torch.sqrt(F.mse_loss(common_vision, common_radio, reduction='mean')))
        spec_loss = torch.mean(torch.sqrt(F.mse_loss(spec_vision, spec_radio, reduction='mean')))
        com_spec_loss = com_loss / spec_loss

        return com_loss, spec_loss, com_spec_loss



class VAE(nn.Module):
    def __init__(self, z_dim = 128, peculiar_dim=128, common_dim=128, n_classes =2):
        super(VAE, self).__init__()

        self.z_dim =32
        self.device = 'cuda'
        self.encoder = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
        )

        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
        )

        self.encoder_common = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )

        self.encoder_peculiar = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )

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
        
        return mu, logvar, peculiar_feature, common_feature

class VAE_CLNet(nn.Module):
    def __init__(self, vision_encoder_selection=1, vision_dim = 2048, z_dim = 32, peculiar_dim=128, common_dim=128, n_classes =2, w = 0.5):
        super(VAE_CLNet, self).__init__()
        self.vision_encoder_selection = vision_encoder_selection
        if vision_encoder_selection == 1:
            self.Resnet = get_pretrained_Vision_Encoder()
        elif vision_encoder_selection == 2:
            self.Resnet = pretrained_resnet_2D()

        self.v_VAE = VAE()
        self.r_VAE = VAE() 

        self.v_encoder_0 = nn.Linear(2048, 256)

        self.r_encoder_0 = nn.Linear(1781, 256)

        self.decoder = nn.ModuleList()

        for i in range(4):
            self.decoder.append(nn.Sequential(
                nn.Linear(64, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(128, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(128, 256),
            ))

        self.classify_head = nn.Linear(z_dim *3 , n_classes)
        self.w = w

        self.loss_classify = nn.CrossEntropyLoss()

    def forward(self, radio, img):
        '''
        :param radio: torch.Size([B, 1781])
        :param img: torch.Size([B, 1, 64, 512, 512])
        :return: torch.Size([B, 2])
        '''
        vision_feature = self.Resnet(img)
        vision_feature = self.v_encoder_0(vision_feature)
        radio_feature = self.r_encoder_0(radio)

        mu_vison, logvar_vision, peculiar_feature_vision, common_feature_vision = self.v_VAE(vision_feature)
        mu_radio, logvar_radio, peculiar_feature_radio, common_feature_radio = self.r_VAE(radio_feature)

        common_feature = self.w * common_feature_vision + (1 - self.w)* common_feature_radio

        x_vv = self.decoder[0](torch.cat((common_feature_vision, peculiar_feature_vision), dim= 1))
        x_vr = self.decoder[1](torch.cat((common_feature_vision, peculiar_feature_radio), dim= 1))
        x_rr = self.decoder[2](torch.cat((common_feature_radio, peculiar_feature_radio), dim= 1))
        x_rv = self.decoder[3](torch.cat((common_feature_radio, peculiar_feature_vision), dim= 1))

        global_feature = torch.cat((peculiar_feature_vision, common_feature, peculiar_feature_radio), dim= 1)

        logit = self.classify_head(global_feature)

        outputs = {
            'x_v': vision_feature,
            'x_r': radio_feature,
            'x_vv': x_vv,
            'x_vr': x_vr,
            'x_rr': x_rr,
            'x_rv': x_rv,
            'global_feature': global_feature,
            'common_vision': common_feature_vision,
            'common_radio': common_feature_radio,
            'spec_vision': peculiar_feature_vision,
            'spec_radio': peculiar_feature_radio,
            'mu_v': mu_vison,
            'logvar_v':logvar_vision,
            'mu_r':mu_radio,
            'logvar_r':logvar_radio,
            'logit': logit
        }

        if self.training:
            return outputs, logit
        else:
            return logit







