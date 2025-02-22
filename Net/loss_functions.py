import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Masked_Language_Modeling_Loss(nn.Module):
    """
    Masked Language Modeling (MLM) Loss
    """

    def __init__(self):
        super(Masked_Language_Modeling_Loss, self).__init__()
        self.criterion = nn.NLLLoss(ignore_index=0)

    def forward(self, datas, labels):
        loss = 0.0
        for i in range(datas):
            next_sent_output, mask_lm_output = torch.eq(datas[i + 1], datas[i])
            next_loss = self.criterion(next_sent_output, datas[i + 1])
            mask_loss = self.criterion(mask_lm_output.transpose(1, 2), labels[i])
            loss += (next_loss + mask_loss)
        return loss


class Constract_Loss(nn.Module):
    def __init__(self, device="cpu"):
        super(Constract_Loss, self).__init__()

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.device = device

    def forward(self, image_features, text_features):
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size, device=self.device).long()
        total_loss = (
                             F.cross_entropy(logits_per_image, labels) +
                             F.cross_entropy(logits_per_text, labels)
                     ) / 2

        return total_loss

class Similarity_Distribution_Matching_Loss(nn.Module):
    """
    Similarity Distribution Matching (SDM) Loss,
    Adapted from: https://github.com/anosorae/IRRA
    """

    def __init__(self, length):
        super(Similarity_Distribution_Matching_Loss, self).__init__()
        self.length = length

    def forward(self, vision_fetures, text_fetures, labels, epsilon=1e-8):
        logit_scale = self.length
        labels = labels - labels.t()
        labels = (labels == 0).float()

        image_norm = vision_fetures / vision_fetures.norm(dim=1, keepdim=True)
        text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

        t2i_cosine_theta = text_norm @ image_norm.t()
        i2t_cosine_theta = t2i_cosine_theta.t()

        text_proj_image = logit_scale * t2i_cosine_theta
        vision_proj_text = logit_scale * i2t_cosine_theta

        # normalize the true matching distribution
        labels_distribute = labels / labels.sum(dim=1)

        i2t_pred = F.softmax(vision_proj_text, dim=1)
        i2t_loss = i2t_pred * (F.log_softmax(vision_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
        t2i_pred = F.softmax(text_proj_image, dim=1)
        t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

        loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

        return loss

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

def get_KL_loss(mu, std):
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

def get_disen_loss(common_vision, common_radio, spec_vision, spec_radio):

    com_loss = torch.mean(torch.sqrt(F.mse_loss(common_vision, common_radio, reduction='none')))
    spec_loss = torch.mean(torch.sqrt(F.mse_loss(spec_vision, spec_radio, reduction='none')))
    com_spec_loss = com_loss / spec_loss

    return com_loss, spec_loss, com_spec_loss



class IB_loss(nn.Module):
    def __init__(self, alpha, beta):
        super(IB_loss,self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.loss_classify = nn.CrossEntropyLoss()

    def forward(self, mu, logvar, logits, label):

        IB_loss = self.alpha * self.loss_classify(logits, label) \
                                + self.beta * get_KL_loss(mu, logvar)
        
        return IB_loss
    
# class Our_loss(nn.Module):
#     def __init__(self, alpha, beta):
#         super(IB_loss,self).__init__()
#         self.alpha = alpha
#         self.beta = beta
#         self.loss_classify = nn.CrossEntropyLoss()

#     def forward(self, mu, logvar, logits, label):

class joint_loss(nn.Module):
    def __init__(self, w1=0.2, w2=0.01, w3=1):
        super(joint_loss, self).__init__()
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

    def forward(self, IB_loss, disen_loss, task_loss):
        w1 = self.w1
        w2 = self.w2
        w3 = self.w3

        return w1* IB_loss+ w2* disen_loss+ w3* task_loss
    
class joint_loss_1(nn.Module):
    def __init__(self, w1=0.2, w2=0.01, w3=1):
        super(joint_loss_1, self).__init__()
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

    def forward(self, IB_loss, disen_loss, diff_loss, task_loss):
        w1 = self.w1
        w2 = self.w2
        w3 = self.w3

        return w1* IB_loss+ w2* disen_loss+ diff_loss+ w3* task_loss
    
class joint_loss_2(nn.Module):
    def __init__(self, w1=3, w2=0.25, w3=1, w4= 0.25):
        super(joint_loss_2, self).__init__()

        self.loss_classify = nn.CrossEntropyLoss()

    def forward(self, outputs, logit, label):
        w1 = self.w1
        w2 = self.w2
        w3 = self.w3
        w4 = self.w4

        task_loss = self.loss_classify(logit, label)

        _, _, disen_loss = get_disen_loss(outputs['common_vision'], outputs['common_radio'], outputs['spec_vision'], outputs['spec_radio'])

        kl_loss = -0.5 * torch.mean(1 + outputs['logvar_v'] - outputs['mu_v'].pow(2) - torch.exp(outputs['logvar_v']))\
                     + -0.5 * torch.mean(1 + outputs['logvar_r'] - outputs['mu_r'].pow(2) - torch.exp(outputs['logvar_r']))
        
        reconstruction_loss = torch.mean(torch.sqrt(F.mse_loss(outputs['x_v'], outputs['x_vv'], reduction='none')))\
                                + torch.mean(torch.sqrt(F.mse_loss(outputs['x_v'], outputs['x_vr'], reduction='none')))\
                                    + torch.mean(torch.sqrt(F.mse_loss(outputs['x_r'], outputs['x_rr'], reduction='none')))\
                                        + torch.mean(torch.sqrt(F.mse_loss(outputs['x_r'], outputs['x_rv'], reduction='none')))

        return w1* reconstruction_loss + w2* kl_loss + w3* disen_loss+ + w4* task_loss

if __name__ == '__main__':
    # smaller to test on local
    # tensor = torch.randn(size=(1, 768))
    # ltensor = torch.randn(size=(1, 1))
    # crien = Masked_Language_Modeling_Loss()
    # output = crien(tensor, ltensor)
    # print(output)
    # print(output.shape)

    img = torch.randn(size=(2, 512))
    radio = torch.randn(size=(2, 512))
    crien = Constract_Loss()
    output = crien(img, radio)
    print(output)
    print(output.shape)
