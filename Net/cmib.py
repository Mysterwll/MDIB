import torch
import torch.utils.checkpoint
from torch import nn
#from torch.nn import CrossEntropyLoss, MSELoss

# from modules.transformer import TransformerEncoder
from torch.nn import CrossEntropyLoss, L1Loss, MSELoss

import torch.optim as optim
from itertools import chain

beta   = 1e-3
from torch.optim import AdamW, Adam
from torch.nn import Sequential
from torch.nn import functional as F

from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal
from torch.autograd import Variable
import numpy as np

class MIB(nn.Module):
    def __init__(self, v_dim = 512, r_dim = 1781, c_dim = 58):
        super().__init__()
        self.num_labels = 2

        self.d_l = 50

        self.dropout = nn.Dropout(0.5)
      
        self.attn_dropout = 0.5
        # self.proj_a = nn.Conv1d(1, self.d_l, kernel_size=3, stride=1, padding=1, bias=False)
        # self.proj_v = nn.Conv1d(1, self.d_l, kernel_size=3, stride=1, padding=1, bias=False)
        # self.proj_t = nn.Conv1d(1, self.d_l, kernel_size=3, stride=1, padding=1, bias=False)

        self.proj_a = nn.Sequential(
            nn.Linear(r_dim, self.d_l),
            nn.ReLU(inplace=True))
        
        self.proj_v = nn.Sequential(
            nn.Linear(v_dim, self.d_l),
            nn.ReLU(inplace=True))
        
        self.proj_t = nn.Sequential(
            nn.Linear(c_dim, self.d_l),
            nn.ReLU(inplace=True))
       
        self.fusion = fusion(self.d_l)
      
        # self.mean = nn.AdaptiveAvgPool1d(1)

        # self.init_weights()

    def forward(
        self,
        visual,
        acoustic,
        text,
        label
    ):
 
        acoustic = self.proj_a(acoustic)
        visual = self.proj_v(visual)
        text = self.proj_t(text)

        # acoustic = acoustic.permute(2, 0, 1)
        # visual = visual.permute(2, 0, 1)
        # text = text.permute(2, 0, 1)

        # outputa = self.transa(acoustic)
        # outputv = self.transv(visual)
        # outputt = self.transt(text)
        # output_a = outputa[-1]  # 48 50
        # output_v = outputv[-1]
        # output_t = outputt[-1]

        output_t = text
        output_a = acoustic
        output_v = visual

        outputf, loss_u = self.fusion(output_t, output_a, output_v, label)

        loss_fct = CrossEntropyLoss()

        loss_m = loss_fct(outputf, label)

        loss = loss_u + loss_m

        return outputf, loss



    def test(self,

        visual,
        acoustic,
        text
    ):

        acoustic = self.proj_a(acoustic)
        visual = self.proj_v(visual)
        text = self.proj_t(text)

        # acoustic = acoustic.permute(2, 0, 1)
        # visual = visual.permute(2, 0, 1)
        # text = text.permute(2, 0, 1)

        # outputa = self.transa(acoustic)
        # outputv = self.transv(visual)
        # outputt = self.transt(text)
        # output_a = outputa[-1]  # 48 50
        # output_v = outputv[-1]
        # output_t = outputt[-1]

        output_t = text
        output_a = acoustic
        output_v = visual


        outputf = self.fusion.test(output_t, output_a, output_v)


        return outputf







class fusion(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.d_l = dim
        
        # build encoder
        self.encoder_l = nn.Sequential(nn.Linear(self.d_l, 1024),
                                   #  nn.ReLU(inplace=True),
                                    # nn.Linear(1024, 1024),
                                     nn.ReLU(inplace=True) )  


        self.encoder_a = nn.Sequential(nn.Linear(self.d_l, 1024),
                                   #  nn.ReLU(inplace=True),
                                    # nn.Linear(1024, 1024),
                                     nn.ReLU(inplace=True) )  

        self.encoder_v = nn.Sequential(nn.Linear(self.d_l, 1024),
                                   #  nn.ReLU(inplace=True),
                                    # nn.Linear(1024, 1024),
                                     nn.ReLU(inplace=True) )  
        self.encoder = nn.Sequential(nn.Linear(self.d_l, 1024),
                                   #  nn.ReLU(inplace=True),
                                    # nn.Linear(1024, 1024),
                                     nn.ReLU(inplace=True) )  

        self.fc_mu_l  = nn.Linear(1024, self.d_l) 
        self.fc_std_l = nn.Linear(1024, self.d_l)

        self.fc_mu_a  = nn.Linear(1024, self.d_l) 
        self.fc_std_a = nn.Linear(1024, self.d_l)

        self.fc_mu_v  = nn.Linear(1024, self.d_l) 
        self.fc_std_v = nn.Linear(1024, self.d_l)

        self.fc_mu  = nn.Linear(1024, self.d_l) 
        self.fc_std = nn.Linear(1024, self.d_l)
        
        # build decoder
        self.decoder_l = nn.Linear(self.d_l, 2)
        self.decoder_a = nn.Linear(self.d_l, 2)
        self.decoder_v = nn.Linear(self.d_l, 2)
        self.decoder = nn.Linear(self.d_l, 2)

      #  self.fusion1 = graph_fusion(self.d_l, self.d_l)
        self.fusion1 = concat(self.d_l, self.d_l)
      #  self.fusion1 = tensor(self.d_l, self.d_l)
      #  self.fusion1 = addition(self.d_l, self.d_l)
      #  self.fusion1 = multiplication(self.d_l, self.d_l)
     #   self.fusion1 = low_rank(self.d_l, self.d_l)

    def encode(self, x):
        """
        x : [batch_size,784]
        """
        x = self.encoder(x)
        return self.fc_mu(x), F.softplus(self.fc_std(x)-5, beta=1)


    def encode_l(self, x):
        """
        x : [batch_size,784]
        """
        x = self.encoder_l(x)
        return self.fc_mu_l(x), F.softplus(self.fc_std_l(x)-5, beta=1)

    def encode_a(self, x):
        """
        x : [batch_size,784]
        """
        x = self.encoder_a(x)
        return self.fc_mu_a(x), F.softplus(self.fc_std_a(x)-5, beta=1)

    def encode_v(self, x):
        """
        x : [batch_size,784]
        """
        x = self.encoder_v(x)
        return self.fc_mu_v(x), F.softplus(self.fc_std_v(x)-5, beta=1)
    
    def decode_l(self, z):

        return self.decoder_l(z)

    def decode_a(self, z):

        return self.decoder_a(z)

    def decode(self, z):

        return self.decoder(z)

    def decode_v(self, z):

        return self.decoder_v(z)
    
    def reparameterise(self, mu, std):
        """
        mu : [batch_size,z_dim]
        std : [batch_size,z_dim]        
        """        
        # get epsilon from standard normal
        eps = torch.randn_like(std)
        return mu + std*eps

    def loss_function(self, y_pred, y, mu, std):
   
        loss_fct = CrossEntropyLoss()

        # CE = loss_fct(y_pred.view(-1,), y.view(-1,))
        CE = loss_fct(y_pred, y)
        KL = 0.5 * torch.mean(mu.pow(2) + std.pow(2) - 2*std.log() - 1)
        return (beta*KL + CE) 




    def forward(
        self,
        x_l,
        x_a,
        x_v,
        label_ids
    ):


        mu_l, std_l = self.encode_l(x_l)
        z_l = self.reparameterise(mu_l, std_l)
        output_l =  self.decode_l(z_l)
 
        loss_l = self.loss_function(output_l, label_ids, mu_l, std_l)

        mu_a, std_a = self.encode_a(x_a)
        z_a = self.reparameterise(mu_a, std_a)
        output_a =  self.decode_a(z_a)
 
        loss_a = self.loss_function(output_a, label_ids, mu_a, std_a)

        mu_v, std_v = self.encode_v(x_v)
        z_v = self.reparameterise(mu_v, std_v)
        output_v =  self.decode_v(z_v)
 
        loss_v = self.loss_function(output_v, label_ids, mu_v, std_v)

       # outputf = torch.cat([z_l, z_a, z_v], dim=-1)

        outputf = self.fusion1(z_l, z_a, z_v)

        mu, std = self.encode(outputf)
        z = self.reparameterise(mu, std)
        output =  self.decode(z)


        loss = self.loss_function(output, label_ids, mu, std)

        return output, loss_l + loss_a + loss_v + loss


    def test(
        self,
        x_l,
        x_a,
        x_v
    ):


        mu_l, std_l = self.encode_l(x_l)
        z_l = self.reparameterise(mu_l, std_l)
        output_l =  self.decode_l(z_l)
 


        mu_a, std_a = self.encode_a(x_a)
        z_a = self.reparameterise(mu_a, std_a)
        output_a =  self.decode_a(z_a)


        mu_v, std_v = self.encode_v(x_v)
        z_v = self.reparameterise(mu_v, std_v)
        output_v =  self.decode_v(z_v)
 
        outputf = self.fusion1(z_l, z_a, z_v)

        mu, std = self.encode(outputf)
        z = self.reparameterise(mu, std)
        output =  self.decode(z)
    


        return output
    

class concat(nn.Module):

    def __init__(self, in_size, output_dim,hidden = 50, dropout=0.5):

        super(concat, self).__init__()
 

        self.linear_1 = nn.Linear(in_size*3, output_dim)
       # self.linear_1 = nn.Linear(in_size, hidden)


    def forward(self, l1, a1, v1):
     
        fusion = torch.cat([l1, a1, v1], dim=-1)

        y_1 = torch.relu(self.linear_1(fusion))


        return y_1
    
if __name__ == '__main__':
    model = MIB()
    r = torch.tensor(torch.rand(2, 1781))
    v = torch.tensor(torch.rand(2, 512))
    t = torch.tensor(torch.rand(2, 58))
    label = torch.tensor([1, 0])
    output, loss = model(v, r,  t, label)
    print(output.shape)
    print(loss)