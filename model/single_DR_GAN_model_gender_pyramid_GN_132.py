#!/usr/bin/env python
# encoding: utf-8

import torch
from torch import nn, optim
from torch.autograd import Variable, grad
import math
import torch.nn.functional as F
from torch.nn import Parameter
from torchvision import models
from collections import namedtuple


def myphi(x,m):
    x = x * m
    return 1-x**2/math.factorial(2)+x**4/math.factorial(4)-x**6/math.factorial(6) + \
            x**8/math.factorial(8) - x**9/math.factorial(9)

class AngleLinear(nn.Module):
    def __init__(self, in_features, out_features, m = 2, phiflag=True):
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features,out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.phiflag = phiflag
        self.m = m
        self.mlambda = [
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2*x**2-1,
            lambda x: 4*x**3-3*x,
            lambda x: 8*x**4-8*x**2+1,
            lambda x: 16*x**5-20*x**3+5*x
        ]

    def forward(self, input):
        x = input   # size=(B,F)    F is feature len
        w = self.weight # size=(F,Classnum) F=in_features Classnum=out_features

        ww = w.renorm(2,1,1e-5).mul(1e5)
        xlen = x.pow(2).sum(1).pow(0.5) # size=B
        wlen = ww.pow(2).sum(0).pow(0.5) # size=Classnum

        cos_theta = x.mm(ww) # size=(B,Classnum)
        cos_theta = cos_theta / xlen.view(-1,1) / wlen.view(1,-1)
        cos_theta = cos_theta.clamp(-1,1)

        if self.phiflag:
            cos_m_theta = self.mlambda[self.m](cos_theta)
            theta = Variable(cos_theta.data.acos())
            k = (self.m*theta/3.14159265).floor()
            n_one = k*0.0 - 1
            phi_theta = (n_one**k) * cos_m_theta - 2*k
        else:
            theta = cos_theta.acos()
            phi_theta = myphi(theta,self.m)
            phi_theta = phi_theta.clamp(-1*self.m,1)

        cos_theta = cos_theta * xlen.view(-1,1)
        phi_theta = phi_theta * xlen.view(-1,1)
        output = (cos_theta,phi_theta)
        return output # size=(B,Classnum,2)


class AngleLoss(nn.Module):
    def __init__(self, gamma=0):
        super(AngleLoss, self).__init__()
        self.gamma = gamma
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0

    def forward(self, input, target):
        self.it += 1
        cos_theta,phi_theta = input
        target = target.view(-1,1) #size=(B,1)

        index = cos_theta.data * 0.0 #size=(B,Classnum)
        index.scatter_(1,target.data.view(-1,1),1)
        index = index.byte()
        index = Variable(index)

        self.lamb = max(self.LambdaMin,self.LambdaMax/(1+0.1*self.it))
        output = cos_theta * 1.0 #size=(B,Classnum)
        output[index] -= cos_theta[index]*(1.0+0)/(1+self.lamb)
        output[index] += phi_theta[index]*(1.0+0)/(1+self.lamb)

        logpt = F.log_softmax(output)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss.mean()

        return loss


loss_criterion = nn.CrossEntropyLoss().cuda()
# loss_criterion = nn.BCELoss().cuda()
loss_criterion_Angular = AngleLoss().cuda()

class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.features = []
        self.Na = args.Na
        self.Ng = args.Ng
        self.AngleLoss = args.Angle_Loss
        self.Channel = 3
        convLayers11 = [
            nn.Conv2d(self.Channel, 32, 3, 1, 1, bias=False),  # Bxchx96x96 -> Bx32x96x96
            nn.GroupNorm(2,32),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),  # Bx32x96x96 -> Bx64x96x96
            nn.GroupNorm(4,64),
            nn.ELU(inplace=True),
            nn.ZeroPad2d((0, 1, 0, 1)),  # Bx64x96x96 -> Bx64x97x97
        ]
        convLayers12 = [
            nn.Conv2d(64, 64, 3, 2, 0, bias=False),  # Bx64x97x97 -> Bx64x48x48
            nn.GroupNorm(4,64),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),  # Bx64x48x48 -> Bx64x48x48
            nn.GroupNorm(4,64),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),  # Bx64x48x48 -> Bx128x48x48
            nn.GroupNorm(8,128),
            nn.ELU(inplace=True),
            nn.ZeroPad2d((0, 1, 0, 1)),  # Bx128x48x48 -> Bx128x49x49
        ]
        convLayers13 = [
            nn.Conv2d(128, 128, 3, 2, 0, bias=False),  # Bx128x49x49 -> Bx128x24x24
            nn.GroupNorm(8,128),
            nn.ELU(inplace=True),
            nn.Conv2d(128, 96, 3, 1, 1, bias=False),  # Bx128x24x24 -> Bx96x24x24
            nn.GroupNorm(6,96),
            nn.ELU(inplace=True),
            nn.Conv2d(96, 192, 3, 1, 1, bias=False),  # Bx96x24x24 -> Bx192x24x24
            nn.GroupNorm(12,192),
            nn.ELU(inplace=True),
            nn.ZeroPad2d((0, 1, 0, 1)),  # Bx192x24x24 -> Bx192x25x25
        ]
        convLayers14 = [
            nn.Conv2d(192, 192, 3, 2, 0, bias=False),  # Bx192x25x25 -> Bx192x12x12
            nn.GroupNorm(12,192),
            nn.ELU(inplace=True),
            nn.Conv2d(192, 128, 3, 1, 1, bias=False),  # Bx192x12x12 -> Bx128x12x12
            nn.GroupNorm(8, 128),
            nn.ELU(inplace=True),
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),  # Bx128x12x12 -> Bx256x12x12
            nn.GroupNorm(16, 256),
            nn.ELU(inplace=True),
            nn.ZeroPad2d((0, 1, 0, 1)),  # Bx256x12x12 -> Bx256x13x13
        ]
        convLayers15 = [
            nn.Conv2d(256, 256, 3, 2, 0, bias=False),  # Bx256x13x13 -> Bx256x6x6
            nn.GroupNorm(16, 256),
            nn.ELU(inplace=True),
            nn.Conv2d(256, 160, 3, 1, 1, bias=False),  # Bx256x6x6 -> Bx160x6x6
            nn.GroupNorm(10, 160),
            nn.ELU(inplace=True),
            nn.Conv2d(160, 320, 3, 1, 1, bias=False),  # Bx160x6x6 -> Bx320x6x6
            nn.GroupNorm(20, 320),
            nn.ELU(inplace=True),
            # nn.Dropout(p=0.5),
            nn.AvgPool2d(8, stride=1),  # Bx320x6x6 -> Bx320x1x1
        ]

        convLayers2 = [
            nn.Conv2d(64, 64, 3, 2, 0, bias=False),  # Bx64x97x97 -> Bx64x48x48
            nn.GroupNorm(4,64),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),  # Bx64x48x48 -> Bx64x48x48
            nn.GroupNorm(4,64),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),  # Bx64x48x48 -> Bx128x48x48
            nn.GroupNorm(8,128),
            nn.ELU(inplace=True),
            nn.ZeroPad2d((0, 1, 0, 1)),  # Bx128x48x48 -> Bx128x49x49
            nn.Conv2d(128, 128, 3, 2, 0, bias=False),  # Bx128x49x49 -> Bx128x24x24
            nn.GroupNorm(8,128),
            nn.ELU(inplace=True),
            nn.Conv2d(128, 96, 3, 1, 1, bias=False),  # Bx128x24x24 -> Bx96x24x24
            nn.GroupNorm(6,96),
            nn.ELU(inplace=True),
            nn.Conv2d(96, 192, 3, 1, 1, bias=False),  # Bx96x24x24 -> Bx192x24x24
            nn.GroupNorm(12,192),
            nn.ELU(inplace=True),
            nn.ZeroPad2d((0, 1, 0, 1)),  # Bx192x24x24 -> Bx192x25x25
            nn.Conv2d(192, 192, 3, 2, 0, bias=False),  # Bx192x25x25 -> Bx192x12x12
            nn.GroupNorm(12,192),
            nn.ELU(inplace=True),
            nn.Conv2d(192, 128, 3, 1, 1, bias=False),  # Bx192x12x12 -> Bx128x12x12
            nn.GroupNorm(8, 128),
            nn.ELU(inplace=True),
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),  # Bx128x12x12 -> Bx256x12x12
            nn.GroupNorm(16, 256),
            nn.ELU(inplace=True),
            nn.ZeroPad2d((0, 1, 0, 1)),  # Bx256x12x12 -> Bx256x13x13
            nn.Conv2d(256, 256, 3, 2, 0, bias=False),  # Bx256x13x13 -> Bx256x6x6
            nn.GroupNorm(16, 256),
            nn.ELU(inplace=True),
            nn.Conv2d(256, 160, 3, 1, 1, bias=False),  # Bx256x6x6 -> Bx160x6x6
            nn.GroupNorm(10, 160),
            nn.ELU(inplace=True),
            nn.Conv2d(160, 320, 3, 1, 1, bias=False),  # Bx160x6x6 -> Bx320x6x6
            nn.GroupNorm(20, 320),
            nn.ELU(inplace=True),
            # nn.Dropout(p=0.5),
            nn.AvgPool2d(8, stride=1),  # Bx320x6x6 -> Bx320x1x1
        ]

        convLayers3 = [
            nn.Conv2d(128, 128, 3, 2, 0, bias=False),  # Bx128x49x49 -> Bx128x24x24
            nn.GroupNorm(8,128),
            nn.ELU(inplace=True),
            nn.Conv2d(128, 96, 3, 1, 1, bias=False),  # Bx128x24x24 -> Bx96x24x24
            nn.GroupNorm(6,96),
            nn.ELU(inplace=True),
            nn.Conv2d(96, 192, 3, 1, 1, bias=False),  # Bx96x24x24 -> Bx192x24x24
            nn.GroupNorm(12,192),
            nn.ELU(inplace=True),
            nn.ZeroPad2d((0, 1, 0, 1)),  # Bx192x24x24 -> Bx192x25x25
            nn.Conv2d(192, 192, 3, 2, 0, bias=False),  # Bx192x25x25 -> Bx192x12x12
            nn.GroupNorm(12,192),
            nn.ELU(inplace=True),
            nn.Conv2d(192, 128, 3, 1, 1, bias=False),  # Bx192x12x12 -> Bx128x12x12
            nn.GroupNorm(8, 128),
            nn.ELU(inplace=True),
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),  # Bx128x12x12 -> Bx256x12x12
            nn.GroupNorm(16, 256),
            nn.ELU(inplace=True),
            nn.ZeroPad2d((0, 1, 0, 1)),  # Bx256x12x12 -> Bx256x13x13
            nn.Conv2d(256, 256, 3, 2, 0, bias=False),  # Bx256x13x13 -> Bx256x6x6
            nn.GroupNorm(16, 256),
            nn.ELU(inplace=True),
            nn.Conv2d(256, 160, 3, 1, 1, bias=False),  # Bx256x6x6 -> Bx160x6x6
            nn.GroupNorm(10, 160),
            nn.ELU(inplace=True),
            nn.Conv2d(160, 320, 3, 1, 1, bias=False),  # Bx160x6x6 -> Bx320x6x6
            nn.GroupNorm(20, 320),
            nn.ELU(inplace=True),
            # nn.Dropout(p=0.5),
            nn.AvgPool2d(8, stride=1),  # Bx320x6x6 -> Bx320x1x1
        ]

        convLayers4 = [
            nn.Conv2d(192, 192, 3, 2, 0, bias=False),  # Bx192x25x25 -> Bx192x12x12
            nn.GroupNorm(12,192),
            nn.ELU(inplace=True),
            nn.Conv2d(192, 128, 3, 1, 1, bias=False),  # Bx192x12x12 -> Bx128x12x12
            nn.GroupNorm(8, 128),
            nn.ELU(inplace=True),
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),  # Bx128x12x12 -> Bx256x12x12
            nn.GroupNorm(16, 256),
            nn.ELU(inplace=True),
            nn.ZeroPad2d((0, 1, 0, 1)),  # Bx256x12x12 -> Bx256x13x13
            nn.Conv2d(256, 256, 3, 2, 0, bias=False),  # Bx256x13x13 -> Bx256x6x6
            nn.GroupNorm(16, 256),
            nn.ELU(inplace=True),
            nn.Conv2d(256, 160, 3, 1, 1, bias=False),  # Bx256x6x6 -> Bx160x6x6
            nn.GroupNorm(10, 160),
            nn.ELU(inplace=True),
            nn.Conv2d(160, 320, 3, 1, 1, bias=False),  # Bx160x6x6 -> Bx320x6x6
            nn.GroupNorm(20, 320),
            nn.ELU(inplace=True),
            # nn.Dropout(p=0.5),
            nn.AvgPool2d(8, stride=1),  # Bx320x6x6 -> Bx320x1x1
        ]

        convLayers5 = [
            nn.Conv2d(256, 256, 3, 2, 0, bias=False),  # Bx256x13x13 -> Bx256x6x6
            nn.GroupNorm(16, 256),
            nn.ELU(inplace=True),
            nn.Conv2d(256, 160, 3, 1, 1, bias=False),  # Bx256x6x6 -> Bx160x6x6
            nn.GroupNorm(10, 160),
            nn.ELU(inplace=True),
            nn.Conv2d(160, 320, 3, 1, 1, bias=False),  # Bx160x6x6 -> Bx320x6x6
            nn.GroupNorm(20, 320),
            nn.ELU(inplace=True),
            # nn.Dropout(p=0.5),
            nn.AvgPool2d(8, stride=1),  # Bx320x6x6 -> Bx320x1x1
        ]

        self.convLayers11 = nn.Sequential(*convLayers11)
        self.convLayers12 = nn.Sequential(*convLayers12)
        self.convLayers13 = nn.Sequential(*convLayers13)
        self.convLayers14 = nn.Sequential(*convLayers14)
        self.convLayers15 = nn.Sequential(*convLayers15)
        self.convLayers2 = nn.Sequential(*convLayers2)
        self.convLayers3 = nn.Sequential(*convLayers3)
        self.convLayers4 = nn.Sequential(*convLayers4)
        self.convLayers5 = nn.Sequential(*convLayers5)

        # self.FC = nn.Linear(320, self.Ng + self.Na)  # delete the dimension for binary classification for GAN (Real/Fake)
        self.NEW_Wasserstein_FC = nn.Linear(960, 1)
        if self.AngleLoss:
            self.NEW_FC1 = AngleLinear(960, self.Na+self.Ng)
        else:
            self.NEW_FC1 = nn.Linear(960, self.Na+self.Ng)



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)

            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)

    def Disc(self, input):
        x1 = self.convLayers11(input)
        x2 = self.convLayers12(x1)
        x3 = self.convLayers13(x2)
        x4 = self.convLayers14(x3)
        out1 = self.convLayers15(x4)
        # out2 = self.convLayers2(x1)
        out3 = self.convLayers3(x2)
        # out4 = self.convLayers4(x3)
        out5 = self.convLayers5(x4)
        # x = torch.cat((out1,out2,out3,out4,out5), 1)
        x = torch.cat((out1,out3,out5), 1)
        x = x.view(-1, 960)
        y = x
        x = self.NEW_Wasserstein_FC(x)
        return x, y

    def Classify(self, input):
        x = self.NEW_FC1(input)

        return x

    def forward(self, input, EnableWasserstein=True):
        Gan_Out, y = self.Disc(input)
        self.features = y
        Classify_Out = self.Classify(y)

        if EnableWasserstein:
            return Gan_Out, Classify_Out
        else:
            return Classify_Out

    def CriticWithGP_Loss(self, Interpolate, Real_GAN, Syn_GAN, args):

        lmbda = args.lmbda
        loss = Syn_GAN.mean() - Real_GAN.mean()
        critic_loss_only = loss
        disc_out_gan, _ = self.Disc(Interpolate) # Gan_output, Feature = D_model(x)
        # get gradients of discriminator output with respect to input
        gradients = grad(outputs=disc_out_gan.sum(), inputs=Interpolate,
                         create_graph=True)[0]
        # calculate gradient penalty
        gradient_penalty = ((gradients.view(gradients.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()

        loss += lmbda * gradient_penalty

        return loss, gradients.view(gradients.size(0), -1).norm(2, dim=1).mean(), critic_loss_only

    # def ID_Loss(self, FC_Out, ID_Flag, Label, Flag=False):
    #     if Flag:
    #         # Get torch data that needs to calculate ID Label(Decided by ID Flag)
    #         # Use nonzero function to get the index that is non zero in ID Flag
    #         # Squeeze the dim=1 using squeeze to reshape (2,1)-->(2)
    #         # The squeezed idx variable can later fed into Variable.index_select(dim, idx) where idx needs to be a one dimension variable
    #         idx = Variable(torch.nonzero(ID_Flag).cuda())
    #         if len(idx.size()) == 1:
    #             loss = 0
    #         elif len(idx.size()) == 2:
    #             idx = idx.squeeze(1)
    #             batch_id_label_withFlag = Label.index_select(0, idx)
    #             if self.AngleLoss:
    #                 ID_Real = [FC_Out[0].index_select(0, idx), FC_Out[1].index_select(0, idx)]
    #                 loss = loss_criterion_Angular((ID_Real[0][:, :self.Ng], ID_Real[1][:, :self.Ng]), batch_id_label_withFlag)
    #             else:
    #                 ID_Real = FC_Out.index_select(0, idx)
    #                 loss = loss_criterion(ID_Real[:, :self.Ng], batch_id_label_withFlag)
    #     else:
    #         if self.AngleLoss:
    #             loss = loss_criterion_Angular((FC_Out[0][:, :self.Ng], FC_Out[1][:, :self.Ng]), Label)
    #         else:
    #             loss = loss_criterion(FC_Out[:, :self.Ng], Label)

    #     return loss

    def Pose_Loss(self, FC_Out, Label):
        if self.AngleLoss:
            loss = loss_criterion_Angular((FC_Out[0][:, :self.Na], FC_Out[1][:, :self.Na]), Label)
        else:
            loss = loss_criterion(FC_Out[:, :self.Na], Label)

        return loss

    def Gender_Loss(self, FC_Out, Label):
        if self.AngleLoss:
            loss = loss_criterion_Angular((FC_Out[0][:, self.Na:], FC_Out[1][:, self.Na:]), Label)
        else:
            loss = loss_criterion(FC_Out[:, self.Na:], Label)

        return loss


class Crop(nn.Module):

    def __init__(self, crop_list):
        super(Crop, self).__init__()

        # crop_lsit = [crop_top, crop_bottom, crop_left, crop_right]
        self.crop_list = crop_list

    def forward(self, x):
        B,C,H,W = x.size()
        x = x[:,:, self.crop_list[0] : H - self.crop_list[1] , self.crop_list[2] : W - self.crop_list[3]]

        return x


class Generator(nn.Module):

    def __init__(self, args):
        super(Generator, self).__init__()
        self.features = []
        self.Na = args.Na
        self.Ng = args.Ng
        self.Channel = 3
        G_enc_convLayers = [
            nn.Conv2d(self.Channel, 32, 3, 1, 1), # Bxchx96x96 -> Bx3
            nn.GroupNorm(2,32),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1), # Bx32x96x96 -> Bx64x96x96
            nn.GroupNorm(4,64),
            nn.ELU(inplace=True),
            nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx64x96x96 -> Bx64x97x97
            nn.Conv2d(64, 64, 3, 2, 0), # Bx64x97x97 -> Bx64x48x48
            nn.GroupNorm(4,64),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1), # Bx64x48x48 -> Bx64x48x48
            nn.GroupNorm(4,64),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 128, 3, 1, 1), # Bx64x48x48 -> Bx128x48x48
            nn.GroupNorm(8,128),
            nn.ELU(inplace=True),
            nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx128x48x48 -> Bx128x49x49
            nn.Conv2d(128, 128, 3, 2, 0), #  Bx128x49x49 -> Bx128x24x24
            nn.GroupNorm(8,128),
            nn.ELU(inplace=True),
            nn.Conv2d(128, 96, 3, 1, 1), #  Bx128x24x24 -> Bx96x24x24
            nn.GroupNorm(6,96),
            nn.ELU(inplace=True),
            nn.Conv2d(96, 192, 3, 1, 1), #  Bx96x24x24 -> Bx192x24x24
            nn.GroupNorm(12,192),
            nn.ELU(inplace=True),
            nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx192x24x24 -> Bx192x25x25
            nn.Conv2d(192, 192, 3, 2, 0), # Bx192x25x25 -> Bx192x12x12
            nn.GroupNorm(12,192),
            nn.ELU(inplace=True),
            nn.Conv2d(192, 128, 3, 1, 1), # Bx192x12x12 -> Bx128x12x12
            nn.GroupNorm(8, 128),
            nn.ELU(inplace=True),
            nn.Conv2d(128, 256, 3, 1, 1), # Bx128x12x12 -> Bx256x12x12
            nn.GroupNorm(16, 256),
            nn.ELU(inplace=True),
            nn.ZeroPad2d((0, 1, 0, 1)),                # Bx256x12x12 -> Bx256x13x13
            nn.Conv2d(256, 256, 3, 2, 0),  # Bx256x13x13 -> Bx256x6x6
            nn.GroupNorm(16, 256),
            nn.ELU(inplace=True),
            nn.Conv2d(256, 160, 3, 1, 1), # Bx256x6x6 -> Bx160x6x6
            nn.GroupNorm(10, 160),
            nn.ELU(inplace=True),
            nn.Conv2d(160, 320, 3, 1, 1), # Bx160x6x6 -> Bx320x6x6
            nn.GroupNorm(20, 320),
            nn.ELU(inplace=True),
            nn.AvgPool2d(8, stride=1), #  Bx320x6x6 -> Bx320x1x1
        ]
        self.G_enc_convLayers = nn.Sequential(*G_enc_convLayers)

        G_dec_convLayers = [
            nn.ConvTranspose2d(320, 160, 3, 1, 1), # Bx320x6x6 -> Bx160x6x6
            nn.GroupNorm(10, 160),
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(160, 256, 3,1,1), # Bx160x6x6 -> Bx256x6x6
            nn.GroupNorm(16, 256),
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(256, 256, 3,2,0), # Bx256x6x6 -> Bx256x13x13
            nn.GroupNorm(16, 256),
            nn.ELU(inplace=True),
            Crop([0, 1, 0, 1]),
            nn.ConvTranspose2d(256, 128, 3,1,1), # Bx256x12x12 -> Bx128x12x12
            nn.GroupNorm(8, 128),
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(128, 192,  3,1,1), # Bx128x12x12 -> Bx192x12x12
            nn.GroupNorm(12, 192),
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(192, 192,  3,2,0), # Bx128x12x12 -> Bx192x25x25
            nn.GroupNorm(12, 192),
            nn.ELU(inplace=True),
            Crop([0, 1, 0, 1]),
            nn.ConvTranspose2d(192, 96,  3,1,1), # Bx192x24x24 -> Bx96x24x24
            nn.GroupNorm(6, 96),
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(96, 128,  3,1,1), # Bx96x24x24 -> Bx128x24x24
            nn.GroupNorm(8, 128),
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(128, 128,  3,2,0), # Bx128x24x24 -> Bx128x49x49
            nn.GroupNorm(8, 128),
            nn.ELU(inplace=True),
            Crop([0, 1, 0, 1]),
            nn.ConvTranspose2d(128, 64,  3,1,1), # Bx128x48x48 -> Bx64x48x48
            nn.GroupNorm(4, 64),
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(64, 64,  3,1,1), # Bx64x48x48 -> Bx64x48x48
            nn.GroupNorm(4, 64),
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(64, 64,  3,2,0), # Bx64x48x48 -> Bx64x97x97
            nn.GroupNorm(4, 64),
            nn.ELU(inplace=True),
            Crop([0, 1, 0, 1]),
            nn.ConvTranspose2d(64, 32,  3,1,1), # Bx64x96x96 -> Bx32x96x96
            nn.GroupNorm(2, 32),
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(32, self.Channel,  3,1,1), # Bx32x96x96 -> Bxchx96x96
            nn.Tanh(),
        ]

        self.G_dec_convLayers = nn.Sequential(*G_dec_convLayers)

        self.G_DEC_FC1 = nn.Linear(320+self.Na+self.Ng, 320*8*8)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)

            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)

            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)

    def encoder(self, input, Interpolation=False):
        x = self.G_enc_convLayers(input)  # Bxchx96x96 -> Bx320x1x1

        x = x.view(-1, 320)

        y = x

        if Interpolation:
            gp_alpha = torch.FloatTensor(x.size()[0], 1)
            idx = Variable(torch.randperm(x.size()[0])).cuda()
            gp_alpha = Variable(gp_alpha.uniform_()).cuda()
            Shuffle_x = x.index_select(0, idx)
            x = x * gp_alpha + Shuffle_x * (1 - gp_alpha)

        return x

    def decoder(self, Latent, yaw, gender):
        x = torch.cat([Latent, yaw, gender], 1)  # Bx320 -> B x (320+Np+Nz+Nl+Ne)

        x = self.G_DEC_FC1(x)  # B x (320+Np+Nz+Nl+Ne) -> B x (320x6x6)

        x = x.view(-1, 320, 8, 8)  # B x (320x6x6) -> B x 320 x 6 x 6

        x = self.G_dec_convLayers(x)  # B x 320 x 6 x 6 -> Bxchx96x96

        return x

    def forward(self, input, yaw, gender, Interpolation=False):

        x = self.encoder(input, Interpolation=Interpolation)

        self.features = x

        Generate = self.decoder(x, yaw, gender)

        return Generate

    def G_Loss(self, Syn_GAN):

        loss = -Syn_GAN.mean()

        return loss