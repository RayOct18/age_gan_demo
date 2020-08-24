import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import single_DR_GAN_model_gender_pyramid_GN_132 as single_model
import util
import numpy as np
import matplotlib.pyplot as plt
import cv2

class Model():
    def __init__(self, args):
        self.G_model, self.D_model = self.load_model(args)
        self.Na = args.Na
        self.Ng = args.Ng
        

    def load_model(self, args):
        if args.snapshot is None:
            print("Sorry, please set snapshot path while generate")
            exit()
        else:
            print('\nLoading model from [%s]...' % args.snapshot)
            checkpoint = torch.load('{}_checkpoint.pth.tar'.format(args.snapshot))
            D = single_model.Discriminator(args)
            G = single_model.Generator(args)
            D.load_state_dict(checkpoint['D_model'])
            G.load_state_dict(checkpoint['G_model'])

        G.cuda()
        D.cuda()
        G.eval()
        D.eval()

        return G, D

    def classifier(self, batch_image):
        batch_image = torch.FloatTensor(batch_image)
        batch_image = Variable(batch_image.cuda(), requires_grad=True)
        real_output = self.D_model(batch_image, EnableWasserstein=False)
        _, age_real_ans = torch.max(real_output[0][:, :self.Na], 1)
        _, gender_real_ans = torch.max(real_output[0][:, self.Na:], 1)
        return age_real_ans, gender_real_ans
    
    def generate_image(self, batch_image, age, gender):
        batch_image = np.repeat(batch_image, 8, axis=0)
        batch_image = torch.FloatTensor(batch_image)
        batch_image = Variable(batch_image.cuda(), requires_grad=True)

        age_code = torch.LongTensor(np.arange(self.Na)).repeat(1,2)[0]
        batch_age_code = util.one_hot(age_code, self.Na)

        male_code = torch.LongTensor(np.ones(len(batch_image)//2) * int(0))
        female_code = torch.LongTensor(np.ones(len(batch_image)//2) * int(1))
        gender_code = torch.cat((male_code, female_code), 0)
        batch_gender_code = util.one_hot(gender_code, self.Ng)

        batch_age_code, batch_gender_code = \
            Variable(batch_age_code.cuda()), Variable(batch_gender_code.cuda())

        generated = self.G_model(batch_image, batch_age_code, batch_gender_code)
        return generated[age + gender]

    