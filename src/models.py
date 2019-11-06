import os
import torch
import torch.nn as nn
import torch.optim as optim

from ablation.network_fusion import RFFNet_fusion
from .networks import Discriminator, UnetGenerator, UnetGeneratorSame, RFFNet
from .loss import AdversarialLoss, PerceptualLoss, StyleLoss, GradientLoss


class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()

        self.name = name
        self.config = config
        self.iteration = 0
        self.epoch = None
        self.gen_weights_path = os.path.join(config.PATH, name + '_gen.pth')
        self.dis_weights_path = os.path.join(config.PATH, name + '_dis.pth')

    def load(self):
        if os.path.exists(self.gen_weights_path):
            print('Loading %s generator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.gen_weights_path)
            else:
                data = torch.load(self.gen_weights_path, map_location=lambda storage, loc: storage)

            self.generator.load_state_dict(data['generator'])
            # self.iteration = data['iteration']
            if self.config.LOADWITHEPOCH == 1:
                self.epoch = data['epoch']

        # load discriminator only when training
        if self.config.MODE == 1 and os.path.exists(self.dis_weights_path):
            print('Loading %s discriminator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.dis_weights_path)
            else:
                data = torch.load(self.dis_weights_path, map_location=lambda storage, loc: storage)

            self.discriminator.load_state_dict(data['discriminator'])

    def save(self, epoch):
        print('\nsaving %s...\n' % self.name)
        torch.save({
            # 'iteration': self.iteration,
            'generator': self.generator.state_dict(),
            'epoch': epoch
        }, os.path.join(os.path.dirname(self.gen_weights_path), self.name + '_%d_gen.pth' % (epoch)))

        torch.save({
            'discriminator': self.discriminator.state_dict()
        }, os.path.join(os.path.dirname(self.dis_weights_path), self.name + '_%d_dis.pth' % (epoch)))


class InpaintingModel(BaseModel):
    def __init__(self, config):
        super(InpaintingModel, self).__init__('InpaintingModel', config)

        # generator input: [rgb(3)]
        in_channel=3
        generator = RFFNet(in_channel, config.BLOCKS)
        # generator=UnetGeneratorSame()       Unet-lile generator
        print("This Model Total params:", (sum([param.nelement() for param in generator.parameters()])))
        # summary(generator, (3, 256, 256), 6,device='cpu')
        # print(generator)

        # discriminator input: [rgb(3)]
        discriminator = Discriminator(in_channels=3, use_sigmoid=config.GAN_LOSS != 'hinge')
        if len(config.GPU) > 1:
            generator = nn.DataParallel(generator, config.GPU)
            discriminator = nn.DataParallel(discriminator, config.GPU)

        l1_loss = nn.L1Loss()
        perceptual_loss = PerceptualLoss()
        style_loss = StyleLoss()
        adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)
        gradient_loss = GradientLoss(independent=True, distance='L2')

        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)
        self.add_module('l1_loss', l1_loss)
        self.add_module('perceptual_loss', perceptual_loss)
        self.add_module('style_loss', style_loss)
        self.add_module('adversarial_loss', adversarial_loss)
        self.add_module('gradient_loss', gradient_loss)

        self.gen_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )

        self.dis_optimizer = optim.Adam(
            params=discriminator.parameters(),
            lr=float(config.LR) * float(config.D2G_LR),
            betas=(config.BETA1, config.BETA2)
        )

    def process(self, images, masks):
        self.iteration += 1

        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()

        # process outputs
        outputs = self(images, masks)
        gen_loss = 0
        dis_loss = 0

        # discriminator loss
        dis_input_real = images
        # dis_input_real =torch.cat((images, masks), dim=1)
        dis_input_fake = outputs.detach()
        # dis_input_fake =torch.cat((outputs.detach(), masks), dim=1)
        dis_real, dis_real_feat = self.discriminator(dis_input_real)  # in: [rgb(3)]
        dis_fake, dis_fake_feat = self.discriminator(dis_input_fake)  # in: [rgb(3)]

        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2

        # generator adversarial loss
        gen_gan_loss=torch.FloatTensor([0])
        if self.config.INPAINT_ADV_LOSS_WEIGHT > 0:
            gen_input_fake = outputs
            # gen_input_fake = torch.cat((outputs, masks), dim=1)
            gen_fake, gen_fake_feat = self.discriminator(gen_input_fake)  # in: [rgb(3)]
            gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
            gen_loss += gen_gan_loss

        # generator feature matching loss
        gen_fm_loss = torch.FloatTensor([0])
        if self.config.FM_LOSS_WEIGHT > 0:
            gen_fm_loss = 0
            for i in range(len(dis_real_feat)):
                gen_fm_loss += self.l1_loss(gen_fake_feat[i], dis_real_feat[i].detach())
            gen_fm_loss = gen_fm_loss * self.config.FM_LOSS_WEIGHT
            gen_loss += gen_fm_loss

        # generator l1 loss
        gen_l1_loss = torch.FloatTensor([0])
        if self.config.L1_LOSS_WEIGHT > 0:
            gen_l1_loss = self.l1_loss(outputs, images) * self.config.L1_LOSS_WEIGHT / torch.mean(1 - masks)
            gen_loss += gen_l1_loss

        # # generator perceptual loss
        gen_content_loss=torch.FloatTensor([0])
        if self.config.CONTENT_LOSS_WEIGHT>0:
            gen_content_loss = self.perceptual_loss(outputs, images)
            gen_content_loss = gen_content_loss * self.config.CONTENT_LOSS_WEIGHT
            gen_loss += gen_content_loss

        # # generator style loss
        gen_style_loss=torch.FloatTensor([0])
        if self.config.STYLE_LOSS_WEIGHT > 0:
            gen_style_loss = self.style_loss(outputs, images)
            gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_WEIGHT
            gen_loss += gen_style_loss

        # gradient loss
        gen_gradient_loss = torch.FloatTensor([0])
        if self.config.GRADIENT_LOSS_WEIGHT > 0:
            gen_gradient_loss=self.gradient_loss(outputs,images)
            gen_gradient_loss= gen_gradient_loss * self.config.GRADIENT_LOSS_WEIGHT
            gen_loss += gen_gradient_loss
        # create logs
        logs = {
            "l_d2": dis_loss.item(),
            "l_g2": gen_gan_loss.item(),
            "l_l1": gen_l1_loss.item(),
            "l_fm": gen_fm_loss.item(),
            "l_per": gen_content_loss.item(),
            "l_sty": gen_style_loss.item(),
            'l_grad': gen_gradient_loss.item()
        }

        if not self.training:
            val_logs = {}
            for key, value in logs.items():
                key = "val_" + key
                val_logs[key] = value
            logs = val_logs
        return outputs, gen_loss, dis_loss, logs

    def forward(self, images, masks):
        # images_masked = (images * (1 - masks).float()) + masks
        images_masked = (images * (masks).float())
        inputs = images_masked
        outputs = self.generator(inputs)  # in: [rgb(3)]
        return outputs

    def backward(self, gen_loss=None, dis_loss=None):
        dis_loss.backward()
        self.dis_optimizer.step()

        gen_loss.backward()
        self.gen_optimizer.step()
