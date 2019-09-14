import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import Dataset
from .models import InpaintingModel
from .utils import Progbar, create_dir, stitch_images, imsave
from .metrics import PSNR, SSIM
from libs.logger import Logger


# cross_level Features Net
class CLFNet():
    def __init__(self, config):
        self.config = config
        model_name = 'inpaint'
        self.debug = False
        self.model_name = model_name
        self.inpaint_model = InpaintingModel(config).to(config.DEVICE)

        self.psnr = PSNR(255.0).to(config.DEVICE)
        self.ssim = SSIM(window_size=11)

        val_sample = int(float((self.config.EVAL_INTERVAL)))
        # test mode
        if self.config.MODE == 2:
            self.test_dataset = Dataset(config, config.TEST_FLIST, config.TEST_EDGE_FLIST, config.TEST_MASK_FLIST,
                                        augment=False, training=False)
        else:
            self.train_dataset = Dataset(config, config.TRAIN_FLIST, config.TRAIN_EDGE_FLIST, config.TRAIN_MASK_FLIST,
                                         augment=True, training=True)
            self.val_dataset = Dataset(config, config.VAL_FLIST, config.VAL_EDGE_FLIST, config.VAL_MASK_FLIST,
                                       augment=False, training=True, sample_interval=val_sample)
            self.sample_iterator = self.val_dataset.create_iterator(config.SAMPLE_SIZE)

        self.samples_path = os.path.join(config.PATH, 'samples')
        self.results_path = os.path.join(config.PATH, 'results')

        if config.RESULTS is not None:
            self.results_path = os.path.join(config.RESULTS)

        if config.DEBUG is not None and config.DEBUG != 0:
            self.debug = True

        # self.log_file = os.path.join(config.PATH, 'log_' + model_name + '.dat')
        log_path = os.path.join(config.PATH, 'logs_' + model_name)
        create_dir(log_path)
        self.logger = Logger(log_path)

    def load(self):
        self.inpaint_model.load()

    def save(self, epoch):
        self.inpaint_model.save(epoch)

    def train(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=4,
            drop_last=True,
            shuffle=True
        )

        keep_training = True
        model = self.config.MODEL
        # max_iteration = int(float((self.config.MAX_ITERS)))
        step_per_epoch = int(float((self.config.MAX_STEPS)))
        max_epoches = int(float((self.config.MAX_EPOCHES)))
        total = int(len(self.train_dataset))

        if total == 0:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return

        print('\nThe number of Training data is %d' % total)

        epoch = self.inpaint_model.epoch + 1 if self.inpaint_model.epoch != None else 1

        print('\nTraining epoch: %d' % epoch)
        progbar = Progbar(step_per_epoch, width=30, stateful_metrics=['step'])
        logs_ave = {}
        while (keep_training):
            for items in train_loader:
                self.inpaint_model.train()
                images, images_gray, edges, masks = self.cuda(*items)
                # edge model
                # inpaint model

                # train
                outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, masks)
                outputs_merged = (outputs * (1 - masks)) + (images * (masks))

                # metrics
                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                logs['psnr'] = psnr.item()
                logs['mae'] = mae.item()

                # backward
                self.inpaint_model.backward(gen_loss, dis_loss)
                if self.inpaint_model.iteration > step_per_epoch:
                    self.inpaint_model.iteration = 0
                    iteration=0
                iteration = self.inpaint_model.iteration

                if iteration == 1:  # first step in this epoch
                    for tag, value in logs.items():
                        logs_ave[tag] = value
                else:
                    for tag, value in logs.items():
                        logs_ave[tag] += value
                if iteration == 0:  # mean to jump to new epoch

                    self.sample(epoch)
                    self.eval(epoch)
                    self.save(epoch)

                    # log current epoch in tensorboard
                    for tag, value in logs_ave.items():
                        self.logger.scalar_summary(tag, value / step_per_epoch, epoch)

                    # if reach max epoch
                    if epoch >= max_epoches:
                        keep_training = False
                        break
                    epoch += 1
                    # new epoch
                    print('\n\nTraining epoch: %d' % epoch)
                    for tag, value in logs.items():
                        logs_ave[tag] = value
                    progbar = Progbar(step_per_epoch, width=30, stateful_metrics=['step'])
                    self.inpaint_model.iteration += 1  # jump to new epoch and set the iteration to 1
                    iteration += 1
                logs['step'] = iteration
                progbar.add(1,
                            values=logs.items() if self.config.VERBOSE else [x for x in logs.items() if
                                                                             not x[0].startswith('l_')])
            print("The whole data hase been trained %d times"%)

        print('\nEnd training....\n')

    def eval(self, epoch):
        self.val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.BATCH_SIZE,
            drop_last=True,
            shuffle=False,
            num_workers=4
        )
        model = self.config.MODEL
        total = int(len(self.val_dataset))

        self.inpaint_model.eval()

        progbar = Progbar(int(total / self.config.BATCH_SIZE), width=30, stateful_metrics=['step'])
        iteration = 0
        with torch.no_grad():
            for items in self.val_loader:
                iteration += 1
                images, images_gray, edges, masks = self.cuda(*items)
                # inpaint model
                # eval
                outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, masks)
                outputs_merged = (outputs * (1 - masks)) + (images * (masks))

                # metrics
                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                logs['val_psnr'] = psnr.item()
                logs['val_mae'] = mae.item()
                # joint model
                if iteration == 1:  # first iteration
                    logs_ave = {}
                    for tag, value in logs.items():
                        logs_ave[tag] = value
                else:
                    for tag, value in logs.items():
                        logs_ave[tag] += value

                logs["step"] = iteration
                progbar.add(1, values=logs.items())

            for tag, value in logs_ave.items():
                self.logger.scalar_summary(tag, value / iteration, epoch)
            self.inpaint_model.iteration = 0

    def test(self):
        self.inpaint_model.eval()
        damaged_dir = os.path.join(self.results_path, "damaged")
        create_dir(damaged_dir)
        mask_dir = os.path.join(self.results_path, "mask")
        create_dir(mask_dir)
        inpainted_dir = os.path.join(self.results_path, "inpainted")
        create_dir(inpainted_dir)
        damaged_edge_dir = os.path.join(self.results_path, "damaged_edge")
        create_dir(damaged_edge_dir)
        raw_dir = os.path.join(self.results_path, "raw")
        create_dir(raw_dir)

        model = self.config.MODEL
        create_dir(self.results_path)
        sample_interval = 1000
        batch_size = 1
        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=batch_size,
            num_workers=12,
            shuffle=False
        )

        total = int(len(self.test_dataset))
        progbar = Progbar(int(total / batch_size / sample_interval), width=30, stateful_metrics=['step'])

        index = 0
        with torch.no_grad():
            for items in test_loader:
                name = self.test_dataset.load_name(index)
                images, images_gray, edges, masks = self.cuda(*items)

                path = os.path.join(damaged_dir, name)
                damaged_img = self.postprocess(images * masks + (1 - masks))[0]
                imsave(damaged_img, path)
                path = os.path.join(damaged_edge_dir, name)
                damaged_img = self.postprocess(edges * masks + (1 - masks))[0]
                imsave(damaged_img, path)
                path = os.path.join(mask_dir, name)
                imsave(self.postprocess(masks), os.path.splitext(path)[0] + '.png')
                path = os.path.join(raw_dir, name)
                img = self.postprocess(images)[0]
                imsave(img, path)
                # print(index, name)

                index += 1
                if index > total / batch_size / sample_interval:
                    break;
                logs = {}
                # edge model
                if model == 1:
                    outputs = self.edge_model(images_gray, edges, masks)
                    outputs_merged = (outputs * (1 - masks)) + (edges * masks)

                # inpaint model
                elif model == 2:
                    outputs = self.inpaint_model(images, edges, masks)
                    outputs_merged = (outputs * (1 - masks)) + (images * masks)

                # edge-inpaint model
                elif model == 3:
                    outputs = self.edge_model(images_gray, edges, masks).detach()
                    outputs_merged = (outputs * (1 - masks)) + (edges * masks)
                    edge_save = self.color_the_edge(outputs, edges, masks)
                    edge_save = self.postprocess(edge_save)[0]
                    path = os.path.join(self.results_path + "/edge_inpainted", name)
                    # print(index, name)
                    imsave(edge_save, path)
                    outputs = self.inpaint_model(images, outputs_merged, masks)
                    outputs_merged = (outputs * (1 - masks)) + (images * masks)



                # joint model
                else:
                    edges = self.edge_model(images_gray, edges, masks).detach()
                    outputs = self.inpaint_model(images, edges, masks)
                    outputs_merged = (outputs * (1 - masks)) + (images * masks)

                output = self.postprocess(outputs_merged)[0]
                path = os.path.join(inpainted_dir, name)
                # print(index, name)
                imsave(output, path)
                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                one_ssim = self.ssim(images, outputs_merged)
                logs["psnr"] = psnr.item()
                logs["mae"] = mae.item()
                logs["ssim"] = one_ssim.item()
                logs["step"] = index
                progbar.add(1, values=logs.items())

                if self.debug:
                    edges = self.postprocess(1 - edges)[0]
                    masked = self.postprocess(images * (masks) + (1 - masks))[0]
                    fname, fext = name.split('.')

                    imsave(edges, os.path.join(self.results_path, fname + '_edge.' + fext))
                    imsave(masked, os.path.join(self.results_path, fname + '_masked.' + fext))

        print('\nEnd test....')

    def sample(self, it=None):
        # do not sample when validation set is empty
        if len(self.val_dataset) == 0:
            return

        self.inpaint_model.eval()

        model = self.config.MODEL
        with torch.no_grad():
            items = next(self.sample_iterator)
            images, images_gray, edges, masks = self.cuda(*items)
            # (batch,channels,weighth,length)
            iteration = self.inpaint_model.iteration
            inputs = (images * masks) + (1 - masks)
            outputs = self.inpaint_model(images, masks).detach()

            if it is not None:
                iteration = it

            image_per_row = 2
            if self.config.SAMPLE_SIZE <= 6:
                image_per_row = 1

            images = stitch_images(
                self.postprocess(images),
                # self.postprocess(edges),
                self.postprocess(inputs),
                self.postprocess(outputs),
                # self.postprocess(outputs_merged),
                img_per_row=image_per_row
            )

            path = os.path.join(self.samples_path, self.model_name)
            name = os.path.join(path, str(iteration).zfill(3) + ".png")
            create_dir(path)
            print('\nsaving sample ' + name)
            images.save(name)

    def log(self, logs):
        with open(self.log_file, 'a') as f:
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()

    def color_the_edge(self, img, edges, masks):
        img = img.expand(-1, 3, -1, -1)
        yellow_v = (torch.tensor([215. / 255., 87. / 255., 15. / 255.]).reshape(1, 3, 1, 1)).to(self.config.DEVICE)
        yellow = img * (1 - masks) * yellow_v
        img = yellow + (edges * masks)
        return img
