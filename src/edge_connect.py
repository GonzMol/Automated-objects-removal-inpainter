'''
        Code of EdgeConnect is from this repo
        https://github.com/knazeri/edge-connect
        '''



import os
import numpy as np
from torch.utils.data import DataLoader
from .dataset import Dataset
from .models import EdgeModel, InpaintingModel
from .utils import Progbar, create_dir, stitch_images, imsave
from PIL import Image
import matplotlib.pyplot as plt

import cv2
from cv2 import dnn_superres

import torch
import torchvision
import torchvision.transforms as T

class EdgeConnect():
    def __init__(self, config):
        self.config = config

        if config.MODEL == 1:
            model_name = 'edge'
        elif config.MODEL == 2:
            model_name = 'inpaint'
        elif config.MODEL == 3:
            model_name = 'edge_inpaint'
        elif config.MODEL == 4:
            model_name = 'joint'

        self.debug = False
        self.model_name = model_name
        self.edge_model = EdgeModel(config).to(config.DEVICE)
        self.inpaint_model = InpaintingModel(config).to(config.DEVICE)


        # test mode
        self.test_dataset = Dataset(config, config.TEST_FLIST, config.TEST_EDGE_FLIST, augment=False, training=False)

        self.samples_path = os.path.join(config.PATH, 'samples')
        
        self.results_path = os.path.join(config.PATH, 'results')

        if config.RESULTS is not None:
            self.results_path = os.path.join(config.RESULTS)

        if config.DEBUG is not None and config.DEBUG != 0:
            self.debug = True

        self.log_file = os.path.join(config.PATH, 'log_' + model_name + '.dat')

    def load(self):
        if self.config.MODEL == 1:
            self.edge_model.load()

        elif self.config.MODEL == 2:
            self.inpaint_model.load()

        else:
            self.edge_model.load()
            self.inpaint_model.load()

    def save(self):
        if self.config.MODEL == 1:
            self.edge_model.save()

        elif self.config.MODEL == 2 or self.config.MODEL == 3:
            self.inpaint_model.save()

        else:
            self.edge_model.save()
            self.inpaint_model.save()


    def test(self):
        self.edge_model.eval()
        self.inpaint_model.eval()

        model = self.config.MODEL
        create_dir(self.results_path)

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )

        # print("####")
        # print("test dataset")
        # print(str(self.test_dataset.__dict__))
        # print("####")
        
        index = 0

        # UPSCALE #
        # Create an SR object
        sr = dnn_superres.DnnSuperResImpl_create()
        # UPSCALE #

        for items in test_loader:
            name = self.test_dataset.load_name(index)
        
            images, images_gray, edges, masks = self.cuda(*items)
            index += 1

            # edge model
            if model == 1:
                outputs = self.edge_model(images_gray, edges, masks)
                outputs_merged = (outputs * masks) + (edges * (1 - masks))

            # inpaint model
            elif model == 2:
                outputs = self.inpaint_model(images, edges, masks)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

            # inpaint with edge model / joint model
            else:
                edges = self.edge_model(images_gray, edges, masks).detach()
                outputs = self.inpaint_model(images, edges, masks)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

            output = self.postprocess(outputs_merged)[0]

            print("load original image")
            img = Image.open(self.test_dataset.__dict__['data'][0])
            width_og, height_og = img.size

            # CLASSIC UPSCALE #
            # print("reshape output")
            # transform = T.ToPILImage()
            # output_ = transform(output)
            # output_ = np.array(output_.resize((width_og, height_og), Image.LANCZOS))
            # CLASSIC UPSCALE #

            path = os.path.join(self.results_path, name)
            print(index, name)
            imsave(output, path)


            # print('create upscaled image')
            # output_ = Image.open(path)
            # output_ = np.array(output_.resize((width_og, height_og), Image.LANCZOS))
            # outputFinal = Image.fromarray(output_)
            # outputFinal.save(path)

            # UPSCALE #
            # Create an SR object
            # sr = dnn_superres.DnnSuperResImpl_create()
            # Read image
            image_sr = cv2.imread(path)
            # Read the desired model
            # model_sr = "EDSR_x3.pb"
            model_sr = "FSRCNN_x4.pb"
            sr.readModel(model_sr)
            # Set the desired model and scale to get correct pre- and post-processing
            sr.setModel("fsrcnn", 4)
            # Upscale the image
            result_sr = sr.upsample(image_sr)
            # Save the image
            cv2.imwrite(path, result_sr)
            #
            # UPSCALE #

            if self.debug:
                edges = self.postprocess(1 - edges)[0]
                masked = self.postprocess(images * (1 - masks) + masks)[0]
                fname, fext = name.split('.')

                imsave(edges, os.path.join(self.results_path, fname + '_edge.' + fext))
                imsave(masked, os.path.join(self.results_path, fname + '_masked.' + fext))

        print('\nEnd test....')
        return output


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
