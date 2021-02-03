import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from PIL import Image
from models.crnn import CRNN
from keys import alphabetChinese
from dataset import resizeNormalize
import utils 
import argparse
import time

def str2bool(v):
    return v.lower() in ('true', 't', '1')

parser = argparse.ArgumentParser()
parser.add_argument('--imgH', type=int, default=32,
                    help='the height of the input image to network')
parser.add_argument('--nh', type=int, default=256,
                    help='size of the lstm hidden state')
parser.add_argument('--cuda', type=str2bool, default=True,
                    help='enables cuda')
parser.add_argument('--gpus', type=str, default='0', 
                    help='Testing with GPUs, you can specify 1,3 for example')
parser.add_argument('--batch_size', type=int, default=64, 
                    help='batch size for predict batch')
parser.add_argument('--model-path', type=str, help='model file path')
parser.add_argument('--image-path', type=str, help='image path')

img_types = ['jpg', 'png', 'jpeg', 'bmp']

class Demo(object):
    def __init__(self, args):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
        self.args = args
        self.alphabet = alphabetChinese
        nclass = len(self.alphabet) + 1
        nc = 1
        self.net = CRNN(args.imgH, nc, args.nh, nclass)
        self.converter = utils.strLabelConverter(self.alphabet, ignore_case=False)
        self.transformer = resizeNormalize(args.imgH)

        print('loading pretrained model from %s' % args.model_path)
        checkpoint = torch.load(args.model_path)
        if 'model_state_dict' in checkpoint.keys():
            checkpoint = checkpoint['model_state_dict']
        from collections import OrderedDict
        model_dict = OrderedDict()
        for k, v in checkpoint.items():
            if 'module' in k:
                model_dict[k[7:]] = v
            else:
                model_dict[k] = v
        self.net.load_state_dict(model_dict)

        if args.cuda and torch.cuda.is_available():
            print('available gpus is,', torch.cuda.device_count())
            self.net = torch.nn.DataParallel(self.net, output_dim=1).cuda()
        
        self.net.eval()
    
    def predict(self, image):
        image = self.transformer(image)
        if torch.cuda.is_available():
            image = image.cuda()
        image = image.view(1, *image.size())
        image = Variable(image)

        preds = self.net(image)
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        preds_size = Variable(torch.IntTensor([preds.size(0)]))
        raw_pred = self.converter.decode(preds.data, preds_size.data, raw=True)
        sim_pred = self.converter.decode(preds.data, preds_size.data, raw=False)
        print('%-20s => %-20s' % (raw_pred, sim_pred))

        return sim_pred

    def predict_batch(self, images):
        N = len(images)
        n_batch = N // self.args.batch_size
        n_batch += 1 if N % self.args.batch_size else 0
        res = []
        for i in range(n_batch):
            batch = images[i*self.args.batch_size : min((i+1)*self.args.batch_size, N)]
            maxW = 0
            for i in range(len(batch)):
                batch[i] = self.transformer(batch[i])
                imgW = batch[i].shape[2]
                maxW = max(maxW, imgW)
            
            for i in range(len(batch)):
                if batch[i].shape[2] < maxW:
                    batch[i] = torch.cat((batch[i], torch.zeros((1, self.args.imgH, maxW-batch[i].shape[2]), dtype=batch[i].dtype)), 2) 
            batch_imgs = torch.cat([t.unsqueeze(0) for t in batch], 0)
            preds = self.net(batch_imgs)
            preds_size = Variable(torch.IntTensor([preds.size(0)]*len(batch)))
            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            raw_preds = self.converter.decode(preds.data, preds_size.data, raw=True)
            sim_preds = self.converter.decode(preds.data, preds_size.data, raw=False)
            for raw_pred, sim_pred in zip(raw_preds, sim_preds):
                print('%-20s => %-20s' % (raw_pred, sim_pred))
            res.extend(sim_preds)
        return res

    def inference(self, image_path, batch_pred=False):
        if os.path.isdir(image_path):
            file_list = os.listdir(image_path)
            image_list = [os.path.join(image_path, i) for i in file_list if i.rsplit('.')[-1].lower() in img_types] 
        else:
            image_list = [image_path]
        
        res = []
        images = []
        for img_path in image_list:
            image = Image.open(img_path).convert('L')
            if not batch_pred:
                sim_pred = self.predict(image)
                res.append(sim_pred)
            else:
                images.append(image)
        if batch_pred and images:
            res = self.predict_batch(images)
        return res


if __name__ == '__main__':
    args = parser.parse_args()
    args.image_path = '/home/hxt/dataset/synth_data/test_2w/00016255.jpg'
    args.model_path = '/home/hxt/projects/crnn_my/checkpoint/synth-crnn_best.pth'
    demo = Demo(args)
    start = time.time()
    demo.inference(args.image_path)
    print(time.time()-start)

