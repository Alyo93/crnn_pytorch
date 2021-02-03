import os
import logging
import torch
from models.crnn import CRNN
from dataset import BaseDataset, resizeNormalize, NumDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from config import parse_args
import utils
import time
from keys import alphabetChinese
from warpctc_pytorch import CTCLoss
args = parse_args()

class Trainer(object):
    def __init__(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
        if args.chars_file == '':
            self.alphabet = alphabetChinese
        else:
            self.alphabet = utils.load_chars(args.chars_file)
        nclass = len(self.alphabet) + 1
        nc = 1
        self.net = CRNN(args.imgH, nc, args.nh, nclass)
        self.train_dataloader, self.val_dataloader = self.dataloader(self.alphabet)
        self.criterion = CTCLoss()
        self.optimizer = self.get_optimizer()
        self.converter = utils.strLabelConverter(self.alphabet, ignore_case=False)
        self.best_acc = 0.00001

        model_name = '%s'%(args.dataset_name)
        if not os.path.exists(args.save_prefix):
            os.mkdir(args.save_prefix)
        args.save_prefix += model_name

        if args.pretrained != '':
            print('loading pretrained model from %s' % args.pretrained)
            checkpoint = torch.load(args.pretrained)

            if 'model_state_dict' in checkpoint.keys():
                # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                args.start_epoch = checkpoint['epoch']
                self.best_acc = checkpoint['best_acc']
                checkpoint = checkpoint['model_state_dict']

            from collections import OrderedDict
            model_dict = OrderedDict()
            for k, v in checkpoint.items():
                if 'module' in k:
                    model_dict[k[7:]] = v
                else:
                    model_dict[k] = v
            self.net.load_state_dict(model_dict)

        if not args.cuda and torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        
        elif args.cuda and torch.cuda.is_available():        
            print('available gpus is ', torch.cuda.device_count())
            self.net = torch.nn.DataParallel(self.net, output_dim=1).cuda()
            self.criterion = self.criterion.cuda()

    def dataloader(self, alphabet):
        # train_transform = transforms.Compose(
        #     [transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        #     resizeNormalize(args.imgH)])
        # train_dataset = BaseDataset(args.train_dir, alphabet, transform=train_transform)
        train_dataset = NumDataset(args.train_dir, alphabet, transform=resizeNormalize(args.imgH))
        train_dataloader = DataLoader(dataset=train_dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=args.num_workers,
                            pin_memory=True)

        if os.path.exists(args.val_dir):
            # val_dataset = BaseDataset(args.val_dir, alphabet, transform=resizeNormalize(args.imgH)) 
            val_dataset = NumDataset(args.val_dir, alphabet, mode='test', transform=resizeNormalize(args.imgH))
            val_dataloader = DataLoader(dataset=val_dataset, 
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=args.num_workers,
                                pin_memory=True)
        else:
            val_dataloader = None

        return train_dataloader, val_dataloader

    def get_optimizer(self):
        if args.optimizer == 'sgd':
            optimizer = optim.SGD(
                self.net.parameters(),
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.wd, 
            )
        elif args.optimizer == 'adam':
            optimizer = optim.Adam(
                self.net.parameters(), 
                lr=args.lr,
                betas=(args.beta1, 0.999),
            )
        else:
            optimizer = optim.RMSprop(
                self.net.parameters(),
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.wd,
            )
        return optimizer

    def train(self):
        logging.basicConfig()
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        log_file_path = args.save_prefix + '_train.log'
        log_dir = os.path.dirname(log_file_path)
        if log_dir and not os.path.exists(log_dir):
            os.mkdir(log_dir)
        fh = logging.FileHandler(log_file_path)
        logger.addHandler(fh)
        logger.info(args)
        logger.info('Start training from [Epoch {}]'.format(args.start_epoch + 1))
                
        losses = utils.Averager()
        train_accuracy = utils.Averager()
        
        for epoch in range(args.start_epoch, args.nepoch):
            self.net.train()
            btic = time.time()
            for i, (imgs, labels) in enumerate(self.train_dataloader):
                batch_size = imgs.size()[0]
                imgs = imgs.cuda()
                preds = self.net(imgs).cpu()
                text, length = self.converter.encode(labels) # length  一个batch各个样本的字符长度, text 一个batch中所有中文字符所对应的下标
                preds_size = torch.IntTensor([preds.size(0)] * batch_size) 
                loss_avg = self.criterion(preds, text, preds_size, length) / batch_size

                self.optimizer.zero_grad()
                loss_avg.backward()
                self.optimizer.step()

                losses.update(loss_avg.item(), batch_size)

                _, preds_m = preds.max(2)
                preds_m = preds_m.transpose(1, 0).contiguous().view(-1)
                sim_preds = self.converter.decode(preds_m.data, preds_size.data, raw=False)
                n_correct = 0
                for pred, target in zip(sim_preds, labels):
                    if pred == target:
                        n_correct += 1
                train_accuracy.update(n_correct, batch_size, MUL_n=False)

                if args.log_interval and not (i + 1) % args.log_interval:
                    logger.info('[Epoch {}/{}][Batch {}/{}], Speed: {:.3f} samples/sec, Loss:{:.3f}'.format(
                        epoch+1, args.nepoch, i+1, len(self.train_dataloader), batch_size/(time.time()-btic), losses.val()))
                    losses.reset()

            logger.info('Training accuracy: {:.3f}, [#correct:{} / #total:{}]'.format(
                train_accuracy.val(), train_accuracy.sum, train_accuracy.count))
            train_accuracy.reset()

            if args.val_interval and not (epoch+1) % args.val_interval:
                acc = self.validate(logger)
                if acc > self.best_acc:
                    self.best_acc = acc
                    save_path = '{:s}_best.pth'.format(args.save_prefix)
                    torch.save({
                                'epoch': epoch,
                                'model_state_dict': self.net.state_dict(),
                                # 'optimizer_state_dict': self.optimizer.state_dict(),
                                'best_acc': self.best_acc,
                                }, save_path)
                logging.info("best acc is:{:.3f}".format(self.best_acc))
                if args.save_interval and not (epoch+1) % args.save_interval:
                    save_path = '{:s}_{:04d}_{:.3f}.pth'.format(args.save_prefix, epoch+1, acc)
                    torch.save({
                                'epoch': epoch,
                                'model_state_dict': self.net.state_dict(),
                                # 'optimizer_state_dict': self.optimizer.state_dict(),
                                'best_acc': self.best_acc,
                                }, save_path)    
                    


    def validate(self, logger):
        if self.val_dataloader is None:
            return 0
        logger.info('Start validate.')
        losses = utils.Averager()
        self.net.eval()
        n_correct = 0
        with torch.no_grad(): 
            for i, (imgs, labels) in enumerate(self.val_dataloader):
                batch_size = imgs.size()[0]
                imgs = imgs.cuda()
                preds = self.net(imgs).cpu()
                text, length = self.converter.encode(labels) # length  一个batch各个样本的字符长度, text 一个batch中所有中文字符所对应的下标
                preds_size = torch.IntTensor([preds.size(0)] * batch_size) # timestep * batchsize
                loss_avg = self.criterion(preds, text, preds_size, length) / batch_size

                losses.update(loss_avg.item(), batch_size)

                _, preds = preds.max(2)
                preds = preds.transpose(1, 0).contiguous().view(-1)
                sim_preds = self.converter.decode(preds.data, preds_size.data, raw=False)
                for pred, target in zip(sim_preds, labels):
                    if pred == target:
                        n_correct += 1

        accuracy = n_correct / float(losses.count)
        
        logger.info('Evaling loss: {:.3f}, accuracy: {:.3f}, [#correct:{} / #total:{}]'.format(
                    losses.val(), accuracy, n_correct, losses.count))

        return accuracy
                        
if __name__ == '__main__':
    # args.dataset_name = 'synth'
    # args.batch_size = 100
    # args.nepoch = 50
    # args.lr = 0.001
    # args.train_dir = '/home/hxt/dataset/synth_data/train_200w'
    # args.val_dir = '/home/hxt/dataset/synth_data/test_2w'
    # args.gpus = '1' 
    # args.pretrained = '/home/hxt/projects/certificate_v1/inference/torch_crnn/ocr-lstm.pth'
    # args.chars_file = '/home/hxt/projects/crnn_my/chars/char_std_5990.txt'
    # args.num_workers = 4 
    trainer = Trainer()
    trainer.train()
            





