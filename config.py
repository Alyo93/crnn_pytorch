import argparse

def parse_args():
    def str2bool(v):
        return v.lower() in ('true', 't', '1')

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=str2bool, default=True,
                        help='enables cuda')
    parser.add_argument('--gpus', type=str, default='0', 
                        help='Training with GPUs, you can specify 1,3 for example')
    parser.add_argument('--imgH', type=int, default=32,
                        help='the height of the input image to network')
    parser.add_argument('--nh', type=int, default=256,
                        help='size of the lstm hidden state')
    parser.add_argument('--chars_file', type=str, default='')
    parser.add_argument('--log_interval', type=int, default=1,
                        help='Logging mini-batch interval. Default is 20')
    parser.add_argument('--val_interval', type=int, default=1,
                        help='Epoch interval for validation, increase the number will reduce the '
                             'training time if validation is slow.')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Saving parameters epoch interval, best model will always be saved.')
    parser.add_argument('--save_prefix', type=str, default='./checkpoint/',
                        help='Saving parameter prefix')
    parser.add_argument('--dataset_name', type=str,
                        help='the name of training dataset')
    parser.add_argument('--pretrained', type=str, default='',
                        help='path to pretrained model (to continue training)')
    parser.add_argument('--train_dir', type=str)
    parser.add_argument('--val_dir', type=str)

    parser.add_argument('--batch_size', type=int, default=8,
                        help='training mini-batch size')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='number of data loading workers')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='Starting epoch for resuming, default is 0 for new training')
    parser.add_argument('--nepoch', type=int, default=100,
                        help='number of epochs to train for')
    parser.add_argument('--optimizer',type=str, default='rmsprop')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='beta1 for adam. default=0.5')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum, default=0.9')
    parser.add_argument('--wd', type=float, default=0.0001,
                        help='weight decay, default=0.0001')

    args = parser.parse_args()

    return args
