import os
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from keys import alphabetChinese
from utils import load_chars

class BaseDataset(Dataset):
    def __init__(self, root=None, alphabet=alphabetChinese, transform=transforms.ToTensor(), target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.datas = {}
        tmp_labels = os.path.join(root, 'tmp_labels.txt')
        with open(tmp_labels, 'r', encoding='utf-8') as f:
            for c in f.readlines():
                img_index, label = c.strip().split(' ', 1)
                img_name = img_index + '.jpg'
                if not os.path.exists(os.path.join(root, img_name)):
                    continue      
                ignore = {l for l in label if l not in alphabet} 
                if ignore:
                    continue  
                self.datas.update({img_name:label})
        self.image_index = sorted(self.datas.keys())

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        assert index < len(self), 'index range error'
        img_name = self.image_index[index] 
        try:
            img = Image.open(os.path.join(self.root, img_name)).convert('L')
        except:
            print('Corrupted image for %d' % index)
            return self[index + 1]

        if self.transform is not None:
            img = self.transform(img)
        
        label = str(self.datas[img_name])
        
        if self.target_transform is not None:
            label = self.target_transform(label)

        
        return img, label

class NumDataset(BaseDataset):
    def __init__(self, root=None, alphabet=alphabetChinese, mode='train', transform=transforms.ToTensor(), target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.datas = {}
        tmp_labels = os.path.join(root, f'data_{mode}.txt')
        with open(tmp_labels, 'r', encoding='utf-8') as f:
            for c in f.readlines():
                c = c.strip().split(' ')
                img_name = c[0]
                if not os.path.exists(os.path.join(root, img_name)):
                    continue      
                label = ''.join((alphabet[int(num)] for num in c[1:]))
                self.datas.update({img_name:label})
        self.image_index = list(self.datas.keys())

class resizeNormalize(object):

    def __init__(self, imgH, interpolation=Image.BILINEAR):
        self.imgH = imgH
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        scale = img.size[1] * 1.0/ self.imgH
        w = int(img.size[0] / scale) 
        img = img.resize((w, self.imgH), self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img

if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        resizeNormalize(32)])
    # dataset = BaseDataset('/home/hxt/projects/crnn_my/data', transform=transform)
    chars_file = '/home/hxt/projects/crnn_my/chars/char_std_5990.txt'
    alphabet = load_chars(chars_file)
    dataset = NumDataset('/home/hxt/Synth-Chinese/Sythetic_String_Dataset', alphabet=alphabet, mode='train', transform=transform)
    print(len(dataset))
    print(dataset[2])
    # dataloader = DataLoader(dataset=dataset,
    #                         batch_size=len(dataset)//2,
    #                         shuffle=True,
    #                         num_workers=2,
    #                         pin_memory=False)
    # for i, (img, label) in enumerate(dataloader): # img, label 都有batch_size个元素 label为一个tuple
    #     print(i, img.size(), len(label))


