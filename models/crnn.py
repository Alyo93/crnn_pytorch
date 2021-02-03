import torch
import torch.nn as nn

class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True) # T, b, nHidden*2
        self.embedding = nn.Linear(nHidden * 2, nOut) # T*b, nOut
    
    def forward(self, input):
        recurrent, _ = self.rnn(input)
        # print(recurrent.size())
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T*b, h)

        output = self.embedding(t_rec) # [T*b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):

    def __init__(self, imgH, nc, nh, nclass, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ss = [1, 1, 1, 1, 1, 1, 1]
        ps = [1, 1, 1, 1, 1, 1, 0]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i-1]
            nOut = nm[i]
            cnn.add_module("conv{0}".format(i),
                            nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module("batchnorm{0}".format(i), nn.BatchNorm2d(nOut))    
            if leakyRelu:
                cnn.add_module("relu{0}".format(i), nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module("relu{0}".format(i), nn.ReLU(inplace=True))
        
        convRelu(0)
        cnn.add_module("pooling{0}".format(0), nn.MaxPool2d(2, 2)) # 64*16*50
        convRelu(1)
        cnn.add_module("pooling{0}".format(1), nn.MaxPool2d(2, 2)) # 128*8*25
        convRelu(2, True)
        convRelu(3)
        cnn.add_module("pooling{0}".format(3), nn.MaxPool2d((2, 2), (2, 1), (0, 1))) # 256*4*26
        convRelu(4, True)
        convRelu(5)
        cnn.add_module("pooling{0}".format(5), nn.MaxPool2d((2, 2), (2, 1), (0, 1))) #512*2*27
        convRelu(6, True) #512*1*26

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))
        
        self.apply(weights_init)
        
    def forward(self, input):
        conv = self.cnn(input)
        # print(conv.size())
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2) # .squeeze()用于减少某个维度 .unsqueeze用于增加维度， 这里变为 b, c, w
        conv = conv.permute(2, 0, 1)  # [w, b, c] 维度转换 w = img.w/4 c=512
        output = self.rnn(conv) # [w, b, nclass]
        # print(output.size())

        return output


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


if __name__ == "__main__":
    x = torch.rand((4, 1, 32, 256))
    crnn = CRNN(32,1,256,5530)
    print(crnn) # 具体的网络结构
    print("Model's state_dict:")
    # crnn.state_dict() 是一个字典， 通过for访问字典获得其key值 即param_tensor 类似模型参数的名称
    # 通过 crnn.state_dict()[param_tensor]可以访问具体数值 相当于value值
    print(len(crnn.state_dict()))
    for param_tensor in crnn.state_dict(): 
        print(param_tensor, "\t", crnn.state_dict()[param_tensor].size())
        # print(crnn.state_dict()[param_tensor])
        # break
    
    print(len(list(crnn.parameters())))
    for parameter in crnn.parameters():
        print(parameter.size()) 
        print(parameter.requires_grad)
        # print(parameter.data)
        break

    print(crnn.__class__.__name__)
    # 发现paramerter和param_tensor并不是完全一样 
    # 例如这里parameter中缺少了param_tensor中 batchnorm层的running_mean、running_var、num_batches_tracked 参数
    # paramter打印时会显示数值及requires_grad=True 而打印crnn.state_dict()[param_tensor]时只有数值 == 相当于 parameter.data
    y = crnn(x)
    print(y.grad_fn)
    print(y.grad_fn.next_functions[0][0])
    crnn.zero_grad()
    print(crnn.cnn.conv1.weight.grad)