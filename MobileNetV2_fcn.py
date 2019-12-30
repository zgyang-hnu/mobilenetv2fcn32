import torch.nn as nn
import math

from NDNet_ende import *

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

        

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, n_class=19, width_mult=0.25, criterion=None):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 512
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [4, 24, 2, 2],
            [4, 32, 3, 2],
            [4, 64, 6, 2],
            [4, 96, 4, 1],
            [4, 160, 8, 2],
            [4, 320, 8, 1],
        ]
        self.criterion = criterion
        # building first layer
     
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv2d(self.last_channel, n_class, kernel_size=1, stride=1,
                     padding=0, dilation=1, bias=False)
        )

        self._initialize_weights()

    def forward(self, x, label=None, step=None):
        x = self.features(x)
        
        x = self.classifier(x)

        x = F.interpolate(x, scale_factor=32,
                                   mode='bilinear',
                                   align_corners=True)
        if self.criterion:
            
            
            #pred=pred.view()
            
            loss = self.criterion(x, label)
            if step % 10000==0:
               #print('........size of feature maps........',x.shape)
               pred=F.softmax(x, dim=1)
               print('........size of score maps...........',x.shape)
               im=torch.max(x[0].permute(1,2,0),dim=2)[1].cpu().detach().numpy()
               print(im.shape)
               im=Image.fromarray(decode_labels_cityscape(trainid2labelid_efficient(np.uint8(im))))
               #im=Image.fromarray(decode_labels_camvid(np.uint8(im)))
               #im.show()
               im.save('./mid_train_results/'+str(step)+'_pred'+'.jpg')
               im=label[0].cpu().numpy()
               #print(im.shape)
               im=Image.fromarray(decode_labels_cityscape(trainid2labelid_efficient(np.uint8(im))))
               #im=Image.fromarray(decode_labels_camvid(np.uint8(im)))
               #im.show()
               im.save('./mid_train_results/'+str(step)+'_gt'+'.jpg')
            #print(loss)
            #print('....lossshape.....',loss.shape)
            return loss

        return F.log_softmax(x, dim=1)
      

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
