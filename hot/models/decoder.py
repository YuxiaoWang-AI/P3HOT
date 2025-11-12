import torch
import torch.nn as nn
import torch.nn.init as init
import matplotlib.pyplot as plt
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels // 2, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Decoder(nn.Module):
    def __init__(self, in_channel=2048, output_channel=18):
        super(Decoder, self).__init__()
        
        self.up1 = Up(in_channel, in_channel // 2, bilinear=False)
        self.up2 = Up(in_channel // 2, in_channel // 4, bilinear=False)
        self.up3 = Up(in_channel // 4, in_channel // 8, bilinear=False)
        # self.up4 = Up(in_channel // 8, in_channel // 16, bilinear=False)
        # self.up4 = Up(in_channel // 8, in_channel // 16, bilinear=False)
        # self.up_conv = OutConv(in_channel // 16, output_channel)
        
        # self.up_conv = nn.Sequential(
        #         nn.Conv2d(2048, 2048, 3, 1, 1),
        #         nn.BatchNorm2d(2048),
        #         nn.ReLU(inplace=True),
        #         # nn.Conv2d(in_channel // 16, in_channel // 32, 3, 1, 1),
        #         # nn.BatchNorm2d(in_channel // 32),
        #         # nn.ReLU(inplace=True),
        #         # nn.Conv2d(in_channel // 32, output_channel, 3, 1, 1),
        #         # nn.BatchNorm2d(output_channel),
        #         # nn.ReLU(inplace=True),
        #         # nn.Conv2d(in_channel // 16, out_channels, kernel_size=1),
        #         # nn.BatchNorm2d(out_channels),
        #         # nn.ReLU(inplace=True),
        # )

        # self.up3_conv = nn.Sequential(
        #         nn.Conv2d(256, 18, 3, 1, 1),
        #         nn.BatchNorm2d(18),
        #         nn.ReLU(inplace=True),
        #         # nn.Conv2d(in_channel // 16, in_channel // 32, 3, 1, 1),
        #         # nn.BatchNorm2d(in_channel // 32),
        #         # nn.ReLU(inplace=True),
        #         # nn.Conv2d(in_channel // 32, output_channel, 3, 1, 1),
        #         # nn.BatchNorm2d(output_channel),
        #         # nn.ReLU(inplace=True),
        #         # nn.Conv2d(in_channel // 16, out_channels, kernel_size=1),
        #         # nn.BatchNorm2d(out_channels),
        #         # nn.ReLU(inplace=True),
        # )
        self.up3_conv = OutConv(in_channel // 8, output_channel)

        # self.up_conv = nn.Sequential(
        #         nn.Conv2d(in_channel // 8, in_channel // 16, 3, 1, 1),
        #         nn.BatchNorm2d(in_channel // 16),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(in_channel // 16, in_channel // 32, 3, 1, 1),
        #         nn.BatchNorm2d(in_channel // 32),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(in_channel // 32, output_channel, 3, 1, 1),
        #         nn.BatchNorm2d(output_channel),
        #         nn.ReLU(inplace=True),
        #         # nn.Conv2d(in_channel // 16, out_channels, kernel_size=1),
        #         # nn.BatchNorm2d(out_channels),
        #         # nn.ReLU(inplace=True),
        # )

        self.conv0 = nn.Sequential(
                        nn.Conv2d(in_channel // 32, in_channel // 16, 1),
                        nn.BatchNorm2d(in_channel // 16),
                        nn.ReLU(inplace=True)
                    )
        # self.up_x = nn.Sequential(
        #                 nn.ConvTranspose2d(in_channel // 16, in_channel // 16, kernel_size=2, stride=2),
        #                 nn.BatchNorm2d(in_channel // 16),
        #                 nn.ReLU(inplace=True)
        #             )
            
        # 定义一个可学习的参数，初始化为随机值
        self.depth_range = nn.Parameter(torch.tensor(0.2), requires_grad=False)
        # self.depth_range = torch.tensor(0.2)

        # self.out_c = nn.Conv2d(output_channel, output_channel, 1)
        # self.out_c = nn.Sequential(
        #         nn.Conv2d(output_channel, output_channel, 3, 1, 1),
        #         nn.BatchNorm2d(output_channel),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(output_channel, output_channel, 3, 1, 1),
        #         # nn.BatchNorm2d(out_channels),
        #         # nn.ReLU(inplace=True),
        #         # nn.Conv2d(in_channel // 16, out_channels, kernel_size=1),
        #         # nn.BatchNorm2d(out_channels),
        #         # nn.ReLU(inplace=True),
        # )
        self.out_c = nn.Conv2d(output_channel, output_channel, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        """
        初始化网络中的权重和偏置。
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # 对于卷积层
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):  # 对于批归一化层
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):  # 对于全连接层（若有）
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x4, x3, x2, x1, x0, logits_per_image, person_mask, total_person, depth):
        # print(self.depth_range)
        b = x4.shape[0]
        # print(x4.shape)
        # exit()
        # print(x4.shape, x3.shape, x2.shape, x1.shape)
        x = self.up1(x4, x3)
        # print(x.shape)
        # print(x.shape, x2.shape)
        x = self.up2(x, x2)
        
        x = self.up3(x, x1)

        # x = self.up1(x, x3)
        # # print(x.shape)
        # # print(x.shape, x2.shape)
        # x = self.up2(x, x2)
        
        # x = self.up3(x, x1)
        # print("up3", x.shape)
        # x = self.up4(x, self.conv0(x0))
        # print(x.shape) # torch.Size([16, 256, 56, 56])
        # exit()

        # 首先根据mask，取出每个人体的depth的平均深度
        # for person_i in range(total_person[])
        person_depth = person_mask * depth.unsqueeze(1)
        person_depth = person_depth.sum(dim=(2,3)) / (person_mask.sum(dim=(2,3)) + 1e-6)
        # print(person_depth)
        # print(person_depth.shape)
        
        batch_mask = torch.zeros((b, depth.shape[1], depth.shape[2])).to(depth.device)
        # print("batch_mask shape", batch_mask.shape)
        for b_item in range(b):

            # 假设你的深度图 d 是一个张量，大小为 (h, w)
            d = depth[b_item]

            # s 是给定的范围数组
            s = person_depth[b_item]
            
            # # a 是需要训练的变量，初始化为某个值
            # a = torch.tensor(0.05, requires_grad=True)

            # 获取 s 中非零的部分
            non_zero_s = s[s != 0]


            # 对非零的部分生成上下界范围矩阵
            lower_bound = torch.clamp(non_zero_s - torch.clamp(self.depth_range, min=0.0, max=0.4), min=0.0)
            upper_bound = torch.clamp(non_zero_s + torch.clamp(self.depth_range, min=0.0, max=0.4), max=1.0)

            # 初始化一个掩码为0的矩阵，大小与深度图 d 一样
            mask = torch.ones_like(d).to(depth.device)

            # 遍历非零的 s 范围，更新掩码
            for lower, upper in zip(lower_bound, upper_bound):
                mask = mask + torch.relu((d - lower) * (upper - d))
                # mask += torch.sigmoid(10 * (d - lower) * (upper - d))
            mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-6)
            # print(non_zero_s)
            # if len(non_zero_s) == 4:
            #     plt.figure()
            #     plt.imshow(mask.cpu().detach())
            #     plt.savefig("save_results/decoder_mask.jpg")
            #     print(mask.shape)
                # exit()
            batch_mask[b_item] = mask
        # print(batch_mask.shape)
        # 使用 interpolate 下采样
        # batch_mask = F.interpolate(batch_mask.unsqueeze(1), size=(7, 7), mode='bilinear', align_corners=False)
        # batch_mask = batch_mask.squeeze(1)
        # 上采样
        # x = self.up_x(x)
        # print(batch_mask.shape)
        # print()
        # exit()
        # 将batch_mask 与 特征图相乘
        # print(x.shape, batch_mask.shape)
        # exit()
        x = x + batch_mask.unsqueeze(1)

        # x = self.up_conv(x)


        x = self.up3_conv(x)

        # 接下来是乘特征图
        # print(person_mask.shape)
        # print(total_person)
        # print("depth shape", depth.shape)
        # exit()

        
            
            # 使用 sigmoid 平滑处理，而不是硬的阈值化
            # thresholded_d = torch.sigmoid(10 * (mask - 0.5))



        # exit()


        # print(person_mask[0,0].max(), person_mask[0,0].min())
        # print(person_mask[0,5].max(), person_mask[0,5].min())
        # exit()

        # x = self.up4(x, x1)
        # print(x.shape)
        # x = self.up_conv(x)
        # print(x.shape)

        # 扩展通道注意力矩阵并进行相乘
        logits_per_image = torch.cat([torch.ones((b, 1)).to(logits_per_image.device)*0.2, logits_per_image], dim=1)
        # print(logits_per_image)
        # print(logits_per_image.shape)
        # exit()
        logits_per_image = logits_per_image.unsqueeze(2).unsqueeze(3)  # 变为 (b, 18, 1, 1)
        output = x * logits_per_image  # 广播相乘
        # print(output.shape)
        output = self.out_c(output)
        # print(output.shape)
        # exit()
        return output
        
        # x1 = self.inc(x)
        # x2 = self.down1(x1)
        # x3 = self.down2(x2)
        # x4 = self.down3(x3)
        # x5 = self.down4(x4)
        # x = self.up1(x5, x4)
        # x = self.up2(x, x3)
        # x = self.up3(x, x2)
        # x = self.up4(x, x1)
        # logits = self.outc(x)
        # return logits