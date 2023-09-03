import torch.nn as nn


class ResBlockEnc(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(ResBlockEnc, self).__init__()
        stride = (stride, stride)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=stride, padding=(1,1)),
            nn.BatchNorm2d(num_features=out_channels), nn.LeakyReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1,1)),
            nn.BatchNorm2d(num_features=out_channels), nn.LeakyReLU()
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), stride=stride)
        )

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


class ResBlockDec(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(ResBlockDec, self).__init__()
        assert stride > 1
        stride = (stride, stride)
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=stride, output_padding=(1,1)),
            nn.BatchNorm2d(num_features=out_channels), nn.LeakyReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(num_features=out_channels), nn.LeakyReLU()
        )

        self.shortcut = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), stride=stride, output_padding=(1,1))
        )

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, pad=0, bn=True):
        super(ConvLayer, self).__init__()
        if bn:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=(kernel, kernel), stride=(stride, stride), padding=(pad, pad)),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU()
            )
        else:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=(kernel, kernel), stride=(stride, stride), padding=(pad, pad)),
                nn.LeakyReLU()
            )

    def forward(self, x):
        return self.layer(x)


class ConvTransLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, pad=0, out_pad=0, bn=True):
        super(ConvTransLayer, self).__init__()
        if bn:
            self.layer = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel, kernel),
                                   stride=(stride, stride), padding=(pad, pad), output_padding=(out_pad, out_pad)),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU()
            )
        else:
            self.layer = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel, kernel),
                                   stride=(stride, stride), padding=(pad, pad), output_padding=(out_pad, out_pad)),
                nn.LeakyReLU()
            )

    def forward(self, x):
        return self.layer(x)
