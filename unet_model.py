""" Full assembly of the parts to form the complete network """

from unet_parts import *



class UNet(nn.Module):
    def __init__(self,
                 n=64,
                 bilinear=True):
        super(UNet, self).__init__()
        self.bilinear = bilinear

        self.inc = (DoubleConv(1, n))
        self.down1 = (Down(n, n*2))
        self.down2 = (Down(n*2, n*4))
        self.down3 = (Down(n*4, n*8))
        factor = 2 if bilinear else 1
        self.down4 = (Down(n*8, n*16 // factor))
        self.up1 = (Up(n*16, n*8 // factor, bilinear))
        self.up2 = (Up(n*8, n*4 // factor, bilinear))
        self.up3 = (Up(n*4, n*2 // factor, bilinear))
        self.up4 = (Up(n*2, n, bilinear))
        self.outc = (OutConv(n, 1))

    def forward(self, x, bbox = None, restore: bool = False):
        if bbox is not None:
            mask = (x > -17)
            x_size_original = x.size()
            x = x[:, :, bbox[0]:bbox[2], bbox[1]:bbox[3]]
        x_size = x.size()
        batch_size = x_size[0]
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        x = x.view(batch_size,-1)
        x = F.log_softmax(x,dim=1)
        if restore:
            min_values, _ = torch.min(x, dim=1,keepdim=True)
            min_values = min_values.unsqueeze(2)
            x = x.view(x_size)
            if bbox is not None:
                x_res = min_values.repeat(1,x_size_original[2],x_size_original[3])
                x_res = x_res.unsqueeze(1)
                x_res[mask] = 0
                x_res[:, :, bbox[0]:bbox[2], bbox[1]:bbox[3]] = x
                return x_res
        return x

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
















