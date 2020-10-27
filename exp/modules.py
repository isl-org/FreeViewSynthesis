import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import itertools
import sys

sys.path.append("../")
import config
import co
import ext


class VGGPerceptualLoss(nn.Module):
    def __init__(self, inp_scale="-11"):
        super().__init__()
        self.inp_scale = inp_scale
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.vgg = torchvision.models.vgg19(pretrained=True).features

    def forward(self, es, ta):
        self.vgg = self.vgg.to(es.device)
        self.mean = self.mean.to(es.device)
        self.std = self.std.to(es.device)

        if self.inp_scale == "-11":
            es = (es + 1) / 2
            ta = (ta + 1) / 2
        elif self.inp_scale != "01":
            raise Exception("invalid input scale")
        es = (es - self.mean) / self.std
        ta = (ta - self.mean) / self.std

        loss = [torch.abs(es - ta).mean()]
        for midx, mod in enumerate(self.vgg):
            es = mod(es)
            with torch.no_grad():
                ta = mod(ta)

            if midx == 3:
                lam = 1
                loss.append(torch.abs(es - ta).mean() * lam)
            elif midx == 8:
                lam = 0.75
                loss.append(torch.abs(es - ta).mean() * lam)
            elif midx == 13:
                lam = 0.5
                loss.append(torch.abs(es - ta).mean() * lam)
            elif midx == 22:
                lam = 0.5
                loss.append(torch.abs(es - ta).mean() * lam)
            elif midx == 31:
                lam = 1
                loss.append(torch.abs(es - ta).mean() * lam)
                break
        return loss


class LPIPS(nn.Module):
    def __init__(self):
        super().__init__()
        self.mod = None
        self.clip = True

    def forward(self, es, ta):
        if self.mod is None:
            sys.path.append(str(config.lpips_root))
            import PerceptualSimilarity.models as ps

            self.mod = ps.PerceptualLoss()

        if self.clip:
            es = torch.clamp(es, -1, 1)
        out = self.mod(es, ta, normalize=False)

        return out.mean()


class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        if len(args) == 1:
            return args[0]
        else:
            return args


class ConvGRU2d(nn.Module):
    def __init__(
        self,
        channels_x,
        channels_out,
        kernel_size=3,
        padding=1,
        nonlinearity="tanh",
        bias=True,
    ):
        super().__init__()
        self.channels_x = channels_x
        self.channels_out = channels_out

        self.conv_gates = nn.Conv2d(
            in_channels=channels_x + channels_out,
            out_channels=2 * channels_out,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )
        self.conv_can = nn.Conv2d(
            in_channels=channels_x + channels_out,
            out_channels=channels_out,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

        if nonlinearity == "tanh":
            self.nonlinearity = nn.Tanh()
        elif nonlinearity == "relu":
            self.nonlinearity = nn.ReLU()
        else:
            raise Exception("invalid nonlinearity")

    def forward(self, x, h=None):
        if h is None:
            h = torch.zeros(
                (x.shape[0], self.channels_out, x.shape[2], x.shape[3]),
                dtype=x.dtype,
                device=x.device,
            )
        combined = torch.cat([x, h], dim=1)
        combined_conv = torch.sigmoid(self.conv_gates(combined))
        del combined
        r = combined_conv[:, : self.channels_out]
        z = combined_conv[:, self.channels_out :]

        combined = torch.cat([x, r * h], dim=1)
        n = self.nonlinearity(self.conv_can(combined))
        del combined

        h = z * h + (1 - z) * n
        return h


class VGGUNet(nn.Module):
    def __init__(
        self, net="vgg16", pool="average", n_encoder_stages=3, n_decoder_convs=2
    ):
        super().__init__()

        if net == "vgg16":
            vgg = torchvision.models.vgg16(pretrained=True).features
        elif net == "vgg19":
            vgg = torchvision.models.vgg19(pretrained=True).features
        else:
            raise Exception("invalid vgg net")

        encs = []
        enc = []
        encs_channels = []
        channels = -1
        for mod in vgg:
            if isinstance(mod, nn.Conv2d):
                channels = mod.out_channels

            if isinstance(mod, nn.MaxPool2d):
                encs.append(nn.Sequential(*enc))
                encs_channels.append(channels)
                n_encoder_stages -= 1
                if n_encoder_stages <= 0:
                    break
                if pool == "average":
                    enc = [
                        nn.AvgPool2d(
                            kernel_size=2, stride=2, padding=0, ceil_mode=False
                        )
                    ]
                elif pool == "max":
                    enc = [
                        nn.MaxPool2d(
                            kernel_size=2, stride=2, padding=0, ceil_mode=False
                        )
                    ]
                else:
                    raise Exception("invalid pool")
            else:
                enc.append(mod)
        self.encs = nn.ModuleList(encs)

        cin = encs_channels[-1] + encs_channels[-2]
        decs = []
        for idx, cout in enumerate(reversed(encs_channels[:-1])):
            decs.append(self._dec(cin, cout, n_convs=n_decoder_convs))
            cin = cout + encs_channels[max(-idx - 3, -len(encs_channels))]
        self.decs = nn.ModuleList(decs)

    def _dec(self, channels_in, channels_out, n_convs=2):
        mods = []
        for _ in range(n_convs):
            mods.append(
                nn.Conv2d(
                    channels_in,
                    channels_out,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                )
            )
            mods.append(nn.ReLU())
            channels_in = channels_out
        return nn.Sequential(*mods)

    # def train(self, mode=True):
    #     super().train(mode=mode)
    #     if not mode:
    #         return
    #     for mod in self.encs.modules():
    #         if isinstance(mod, nn.BatchNorm2d):
    #             mod.eval()
    #             for param in mod.parameters():
    #                 param.requires_grad_(False)

    def forward(self, x):
        feats = []
        for enc in self.encs:
            x = enc(x)
            feats.append(x)

        for dec in self.decs:
            x0 = feats.pop()
            x1 = feats.pop()
            x0 = F.interpolate(
                x0, size=(x1.shape[2], x1.shape[3]), mode="nearest"
            )
            x = torch.cat((x0, x1), dim=1)
            x = dec(x)
            feats.append(x)

        x = feats.pop()
        return x


class UNet(nn.Module):
    def __init__(
        self,
        in_channels,
        enc_channels=[64, 128, 256],
        dec_channels=[128, 64],
        out_channels=3,
        n_enc_convs=2,
        n_dec_convs=2,
    ):
        super().__init__()

        self.encs = nn.ModuleList()
        self.enc_translates = nn.ModuleList()
        pool = False
        for enc_channel in enc_channels:
            stage = self.create_stage(
                in_channels, enc_channel, n_enc_convs, pool
            )
            self.encs.append(stage)
            translate = nn.Conv2d(enc_channel, enc_channel, kernel_size=1)
            self.enc_translates.append(translate)
            in_channels, pool = enc_channel, True

        self.decs = nn.ModuleList()
        for idx, dec_channel in enumerate(dec_channels):
            in_channels = enc_channels[-idx - 1] + enc_channels[-idx - 2]
            stage = self.create_stage(
                in_channels, dec_channel, n_dec_convs, False
            )
            self.decs.append(stage)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        if out_channels <= 0:
            self.out_conv = None
        else:
            self.out_conv = nn.Conv2d(
                dec_channels[-1], out_channels, kernel_size=1, padding=0
            )

    def convrelu(self, in_channels, out_channels, kernel_size=3, padding=None):
        if padding is None:
            padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
        )

    def create_stage(self, in_channels, out_channels, n_convs, pool):
        mods = []
        if pool:
            mods.append(nn.AvgPool2d(kernel_size=2))
        for _ in range(n_convs):
            mods.append(self.convrelu(in_channels, out_channels))
            in_channels = out_channels
        return nn.Sequential(*mods)

    def forward(self, x):
        outs = []
        for enc, enc_translates in zip(self.encs, self.enc_translates):
            x = enc(x)
            outs.append(enc_translates(x))

        for dec in self.decs:
            x0, x1 = outs.pop(), outs.pop()
            x = torch.cat((self.upsample(x0), x1), dim=1)
            x = dec(x)
            outs.append(x)

        x = outs.pop()
        if self.out_conv:
            x = self.out_conv(x)
        return x


class GRUUNet(nn.Module):
    def __init__(
        self,
        channels_in,
        enc_channels=[32, 64, 64],
        dec_channels=[64, 32],
        n_enc_convs=2,
        n_dec_convs=2,
        gru_all=False,
        gru_nonlinearity="relu",
        bias=False,
    ):
        super().__init__()
        self.n_rnn = 0
        self.gru_nonlinearity = gru_nonlinearity

        stride = 1
        cin = channels_in
        encs = []
        for cout in enc_channels:
            encs.append(
                self._enc(
                    cin,
                    cout,
                    stride=stride,
                    n_convs=n_enc_convs,
                    gru_all=gru_all,
                )
            )
            stride = 2
            cin = cout
        self.encs = nn.ModuleList(encs)

        cin = enc_channels[-1] + enc_channels[-2]
        decs = []
        for idx, cout in enumerate(dec_channels):
            decs.append(
                self._dec(cin, cout, n_convs=n_dec_convs, gru_all=gru_all)
            )
            cin = cout + enc_channels[max(-idx - 3, -len(enc_channels))]
        self.decs = nn.ModuleList(decs)

    def _enc(
        self, channels_in, channels_out, stride=2, n_convs=2, gru_all=False
    ):
        mods = []
        if stride > 1:
            mods.append(nn.AvgPool2d(kernel_size=2))
        for idx in range(n_convs):
            if gru_all or idx == n_convs - 1:
                self.n_rnn += 1
                mods.append(
                    ConvGRU2d(
                        channels_in,
                        channels_out,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                        nonlinearity=self.gru_nonlinearity,
                    )
                )
            else:
                mods.append(
                    nn.Conv2d(
                        channels_in,
                        channels_out,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    )
                )
                mods.append(nn.ReLU())
            channels_in = channels_out
            stride = 1
        return nn.Sequential(*mods)

    def _dec(self, channels_in, channels_out, n_convs=2, gru_all=False):
        mods = []
        for idx in range(n_convs):
            if gru_all or idx == n_convs - 1:
                self.n_rnn += 1
                mods.append(
                    ConvGRU2d(
                        channels_in,
                        channels_out,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                        nonlinearity=self.gru_nonlinearity,
                    )
                )
            else:
                mods.append(
                    nn.Conv2d(
                        channels_in,
                        channels_out,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    )
                )
                mods.append(nn.ReLU())
            channels_in = channels_out
        return nn.Sequential(*mods)

    def forward(self, x, hs=None):
        if hs is None:
            hs = [None for _ in range(self.n_rnn)]

        hidx = 0
        feats = []
        for enc in self.encs:
            for mod in enc:
                if isinstance(mod, ConvGRU2d):
                    x = mod(x, hs[hidx])
                    hs[hidx] = x
                    hidx += 1
                else:
                    x = mod(x)
            feats.append(x)

        for dec in self.decs:
            x0 = feats.pop()
            x1 = feats.pop()
            x0 = F.interpolate(
                x0, size=(x1.shape[2], x1.shape[3]), mode="nearest"
            )
            x = torch.cat((x0, x1), dim=1)
            del x0, x1
            for mod in dec:
                if isinstance(mod, ConvGRU2d):
                    x = mod(x, hs[hidx])
                    hs[hidx] = x
                    hidx += 1
                else:
                    x = mod(x)
            feats.append(x)

        x = feats.pop()
        return x, hs


class RNNWrapper(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, hs=None):
        return self.net(x), hs


class FixedNet(nn.Module):
    def __init__(self, enc_net, dec_net):
        super().__init__()
        self.enc_net = enc_net
        self.dec_net = dec_net

    def forward(self, **kwargs):
        x = kwargs["srcs"]
        sampling_maps = kwargs["sampling_maps"]
        valid_depth_masks = kwargs["valid_depth_masks"]
        valid_map_masks = kwargs["valid_map_masks"]

        bs, nv = x.shape[:2]

        x = x.view(bs * nv, *x.shape[2:])
        x = self.enc_net(x)

        x = F.grid_sample(
            x,
            sampling_maps.view(bs * nv, *sampling_maps.shape[2:]),
            mode="bilinear",
            padding_mode="zeros",
        )
        x = x.view(bs, nv, *x.shape[1:])

        x = torch.cat([x, valid_depth_masks, valid_map_masks], dim=2)

        x = x.view(bs, nv * x.shape[2], *x.shape[3:])
        x = self.dec_net(x)
        return {"out": x}


def get_fixed_net(enc_net, dec_net, n_views):
    if enc_net == "identity":
        enc_net = Identity()
        enc_channels = 3
    elif enc_net == "vgg16unet3":
        enc_net = VGGUNet(net="vgg16", n_encoder_stages=3)
        enc_channels = 64
    else:
        raise Exception("invalid enc_net")

    enc_channels = n_views * (enc_channels + 2)

    if dec_net == "unet4.64.3":
        dec_net = UNet(
            enc_channels,
            enc_channels=[64, 128, 256, 512],
            dec_channels=[256, 128, 64],
            out_channels=3,
            n_enc_convs=3,
            n_dec_convs=3,
        )
    else:
        raise Exception("invalid dec_net")

    return FixedNet(enc_net, dec_net)


class MappingRNN(nn.Module):
    def __init__(
        self,
        enc_net,
        merge_net,
        merge_channels,
        mode="single",
        cat_masks=True,
        cat_global_avg=False,
        cat_global_max=False,
    ):
        super().__init__()
        self.enc_net = enc_net
        self.merge_net = merge_net
        self.mode = mode
        self.cat_masks = cat_masks
        self.cat_global_avg = cat_global_avg
        self.cat_global_max = cat_global_max

        if mode in ["single", "softmax"]:
            self.rgb_conv = nn.Conv2d(
                merge_channels,
                3,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
        if mode == "softmax":
            self.alpha_conv = nn.Conv2d(
                merge_channels,
                1,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )

    def forward_train(self, **kwargs):
        x = kwargs["srcs"]
        sampling_maps = kwargs["sampling_maps"]
        valid_depth_masks = kwargs["valid_depth_masks"]
        valid_map_masks = kwargs["valid_map_masks"]

        bs, nv = x.shape[:2]

        x = x.view(bs * nv, *x.shape[2:])
        x = self.enc_net(x)

        x = F.grid_sample(
            x,
            sampling_maps.view(bs * nv, *sampling_maps.shape[2:]),
            mode="bilinear",
            padding_mode="zeros",
        )
        x = x.view(bs, nv, *x.shape[1:])

        if self.cat_masks:
            x = torch.cat([x, valid_depth_masks, valid_map_masks], dim=2)

        if self.cat_global_avg:
            x_avg = x.mean(axis=1, keepdim=True)
            x = torch.cat([x, x_avg.expand_as(x)], dim=2)

        if self.cat_global_max:
            x_max = x.max(axis=1, keepdim=True)[0]
            x = torch.cat([x, x_max.expand_as(x)], dim=2)

        hs = None
        if self.mode == "softmax":
            rgbs = []
            alphas = []
        for vidx in range(nv):
            y, hs = self.merge_net(x[:, vidx], hs)

            if self.mode == "softmax":
                rgbs.append(self.rgb_conv(y))
                alphas.append(self.alpha_conv(y))

        if self.mode == "single":
            x = self.rgb_conv(y)
        elif self.mode == "softmax":
            rgbs = torch.stack(rgbs)
            alphas = torch.stack(alphas)
            alphas = torch.softmax(alphas, dim=0)
            x = (alphas * rgbs).sum(dim=0)
            del rgbs, alphas

        return {"out": x}

    def forward_eval(self, **kwargs):
        x_all = kwargs["srcs"]
        sampling_maps = kwargs["sampling_maps"]
        valid_depth_masks = kwargs["valid_depth_masks"]
        valid_map_masks = kwargs["valid_map_masks"]

        bs, nv = x_all.shape[:2]

        if self.cat_global_avg:
            x_mean = None
            for vidx in range(nv):
                x = x_all[:, vidx]
                x = self.enc_net(x)
                x = F.grid_sample(
                    x,
                    sampling_maps[:, vidx],
                    mode="bilinear",
                    padding_mode="zeros",
                )
                if self.cat_masks:
                    x = torch.cat(
                        [
                            x,
                            valid_depth_masks[:, vidx],
                            valid_map_masks[:, vidx],
                        ],
                        dim=1,
                    )
                if x_mean is None:
                    x_mean = x / nv
                else:
                    x_mean += x / nv

        if self.cat_global_max:
            x_max = None
            for vidx in range(nv):
                x = x_all[:, vidx]
                x = self.enc_net(x)
                x = F.grid_sample(
                    x,
                    sampling_maps[:, vidx],
                    mode="bilinear",
                    padding_mode="zeros",
                )
                if self.cat_masks:
                    x = torch.cat(
                        [
                            x,
                            valid_depth_masks[:, vidx],
                            valid_map_masks[:, vidx],
                        ],
                        dim=1,
                    )
                if x_max is None:
                    x_max = x
                else:
                    x_max = torch.max(x, x_max)

        hs = None
        if self.mode == "softmax":
            rgbs = []
            alphas = []
        for vidx in range(nv):
            x = x_all[:, vidx]
            x = self.enc_net(x)

            x = F.grid_sample(
                x, sampling_maps[:, vidx], mode="bilinear", padding_mode="zeros"
            )

            if self.cat_masks:
                x = torch.cat(
                    [x, valid_depth_masks[:, vidx], valid_map_masks[:, vidx]],
                    dim=1,
                )

            if self.cat_global_avg:
                x = torch.cat([x, x_mean], dim=1)

            if self.cat_global_max:
                x = torch.cat([x, x_max], dim=1)

            x, hs = self.merge_net(x, hs)

            if self.mode == "softmax":
                rgbs.append(self.rgb_conv(x))
                alphas.append(self.alpha_conv(x))

        del x_all
        if self.mode == "single":
            x = self.rgb_conv(x)
        elif self.mode == "softmax":
            rgbs = torch.stack(rgbs)
            alphas = torch.stack(alphas)
            alphas = torch.softmax(alphas, dim=0)
            x = (alphas * rgbs).sum(dim=0)
            del rgbs, alphas

        return {"out": x}

    def forward(self, **kwargs):
        if self.training:
            return self.forward_train(**kwargs)
        else:
            return self.forward_eval(**kwargs)


def get_rnn_net(
    enc_net,
    merge_net,
    mode="softmax",
    cat_masks=True,
    cat_global_avg=False,
    cat_global_max=False,
    gru_nonlinearity="relu",
):
    if enc_net == "identity":
        enc_net = Identity()
        enc_channels = 3
    elif enc_net == "vgg16unet3":
        enc_net = VGGUNet(net="vgg16", n_encoder_stages=3)
        enc_channels = 64
    else:
        raise Exception("invalid enc_net")

    if cat_masks:
        enc_channels = enc_channels + 2

    if cat_global_avg:
        enc_channels *= 2

    if cat_global_max:
        enc_channels *= 2

    if merge_net == "gruunet4.64.3":
        merge_net = GRUUNet(
            enc_channels,
            enc_channels=[64, 128, 256, 512],
            dec_channels=[256, 128, 64],
            n_enc_convs=3,
            n_dec_convs=3,
            gru_nonlinearity=gru_nonlinearity,
        )
        merge_channels = 64
    elif merge_net == "unet4.64.3":
        merge_net = RNNWrapper(
            UNet(
                enc_channels,
                enc_channels=[64, 128, 256, 512],
                dec_channels=[256, 128, 64],
                n_enc_convs=3,
                n_dec_convs=3,
                out_channels=-1,
            )
        )
        merge_channels = 64
    else:
        raise Exception("invalid merge_net")

    return MappingRNN(
        enc_net,
        merge_net,
        merge_channels,
        mode,
        cat_masks,
        cat_global_avg,
        cat_global_max,
    )


class AggrNet(nn.Module):
    def __init__(
        self,
        enc_net,
        enc_net_channels,
        enc_channels=[64, 128, 256],
        dec_channels=[128, 64],
        n_enc_convs=2,
        n_dec_convs=2,
        aggr_mode="mean",
        multi_aggr="none",
        cat_masks=True,
    ):
        super().__init__()
        self.enc_net = enc_net
        self.aggr_mode = aggr_mode
        self.multi_aggr = multi_aggr
        self.cat_masks = cat_masks

        self.zero_tensor = torch.tensor([0]).float()

        self.encs = nn.ModuleList()
        self.enc_translates = nn.ModuleList()
        in_channels = 2 * enc_net_channels
        for enc_channel in enc_channels:
            stage = self.create_stage(in_channels, enc_channel, n_enc_convs)
            self.encs.append(stage)
            translate = nn.Conv2d(enc_channel, enc_channel, kernel_size=1)
            self.enc_translates.append(translate)
            in_channels = enc_channel
            if multi_aggr == "multiple":
                in_channels *= 2
            elif multi_aggr == "simple":
                in_channels += enc_net_channels

        self.decs = nn.ModuleList()
        for idx, dec_channel in enumerate(dec_channels):
            in_channels = enc_channels[-idx - 1] + enc_channels[-idx - 2]
            stage = self.create_stage(in_channels, dec_channel, n_dec_convs)
            self.decs.append(stage)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.alpha_conv = nn.Conv2d(
            dec_channels[-1], 1, kernel_size=1, padding=0
        )
        self.rgb_conv = nn.Conv2d(dec_channels[-1], 3, kernel_size=1, padding=0)

    def convrelu(self, in_channels, out_channels, kernel_size=3, padding=None):
        if padding is None:
            padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
        )

    def create_stage(self, in_channels, out_channels, n_convs):
        mods = []
        for _ in range(n_convs):
            mods.append(self.convrelu(in_channels, out_channels))
            in_channels = out_channels
        return nn.Sequential(*mods)

    def forward(self, **kwargs):
        x = kwargs["srcs"]
        sampling_maps = kwargs["sampling_maps"]
        valid_depth_masks = kwargs["valid_depth_masks"]
        valid_map_masks = kwargs["valid_map_masks"]

        bs, nv = x.shape[:2]

        x = x.view(bs * nv, *x.shape[2:])
        x = self.enc_net(x)

        x = F.grid_sample(
            x,
            sampling_maps.view(bs * nv, *sampling_maps.shape[2:]),
            mode="bilinear",
            padding_mode="zeros",
        )
        x = x.view(bs, nv, *x.shape[1:])

        if self.cat_masks:
            x = torch.cat([x, valid_depth_masks, valid_map_masks], dim=2)

        if self.aggr_mode == "maskedmean":
            with torch.no_grad():
                valid_map_mask_sum = valid_map_masks.sum(dim=1, keepdim=True)

        x = x.view(bs * nv, *x.shape[2:])
        outs = []
        for enc_idx, enc, enc_translates in zip(
            itertools.count(), self.encs, self.enc_translates
        ):
            if enc_idx == 0 or self.multi_aggr == "multiple":
                x = x.view(bs, nv, *x.shape[1:])
                if self.aggr_mode == "mean":
                    x_global = x.mean(axis=1, keepdim=True)
                elif self.aggr_mode == "max":
                    x_global = x.max(axis=1, keepdim=True)[0]
                elif self.aggr_mode == "maskedmean":
                    x_global = x.sum(axis=1, keepdim=True)
                    self.zero_tensor = self.zero_tensor.to(
                        valid_map_masks.device
                    )
                    x_global = torch.where(
                        valid_map_mask_sum.expand_as(x_global) > 0,
                        x_global / (valid_map_mask_sum + 1e-6),
                        self.zero_tensor,
                    )
                    if enc_idx < len(self.encs) - 1:
                        valid_map_mask_sum = valid_map_mask_sum.view(
                            bs, *valid_map_mask_sum.shape[2:]
                        )
                        valid_map_mask_sum = F.max_pool2d(
                            valid_map_mask_sum, kernel_size=2
                        )
                        valid_map_mask_sum = valid_map_mask_sum.view(
                            bs, 1, *valid_map_mask_sum.shape[1:]
                        )
                else:
                    raise Exception("invalid aggr_mode")
                x = torch.cat([x, x_global.expand_as(x)], dim=2)
                x = x.view(bs * nv, *x.shape[2:])
            elif enc_idx > 0 and self.multi_aggr == "simple":
                x_global = x_global.view(bs, *x_global.shape[2:])
                x_global = F.avg_pool2d(x_global, kernel_size=2)
                x_global = x_global.view(bs, 1, *x_global.shape[1:])
                x = x.view(bs, nv, *x.shape[1:])
                x = torch.cat([x, x_global.expand(-1, nv, -1, -1, -1)], dim=2)
                x = x.view(bs * nv, *x.shape[2:])
            x = enc(x)
            outs.append(enc_translates(x))
            if enc_idx < len(self.encs) - 1:
                x = F.avg_pool2d(x, kernel_size=2)

        for dec in self.decs:
            x0, x1 = outs.pop(), outs.pop()
            x = torch.cat((self.upsample(x0), x1), dim=1)
            x = dec(x)
            outs.append(x)

        x = outs.pop()
        alpha = self.alpha_conv(x)
        alpha = alpha.view(bs, nv, *alpha.shape[1:])
        x = self.rgb_conv(x)
        x = x.view(bs, nv, *x.shape[1:])
        x = (x * alpha).sum(dim=1)

        return {"out": x}


def get_aggr_net(
    enc_net, merge_net, aggr_mode, multi_aggr="none", cat_masks=True
):
    if enc_net == "identity":
        enc_net = Identity()
        enc_net_channels = 3
    elif enc_net == "vgg16unet3":
        enc_net = VGGUNet(net="vgg16", n_encoder_stages=3)
        enc_net_channels = 64
    else:
        raise Exception("invalid enc_net")

    if cat_masks:
        enc_net_channels = enc_net_channels + 2

    if merge_net == "unet4.64.3":
        enc_channels = [64, 128, 256, 512]
        dec_channels = [256, 128, 64]
        n_enc_convs = 3
        n_dec_convs = 3
    else:
        raise Exception("invalid merge_net")

    return AggrNet(
        enc_net,
        enc_net_channels,
        enc_channels=enc_channels,
        dec_channels=dec_channels,
        n_enc_convs=n_enc_convs,
        n_dec_convs=n_dec_convs,
        aggr_mode=aggr_mode,
        multi_aggr=multi_aggr,
        cat_masks=cat_masks,
    )

