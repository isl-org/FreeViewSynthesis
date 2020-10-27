import numpy as np
import PIL
import logging
import sys

sys.path.append("../")
import co
import ext


def load(p, height=None, width=None):
    if p.suffix == ".npy":
        return np.load(p)
    elif p.suffix in [".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"]:
        im = PIL.Image.open(p)
        im = np.array(im)
        if (
            height is not None
            and width is not None
            and (im.shape[0] != height or im.shape[1] != width)
        ):
            raise Exception("invalid size of image")
        im = (im.astype(np.float32) / 255) * 2 - 1
        im = im.transpose(2, 0, 1)
        return im
    else:
        raise Exception("invalid suffix")


class Dataset(co.mytorch.BaseDataset):
    def __init__(
        self,
        *,
        name,
        tgt_im_paths,
        tgt_dm_paths,
        tgt_Ks,
        tgt_Rs,
        tgt_ts,
        tgt_counts,
        src_im_paths,
        src_dm_paths,
        src_Ks,
        src_Rs,
        src_ts,
        im_size=None,
        pad_width=None,
        patch=None,
        n_nbs=5,
        nbs_mode="sample",
        bwd_depth_thresh=0.1,
        invalid_depth_to_inf=True,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.tgt_im_paths = tgt_im_paths
        self.tgt_dm_paths = tgt_dm_paths
        self.tgt_Ks = tgt_Ks
        self.tgt_Rs = tgt_Rs
        self.tgt_ts = tgt_ts
        self.tgt_counts = tgt_counts

        self.src_im_paths = src_im_paths
        self.src_dm_paths = src_dm_paths
        self.src_Ks = src_Ks
        self.src_Rs = src_Rs
        self.src_ts = src_ts

        self.im_size = im_size
        self.pad_width = pad_width
        self.patch = patch
        self.n_nbs = n_nbs
        self.nbs_mode = nbs_mode
        self.bwd_depth_thresh = bwd_depth_thresh
        self.invalid_depth_to_inf = invalid_depth_to_inf

        tmp = np.load(tgt_dm_paths[0])
        self.height, self.width = tmp.shape
        del tmp

        n_tgt_im_paths = len(tgt_im_paths) if tgt_im_paths else 0
        shape_tgt_im = (
            self.load_pad(tgt_im_paths[0]).shape if tgt_im_paths else None
        )
        logging.info(
            f"    #tgt_im_paths={n_tgt_im_paths}, #tgt_counts={tgt_counts.shape}, tgt_im={shape_tgt_im}, tgt_dm={self.load_pad(tgt_dm_paths[0]).shape}"
        )

    def pad(self, im):
        if self.im_size is not None:
            shape = [s for s in im.shape]
            shape[-2] = self.im_size[0]
            shape[-1] = self.im_size[1]
            im_p = np.zeros(shape, dtype=im.dtype)
            sh = min(im_p.shape[-2], im.shape[-2])
            sw = min(im_p.shape[-1], im.shape[-1])
            im_p[..., :sh, :sw] = im[..., :sh, :sw]
            im = im_p
        if self.pad_width is not None:
            h, w = im.shape[-2:]
            mh = h % self.pad_width
            ph = 0 if mh == 0 else self.pad_width - mh
            mw = w % self.pad_width
            pw = 0 if mw == 0 else self.pad_width - mw
            shape = [s for s in im.shape]
            shape[-2] += ph
            shape[-1] += pw
            im_p = np.zeros(shape, dtype=im.dtype)
            im_p[..., :h, :w] = im
            im = im_p
        return im

    def load_pad(self, p):
        im = load(p)
        return self.pad(im)

    def base_len(self):
        return len(self.tgt_dm_paths)

    def base_getitem(self, idx, rng):
        count = self.tgt_counts[idx]
        if self.nbs_mode == "argmax":
            nbs = np.argsort(count)[::-1]
            nbs = nbs[: self.n_nbs]
        elif self.nbs_mode == "sample":
            nbs = rng.choice(
                count.shape[0], self.n_nbs, replace=False, p=count / count.sum()
            )
        else:
            raise Exception("invalid nbs_mode")

        ret = {}

        tgt_dm = load(self.tgt_dm_paths[idx])
        tgt_dm = self.pad(tgt_dm)
        if not self.train:
            ret["tgt_dm"] = tgt_dm

        tgt_K = self.tgt_Ks[idx]
        tgt_R = self.tgt_Rs[idx]
        tgt_t = self.tgt_ts[idx]

        src_dms = np.array([load(self.src_dm_paths[ii]) for ii in nbs])
        src_dms = self.pad(src_dms)

        src_Ks = np.array([self.src_Ks[ii] for ii in nbs])
        src_Rs = np.array([self.src_Rs[ii] for ii in nbs])
        src_ts = np.array([self.src_ts[ii] for ii in nbs])

        if self.patch:
            patch_h_from = rng.randint(0, tgt_dm.shape[0] - self.patch[0])
            patch_w_from = rng.randint(0, tgt_dm.shape[1] - self.patch[1])
            patch_h_to = patch_h_from + self.patch[0]
            patch_w_to = patch_w_from + self.patch[1]
            patch = np.array(
                (patch_h_from, patch_h_to, patch_w_from, patch_w_to),
                dtype=np.int32,
            )
        else:
            patch = np.array(
                (0, tgt_dm.shape[0], 0, tgt_dm.shape[1]), dtype=np.int32
            )

        sampling_maps, valid_depth_masks, valid_map_masks = ext.preprocess.get_sampling_map(
            tgt_dm,
            tgt_K,
            tgt_R,
            tgt_t,
            src_dms,
            src_Ks,
            src_Rs,
            src_ts,
            patch,
            self.bwd_depth_thresh,
            self.invalid_depth_to_inf,
        )
        ret["sampling_maps"] = sampling_maps
        ret["valid_depth_masks"] = valid_depth_masks
        ret["valid_map_masks"] = valid_map_masks

        if self.tgt_im_paths:
            tgt = load(self.tgt_im_paths[idx])
            ret["tgt"] = tgt[
                :,
                patch[0] : min(tgt.shape[1], patch[1]),
                patch[2] : min(tgt.shape[2], patch[3]),
            ]
        else:
            tgt_height = min(tgt_dm.shape[0], patch[1]) - patch[0]
            tgt_width = min(tgt_dm.shape[1], patch[3]) - patch[2]
            ret["tgt"] = np.zeros((3, tgt_height, tgt_width), dtype=np.float32)

        srcs = np.array([self.load_pad(self.src_im_paths[ii]) for ii in nbs])
        ret["srcs"] = srcs

        return ret
