import numpy as np
from PIL.Image import Image
from xmodaler.datasets.images.mscoco_raw import MSCoCoRawDataset
from xmodaler.config import kfg
from xmodaler.functional import dict_as_tensor
import xmodaler.utils.comm as comm
from xmodaler.config import get_cfg
from xmodaler.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch, build_engine
from xmodaler.modeling import add_config


class ArtGraphRawDataset(MSCoCoRawDataset):
    def __init__(self,
                 max_seq_len: int,
                 max_feat_num: int,
                 sample_ids,
                 file_paths):
        super().__init__(max_seq_len=max_seq_len,
                         max_feat_num=max_feat_num,
                         sample_ids=sample_ids,
                         file_paths=file_paths)

    def __call__(self, img_path):
        sample_id = img_path
        image = self.preprocess(Image.open(img_path)).unsqueeze(0).to('cuda')
        att_feats, global_feat = self.model.encode_image(image)
        att_feats = self.pool2d(att_feats)
        att_feats = att_feats.permute(0, 2, 3, 1)
        att_feats = att_feats.reshape(-1, att_feats.shape[-1])
        att_feats = att_feats[0:self.max_feat_num]

        ret = {
            kfg.IDS: sample_id,
            kfg.ATT_FEATS: att_feats.data.cpu().float().numpy(),
            kfg.GLOBAL_FEATS: global_feat.data.cpu().float().numpy()
        }

        g_tokens_type = np.ones((self.max_seq_len,), dtype=np.int64)
        ret.update({kfg.G_TOKENS_TYPE: g_tokens_type})
        dict_as_tensor(ret)
        return ret


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    tmp_cfg = cfg.load_from_file_tmp(args.config_file)
    add_config(cfg, tmp_cfg)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    # cfg.MODEL.WEIGTHS='./configs/image_caption/cosnet/cosnet_xe.pth'
    trainer = build_engine(cfg)
    trainer.resume_or_load(True)
    args.eval_only = True
    return trainer

if __name__ == '__main__':
    img_dir=r'D:\raffaele\UNIVERSITA\magistrale_data_science\2_anno\tesi\feature_extraction\dataset\images-resized'

    print(kfg.G_TOKENS_TYPE, kfg.SEMANTICS_IDS)
    dat=MSCoCoRawDataset(
        max_seq_len=20,
        max_feat_num=50,
        sample_ids=[],
        file_paths=img_dir
    )

    dataset_dict={kfg.IDS: fr'{img_dir}\leonardo-da-vinci_mona-lisa.jpg',
                  'path': fr'{img_dir}\leonardo-da-vinci_mona-lisa.jpg'}
    out = dat(dataset_dict)


    args = default_argument_parser().parse_args()
    trainer = main(args)
    trainer.model(out)

