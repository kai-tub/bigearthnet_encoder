from sys import prefix
from typing import Optional
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.profiler import AdvancedProfiler
from pytorch_lightning.loggers import WandbLogger

import torch
import numpy as np
import timm
from bigearthnet_encoder.squirrel_ext import ConfigurableMessagePackDriver
from squirrel.iterstream.torch_composables import TorchIterable
from torch.optim import Adam, RAdam
from bigearthnet_common.constants import BEN_10m_CHANNELS, BEN_20m_CHANNELS, BAND_STATS_S2
from bigearthnet_common.base import ben_19_labels_to_multi_hot
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader
from functools import partial
import kornia.augmentation as K
from torchmetrics import AveragePrecision, MetricCollection

# Must be tuned according to environment and data loader!
torch.set_num_threads(4)

def torch_interpolate(tns: torch.Tensor, target_shape: int = 120):
    squeeze = False
    if tns.ndim == 3:
        squeeze = True
        tns = tns.unsqueeze(dim=0)

    tns = interpolate(
        tns,
        (target_shape, target_shape),
        mode="bicubic",
    )
    if squeeze:
        tns = tns.squeeze(0)

    return tns

def build_ben_tns(patch_dict, in_chans: int = 10, norm_factor=10_000):
    lbls = patch_dict["new_labels"]
    lbls_tns = torch.Tensor(ben_19_labels_to_multi_hot(lbls))

    arrs_10m = [patch_dict[key] for key in BEN_10m_CHANNELS]
    arrs_10m = [patch_dict[key] for key in BEN_10m_CHANNELS]
    # Convert to float32 including 'smart' normalization
    # for further processing
    arr_10m = np.stack(arrs_10m) / norm_factor
    tns_10m = torch.Tensor(arr_10m)
    if in_chans == 4:
        return tns_10m, lbls_tns
    elif in_chans == 10:
        arrs_20m = [patch_dict[key] for key in BEN_20m_CHANNELS]
        arr_20m = np.stack(arrs_20m) / norm_factor
        tns_20m = torch.Tensor(arr_20m)
        tns_20m_interp = torch_interpolate(tns_20m, target_shape=120)
        return torch.cat((tns_10m, tns_20m_interp)), lbls_tns
        # return torch.cat((tns_10m, tns_20m_interp))
    raise ValueError

class FastBigEarthNet(pl.LightningModule):
    def __init__(
        self,
        data_dir: str = "/data/datasets/BigEarthNet/BigEarthNet-S2/S2_squirrel/",
        batch_size: int = 512,
        in_chans: int = 10,
        num_classes: int = 19,
        learning_rate: float = 1e-4,
        norm_factor: int = 10_000,
        model: str = "resnet50",
        take: Optional[int] = None,
    ):
        super().__init__()
        self.save_hyperparameters("batch_size", "in_chans", "learning_rate", "model", "take")
        self.num_classes = num_classes

        self.model = timm.create_model(
            # "efficientnetv2_l",
            model,
            pretrained=False,
            in_chans=in_chans,
            num_classes=num_classes
        )
        # most CPU time is probably spend inside of collate function!
        self.take = take

        self.data_driver = ConfigurableMessagePackDriver(data_dir)
        assert self.data_driver.keys()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.in_chans = in_chans
        self.learning_rate = learning_rate
        self.norm_factor = norm_factor
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.metrics = MetricCollection(
            {
                "macro_mAP": AveragePrecision(num_classes=self.num_classes, pos_label=1, average="macro", compute_on_cpu=False),
                "micro_mAP": AveragePrecision(num_classes=self.num_classes, pos_label=1, average="micro", compute_on_cpu=False)
            },
        )

        if self.in_chans == 10:
            band_keys = [*BEN_10m_CHANNELS, *BEN_20m_CHANNELS]
        elif self.in_chans == 4:
            band_keys = [*BEN_10m_CHANNELS]
        else:
            raise ValueError

        # TODO: Hack for now
        mean = np.array([BAND_STATS_S2["mean"][b] for b in band_keys]) / self.norm_factor
        std = np.array([BAND_STATS_S2["std"][b] for b in band_keys]) / self.norm_factor

        self.train_augs =  K.AugmentationSequential(
            K.RandomHorizontalFlip(),
            K.RandomVerticalFlip(),
            K.Normalize(mean=mean, std=std),
            # K.RandomResizedCrop(size=(120, 120)),
            # Setting this to true has a very limited performance impact
            # at least for a batch size of 1024
            same_on_batch=False,
        )

    def configure_optimizers(self):
        return RAdam(self.parameters(), lr=self.learning_rate)

    def _init_dataloder(self, split: str):
        builder = partial(build_ben_tns, in_chans=self.in_chans, norm_factor=self.norm_factor)
        it = (
            self.data_driver
            .get_iter(split, shuffle_key_buffer=1000, shuffle_item_buffer=1_000, max_workers=4) # fitted to dataset
            .async_map(builder, max_workers=4)
            .take(self.take)
            .compose(TorchIterable)
        )
        return DataLoader(it, batch_size=self.batch_size, pin_memory=True)

    def train_dataloader(self):
        return self._init_dataloder("train")

    def val_dataloader(self):
        return self._init_dataloder("validation")

    def test_dataloader(self):
        return self._init_dataloder("test")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        # like to keep it separate from inference `forward`
        # This is also the place where I would do the kornia
        # train transformations
        aug_inputs = self.train_augs(inputs)
        outputs = self.model(aug_inputs)
        loss = self.criterion(outputs, targets)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        self.log("val_loss", loss)
        metrics = self.metrics(outputs, targets)
        self.log_dict(metrics)

        # return outputs.cpu()

    # def validation_epoch_end(self, validation_step_outputs):
    #     all_preds = torch.stack(validation_step_outputs)

# bs=1600
# with val + in_chans=4 + 4async + 1map: 1s/it with huge CPU usage
# with val + in_chans=4 + 1sync + 1map: 1s/it with huge CPU usage

if __name__ == "__main__":
    wandb_logger = WandbLogger(
        project="ben_loader_tests"
    )

    module = FastBigEarthNet(
        batch_size=1600,
        # in_chans=4,
        in_chans=10,
        # take=50_000,
    )

    # uses gpu 4
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[4],
        enable_checkpointing=False,
        max_epochs=2,
        logger=wandb_logger,
        # auto_lr_find=True,
        # profiler=AdvancedProfiler(dirpath=".", filename="profile"),
        # profiler="simple",
        # callbacks=[TQDMProgressBar(refresh_rate=1)]
    )
    # lr_finder = trainer.tuner.lr_find(module, min_lr=1e-3, max_lr=10)
    # fig = lr_finder.plot(suggest=True)
    # fig.savefig("out.png")

    trainer.fit(module)

    # benchmark with validate to ensure that the optimizer is not causing any issues
    # trainer.validate(module); for 50_000 samples, it takes ~50s
