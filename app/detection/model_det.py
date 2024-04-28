import torch
import torch.utils.data
from effdet import get_efficientdet_config, create_model_from_config
from Fitter import Fitter
from TrainGlobalConfig import TrainGlobalConfig
from detection_dataset import DetectionDataset

bs = 16

train_dataset = DetectionDataset(base_folder="D:/Licenta/Proiect/data/", split_name="train", number_of_slices=22,
                                 batch_size=bs)
test_dataset = DetectionDataset(base_folder="D:/Licenta/Proiect/data/", split_name="test", number_of_slices=22,
                                batch_size=bs)


def collate_fn(batch):
    return tuple(zip(*batch))


def get_net():
    config = get_efficientdet_config('tf_efficientdet_d0')

    config.image_size = (512, 512)
    config.norm_kwargs = dict(eps=.001, momentum=.01)

    net = create_model_from_config(config=config, bench_task='train', num_classes=2, pretrained=True,
                                   bench_labeler=True)

    return net


net = get_net()


def run_training():
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=bs,
        num_workers=0,
        collate_fn=collate_fn,
    )

    val_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=0,
        collate_fn=collate_fn,
    )

    fitter = Fitter(model=net, config=TrainGlobalConfig)
    fitter.fit(train_loader, val_loader)


run_training()
