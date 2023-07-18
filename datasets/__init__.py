import torch.utils.data
import torchvision

from .hico import build as build_hico
from .vcoco import build as build_vcoco
import torch.utils.data

from .OR import build as build_or


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco

def build_dataset(image_set, args):
    if args.dataset_file == 'hico':
        return build_hico(image_set, args)
    if args.dataset_file == 'vcoco':
        return build_vcoco(image_set, args)
    if args.dataset_file == 'or':
        return build_or(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
