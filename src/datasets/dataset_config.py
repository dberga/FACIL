dataset_config = {
    'flowers': {
        'path': 'data/datasets/Framework/flowers/',
        'resize': (256, 256),
        'crop': 224,
        'flip': True,
        'normalize': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    },
    'scenes': {
        'path': 'data/datasets/Framework/indoor_scenes/',
        'resize': (256, 256),
        'crop': 224,
        'flip': True,
        'normalize': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    },
    'birds': {
        'path': 'data/datasets/Framework/cubs_cropped/',
        'resize': (224, 224),
        'crop': None,
        'flip': True,
        'normalize': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    },
    'cars': {
        'path': 'data/datasets/Framework/stanford_cars_cropped/',
        'resize': (224, 224),
        'crop': None,
        'flip': True,
        'normalize': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    },
    'aircraft': {
        'path': 'data/datasets/Framework/fgvc-aircraft-2013b/',
        'resize': (224, 224),
        'crop': None,
        'flip': True,
        'normalize': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    },
    'actions': {
        'path': 'data/datasets/Framework/Stanford40/',
        'resize': (224, 224),
        'crop': None,
        'flip': True,
        'normalize': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    },
    'letters': {
        'path': 'data/datasets/letters/',
        'resize': (224, 224),
        'crop': None,
        'flip': False,
        'normalize': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    },
    'svhn': {
        'path': 'data/datasets/svhn/',
        'resize': (224, 224),
        'crop': None,
        'flip': False,
        'normalize': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    },
    'cifar100': {
        'path': 'data/datasets/cifar100/',
        'resize': None,
        'pad': 4,
        'crop': 32,
        'flip': True,
        'normalize': ((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023))
    },
    'cifar100_icarl': {
        'path': '/data/datasets/cifar100/',
        'resize': None,
        'pad': 4,
        'crop': 32,
        'flip': True,
        'normalize': ((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)),
        'class_order': [
            68, 56, 78, 8, 23, 84, 90, 65, 74, 76, 40, 89, 3, 92, 55, 9, 26, 80, 43, 38, 58, 70, 77, 1, 85, 19, 17, 50,
            28, 53, 13, 81, 45, 82, 6, 59, 83, 16, 15, 44, 91, 41, 72, 60, 79, 52, 20, 10, 31, 54, 37, 95, 14, 71, 96,
            98, 97, 2, 64, 66, 42, 22, 35, 86, 24, 34, 87, 21, 99, 0, 88, 27, 18, 94, 11, 12, 47, 25, 30, 46, 62, 69,
            36, 61, 7, 63, 75, 5, 32, 4, 51, 48, 73, 93, 39, 67, 29, 49, 57, 33
        ]
    },
    'tiny_imagenet': {
        'path': 'data/datasets/tiny_imagenet/tiny-imagenet-200/',
        'resize': None,
        'crop': None,
        'flip': True,
        'normalize': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    },
    'imagenet_256': {
        'path': 'data/datasets/Framework/ILSVRC12_256/',
        'resize': None,
        'crop': 224,
        'flip': True,
        'normalize': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    },
    'imagenet_no_birds': {
        'path': 'data/datasets/Framework/ILSVRC12_256/',
        'resize': None,
        'crop': 224,
        'flip': True,
        'normalize': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    }
}
# Add missing keys:
for dset in dataset_config.keys():
    for k in ['resize', 'pad', 'crop', 'normalize', 'class_order']:
        if k not in dataset_config[dset].keys():
            dataset_config[dset][k] = None
    if 'flip' not in dataset_config[dset].keys():
        dataset_config[dset]['flip'] = False
