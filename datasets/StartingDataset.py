import torch
from torchvision import transforms
from PIL import Image


class StartingDataset(torch.utils.data.Dataset):
    def __init__(self, images, truth, base, size):
        self.images = images
        self.truth = truth
        self.base = base
        self.size = size

    def __getitem__(self, index):
        path = self.images[index]
        label = self.truth[index]

        img = Image.open(f'{self.base}/{path}')
        # preprocess = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #     transforms.Resize([224,224])
        # ])

        preprocess2 = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        preprocess_for_efficient_net = transforms.Compose([
            transforms.Resize(self.size),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        tensor = preprocess_for_efficient_net(img)
        return tensor, int(label)

    def __len__(self):
        return len(self.truth)
