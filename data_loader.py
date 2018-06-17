import torch
from torchvision import datasets
from torchvision import transforms

def get_loader(image_path, image_size, dataset, batch_size, num_workers=2):
    """Build and return data loader."""
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        # transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    if dataset == 'LSUN':
        dataset = datasets.LSUN(image_path, classes=['church_outdoor_train'], transform=transform)
    elif dataset == 'CelebA_FD':
        dataset = datasets.ImageFolder(image_path, transform=transform)
    elif dataset == 'cifar':
        dataset = datasets.CIFAR10(image_path, transform=transform, download=True)
    elif dataset == 'CelebA':
        transform = transforms.Compose([
            transforms.CenterCrop(160),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = datasets.ImageFolder(image_path+'/CelebA', transform=transform)
    
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size, 
                                              shuffle=True,
                                              num_workers=2,
                                              drop_last= True)
    return data_loader
