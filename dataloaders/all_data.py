import torchvision
import torchvision.transforms as transforms
def process_dataset(rootdir, data_name):
    if data_name == 'MNIST':
        train_dataset = torchvision.datasets.MNIST(root = rootdir,
                                           train = True,
                                           transform = transforms.Compose([
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.1307,), (0.3081,))]),
                                           download = True)
        test_dataset = torchvision.datasets.MNIST(root = rootdir,
                                          train = False,
                                          transform = transforms.Compose([
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.1307,), (0.3081,))]),
                                          download=True)
    elif data_name == 'CIFAR10':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        train_dataset = torchvision.datasets.CIFAR10(root = rootdir, 
                                            train=True, 
                                            transform=transforms.Compose([
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomCrop(32, 4),
                                            transforms.ToTensor(),
                                            normalize,
                                        ]), download=True)
        test_dataset = torchvision.datasets.CIFAR10(root = rootdir,
                                          train = False,
                                          transform = transforms.Compose([
                                          transforms.ToTensor(),
                                          normalize,]),
                                          download = True
                                          )
    elif data_name == 'CIFAR100':
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                     std=[0.2675, 0.2565, 0.2761])
        train_dataset = torchvision.datasets.CIFAR100(root = rootdir, 
                                            train=True, 
                                            transform=transforms.Compose([
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomCrop(32, 4),
                                            transforms.ToTensor(),
                                            normalize,
                                        ]), download=True)
        test_dataset = torchvision.datasets.CIFAR100(root = rootdir,
                                          train = False,
                                          transform = transforms.Compose([
                                          transforms.ToTensor(),
                                          normalize,]),
                                          download = True
                                          )   
    elif data_name == 'ImageNet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        train_dataset = torchvision.datasets.ImageNet(root = rootdir, 
                                            split='train', transform=transforms.Compose([
                                            transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            normalize,
                                        ]))
        test_dataset = torchvision.datasets.ImageNet(root= rootdir,
                                                  split = 'val',
                                                  transform = transforms.Compose([
                                                  transforms.Scale(256),
                                                  transforms.CenterCrop(224),
                                                  transforms.ToTensor(),
                                                  normalize,
                                                  ]))
    return train_dataset, test_dataset
        
