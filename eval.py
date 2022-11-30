import torch
def predict(model, dataloader):
    model.eval()
    correct=0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)
    All_loss = 0
    All_size = 0
    with torch.no_grad():
        for batch, (image, target) in enumerate(dataloader):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss1 = criterion(output, target)
            All_loss += loss1.item() * image.size(0)
            All_size += image.size(0)
            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()
            acc = correct * 100 / All_size
        return acc, All_loss/All_size