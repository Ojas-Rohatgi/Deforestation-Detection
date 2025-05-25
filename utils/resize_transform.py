from torchvision import transforms

resize_transform = transforms.Compose([
    transforms.Resize((256, 256)),
])