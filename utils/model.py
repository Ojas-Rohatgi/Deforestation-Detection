import segmentation_models_pytorch as smp
from utils.config import *

model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    in_channels=4,
    classes=1,
    activation=None,
).to(DEVICE)