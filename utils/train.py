from utils.config import *
from utils.model import model
import time
from tqdm import tqdm
import json
from utils.evaluation import evaluate
from utils.vizualization import plot_training


def train(train_loader, val_loader, loss_fn, optimizer):
    history = {
        "epoch": [],
        "train_loss": [],
        "val_iou": [],
        "time": []
    }

    for epoch in range(1, EPOCHS + 1):
        model.train()
        start_time = time.time()
        total_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)

        for images, masks in progress_bar:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            preds = model(images)
            preds = torch.sigmoid(preds)
            loss = loss_fn(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        val_iou = evaluate(val_loader, return_score=True)
        duration = time.time() - start_time

        history["epoch"].append(epoch)
        history["train_loss"].append(avg_loss)
        history["val_iou"].append(val_iou)
        history["time"].append(duration)

        print(f"\n[Epoch {epoch}] Loss: {avg_loss:.4f} | Val IoU: {val_iou:.4f} | Time: {duration:.1f}s")

    # Save history and plot
    with open("training_log.json", "w") as f:
        json.dump(history, f, indent=4)

    plot_training(history)

    # Save the trained model weights
    torch.save(model.state_dict(), "deforestation_unet.pth")
    print("Model saved to deforestation_unet.pth")

    # Save the trained model
    torch.save(model, "deforestation_unet_full_model.pt")
    print("Full model saved to deforestation_unet_full_model.pt")

