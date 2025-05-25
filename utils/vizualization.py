import matplotlib.pyplot as plt

def plot_training(history):
    epochs = history["epoch"]
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["val_iou"], label="Validation IoU", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("IoU Score")
    plt.title("Validation IoU")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("training_plots.png")
    plt.show()
