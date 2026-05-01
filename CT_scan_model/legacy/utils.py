import torch
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn
import torchvision.utils
from torch.distributions import uniform
from mpl_toolkits.axes_grid1 import ImageGrid
import os
from .configuration import *
import sys
import os
import datetime
import csv


# Create a figure with a large size to accommodate multiple images
fig = plt.figure(figsize=(100, 100))

def show_images(images, index, label):
    """
    Display a single image with a corresponding label.

    Parameters:
        images (torch.Tensor): The image tensor to be displayed.
        index (int): The position index in the figure.
        label (str): The title label for the image.
    """
    ax = fig.add_subplot(21, 1, index + 1, xticks=[], yticks=[])
    plt.gca().set_title(label)  # Set the title of the image
    plt.imshow(images.cpu(), cmap='gray')  # Convert tensor to CPU and display in grayscale
    plt.show()


def show_images_color(images, index, label):
    """
    Display a single image with a corresponding label.

    Parameters:
        images (torch.Tensor): The image tensor to be displayed.
        index (int): The position index in the figure.
        label (str): The title label for the image.
    """
    ax = fig.add_subplot(21, 1, index + 1, xticks=[], yticks=[])
    plt.gca().set_title(label)  # Set the title of the image
    plt.imshow(images.cpu().permute(1,2,0))  # Convert tensor to CPU and display in grayscale
    plt.show()


def show_grids(images, title, n_epoch):
    """
    Display a grid of images with their corresponding labels.

    Parameters:
        images (torch.Tensor): Tensor containing multiple images.
        title ...
        n_epoch (int): The current epoch number for naming the saved image.
    """
    fig = plt.figure(figsize=(40., 40.))

    # Create a 2x2 image grid
    grid = ImageGrid(fig, 111, nrows_ncols=(2, 2), axes_pad=0.5)

    # Support both square and non-square images.
    images_cpu = images.cpu()
    if images_cpu.dim() == 4 and images_cpu.shape[1] == 1:
        images_cpu = images_cpu[:, 0]

    for j, (ax, im) in enumerate(zip(grid, images_cpu)):
        ax.imshow(im, cmap='gray')  # Display each image in grayscale
        ax.title.set_text(title)  # Set title based on label
        ax.title.set_size(28)  # Increase title font size

    # Save the figure to disk with a unique name based on the epoch
    figname = os.path.join(trained_dir_2nd, str(n_epoch))
    fig.savefig(figname, bbox_inches='tight')
    plt.close(fig)

def show_grids_color(images, title, n_epoch):
    """
    Display a grid of images with their corresponding labels.

    Parameters:
        images (torch.Tensor): Tensor containing multiple images.
        title ...
        n_epoch (int): The current epoch number for naming the saved image.
    """
    fig = plt.figure(figsize=(40., 40.))

    # Create a 2x2 image grid
    grid = ImageGrid(fig, 111, nrows_ncols=(2, 2), axes_pad=0.5)

    for j, (ax, im) in enumerate(zip(grid, images.cpu())):
        ax.imshow(im.permute(1, 2, 0).clip(-1, 1) * 0.5 + 0.5)  # Display each image in grayscale
        ax.title.set_text(title)  # Set title based on label
        ax.title.set_size(28)  # Increase title font size

    # Save the figure to disk with a unique name based on the epoch
    figname = os.path.join(trained_dir, str(n_epoch))
    fig.savefig(figname, bbox_inches='tight')
    plt.close(fig)


def save_model(address, model, ema_model, optimizer):
    """
    Save the model, EMA model, and optimizer states to a file.

    Parameters:
        address (str): The file path where the model checkpoint will be saved.
    """
    checkpoint = {
        "model_state": model.state_dict(),
        "ema_model_state": ema_model.state_dict(),
        "model_optimizer": optimizer.state_dict()
    }
    torch.save(checkpoint, address)


def load_model(address, model, ema_model, optimizer):
    """
    Load the model, EMA model, and optimizer states from a checkpoint.

    Parameters:
        address (str): The file path of the saved model checkpoint.
    """
    checkpoint = torch.load(address)
    model.load_state_dict(checkpoint["model_state"])
    ema_model.load_state_dict(checkpoint["ema_model_state"])
    optimizer.load_state_dict(checkpoint["model_optimizer"])


def load_model2(address, model, optimizer):
    """
    Load only the model and optimizer states from a checkpoint.

    Parameters:
        address (str): The file path of the saved model checkpoint.
    """
    checkpoint = torch.load(address)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["model_optimizer"])

def load_model_mp(address, model, ema_model, optimizer):
    if torch.cuda.is_available():
        checkpoint = torch.load(address)
    else:
        checkpoint = torch.load(address, map_location=torch.device('cpu'))

    model_state_dict = checkpoint["model_state"]
    ema_model_state_dict = checkpoint["ema_model_state"]
    optimizer_state_dict = checkpoint["model_optimizer"]

    # Entferne das Präfix "module." aus den Schlüsseln für model_state_dict
    from collections import OrderedDict
    new_model_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k[7:] if k.startswith('module.') else k  # Entferne "module."
        new_model_state_dict[name] = v

    # Entferne das Präfix "module." aus den Schlüsseln für ema_model_state_dict
    new_ema_model_state_dict = OrderedDict()
    for k, v in ema_model_state_dict.items():
        name = k[7:] if k.startswith('module.') else k  # Entferne "module."
        new_ema_model_state_dict[name] = v

    model.load_state_dict(new_model_state_dict)
    ema_model.load_state_dict(new_ema_model_state_dict)
    optimizer.load_state_dict(optimizer_state_dict)

def class_maker(batch_size, labels, class_table):
    """
    Generate class-specific attributes from a given class table.

    Parameters:
        batch_size (int): The number of samples in a batch.
        labels (torch.Tensor): The class labels for each sample.
        class_table (torch.Tensor): A lookup table mapping class labels to attributes.

    Returns:
        Tuple of torch.Tensors: SHP, Loc, CR, SK, HT, FT, Mag attributes.
    """
    FM = torch.zeros([batch_size])

    for i in range(batch_size):
        FM[i] = class_table[0, labels[i]]

    return FM


def weights_init(m):
    """
    Initialize weights of a given model layer.

    Parameters:
        m (torch.nn.Module): A PyTorch model layer.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # Initialize convolutional layers with a normal distribution (mean=0, std=0.02)
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('Norm') != -1:
        # Initialize normalization layers with mean=1, std=0.02 and bias=0
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)

def save_loss_to_csv(epoch, loss, filepath=trained_dir, filename="training_loss.csv"):
    """
    Speichert den Loss-Wert jeder Epoche in eine CSV-Datei.

    Parameters:
        epoch (int): Die aktuelle Epochennummer
        loss (float): Der Loss-Wert für die Epoche
        filepath (str): Das Verzeichnis, in dem die Datei gespeichert wird
        filename (str): Name der CSV-Datei
    """
    complete_path = os.path.join(filepath, filename)
    file_exists = os.path.isfile(complete_path)

    with open(complete_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['Epoch', 'Loss'])
        writer.writerow([epoch, loss])

class Logger:
    """
    Logger-Klasse zum Umleiten der Standardausgabe in eine Datei und zur Konsole.
    """

    def __init__(self, log_dir=os.path.join(trained_dir, "logs")):
        # Erstelle Logs-Verzeichnis falls nötig
        os.makedirs(log_dir, exist_ok=True)

        # Erzeuge Dateinamen mit Zeitstempel
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = os.path.join(log_dir, f"log_{timestamp}.txt")

        # Speichere ursprünglichen stdout
        self.terminal = sys.stdout

        # Öffne Log-Datei
        self.log_file = open(log_file, "w", encoding="utf-8")

        print(f"Logging in Datei: {log_file}")

    def write(self, message):
        # Schreibe in Konsole und Datei
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()  # Stellt sicher, dass Inhalt sofort geschrieben wird

    def flush(self):
        # Für Kompatibilität mit sys.stdout
        self.terminal.flush()
        self.log_file.flush()

    def __del__(self):
        # Schließe Log-Datei bei Beendigung
        if hasattr(self, 'log_file'):
            self.log_file.close()
