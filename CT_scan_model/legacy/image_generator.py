# Cast-forged AZ80 Magnesium Alloy Microstructure Image Generation

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from numpy.random import randn
import torchvision.utils
from torch.distributions import uniform
from mpl_toolkits.axes_grid1 import ImageGrid
import os
import copy
from .modules import *
from .configuration import *
import time

# === Einstellungen ===

save_path = str(current_dir) + '/Generated-Images/ddpm_1000_steps_v3_nocfg'

reverse_transforms = transforms.Compose([
    transforms.Lambda(lambda t: (t + 1) / 2),
    transforms.Lambda(lambda t: t * 255.),
])


def show_images(images, index, label):
    ax = fig.add_subplot(21, 1, index + 1, xticks=[], yticks=[])
    plt.gca().set_title(label)
    # ax.set_title(label,fontsize = 40)
    plt.imshow(images.cpu(), cmap='gray')
    plt.show()


def show_grids(images, feed_mod_value, epoch):
    fig = plt.figure(figsize=(5.12, 5.12), dpi=100)
    grid = ImageGrid(fig, 111, nrows_ncols=(1, 1), axes_pad=0)
    for ax, im in zip(grid, images.cpu().view(-1, image_size, image_size)):
        ax.axis('off')
        ax.imshow(im, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    plt.show()
    figname_dir = save_path + '/ep_' + str(epoch) + "/" + str(feed_mod_value) + '/'
    os.makedirs(os.path.dirname(figname_dir), exist_ok=True)
    figname_base = figname_dir + str(feed_mod_value)
    figname = figname_base
    counter = 1
    while os.path.exists(figname + '.png'):
        figname = figname_base + f"_{counter}"
        counter += 1
    fig.savefig(figname + '.png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def load_model(address):
    if torch.cuda.is_available() == True:
        checkpoint = torch.load(address)
        model.load_state_dict(checkpoint["model_state"])
        ema_model.load_state_dict(checkpoint["ema_model_state"])
        optimizer.load_state_dict(checkpoint["model_optimizer"])
    else:
        checkpoint = torch.load(address, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        ema_model.load_state_dict(checkpoint["ema_model_state"])
        optimizer.load_state_dict(checkpoint["model_optimizer"])


def load_model_mp(address):
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


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=image_size):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat[t])[:, None, None, None]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, fm, cfg_scale=0):  # change here !!!!!!!!
        model.eval()
        start_time = time.time()
        with torch.no_grad():
            x = torch.randn((n, 1, self.img_size, self.img_size)).to(device)
            for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(n) * i).long().to(device)
                predicted_noise = model(x, t, fm)  # change here !!!!!!!!
                if cfg_scale > 0:
                    empty_condition = torch.zeros((n, 1)).to(device)
                    uncond_predicted_noise = model(x, t, empty_condition)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (
                        x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(
                    beta) * noise
        end_time = time.time()
        print(f"Bildgenerierung in {end_time - start_time:.2f}s abgeschlossen")

        model.train()
        # x = (x.clamp(-1,1)+1)/2
        # x = (x*255).type(torch.uint8)
        return x

    def sample_with_intermediates(self, model, n, fm, cfg_scale=0, save_every=10, save_path=None):
        if save_path is None:
            save_path = str(os.getcwd()) + '/diffusion_progress'

        os.makedirs(save_path, exist_ok=True)

        model.eval()
        start_time = time.time()
        with torch.no_grad():
            x = torch.randn((n, 1, self.img_size, self.img_size)).to(device)

            # Speichere das initiale verrauschte Bild
            noisy_images = reverse_transforms(x.clone())

            for i, img in enumerate(noisy_images):
                fig = plt.figure(figsize=(5.12, 5.12), dpi=100)
                grid = ImageGrid(fig, 111, nrows_ncols=(1, 1), axes_pad=0)
                for ax, im in zip(grid, img.cpu().view(-1, image_size, image_size)):
                    ax.axis('off')
                    ax.imshow(im, cmap='gray')
                    ax.set_xticks([])
                    ax.set_yticks([])
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
                plt.show()
                fig.savefig(f"{save_path}/bild_{i}_schritt_{self.noise_steps}.png", bbox_inches='tight', pad_inches=0)
                plt.close(fig)

            for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(n) * i).long().to(device)
                predicted_noise = model(x, t, fm)  # change here !!!!!!!!
                if cfg_scale > 0:
                    empty_condition = torch.zeros((n, 1)).to(device)
                    uncond_predicted_noise = model(x, t, empty_condition)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (
                        x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(
                    beta) * noise

                # Speichere Zwischenergebnisse alle save_every Schritte
                if i % save_every == 0 or i == 1:
                    current_images = reverse_transforms(x.clone())

                    for j, img in enumerate(current_images):
                        fig = plt.figure(figsize=(5.12, 5.12), dpi=100)
                        grid = ImageGrid(fig, 111, nrows_ncols=(1, 1), axes_pad=0)
                        for ax, im in zip(grid, img.cpu().view(-1, image_size, image_size)):
                            ax.axis('off')
                            ax.imshow(im, cmap='gray')
                            ax.set_xticks([])
                            ax.set_yticks([])
                        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
                        # plt.show()
                        fig.savefig(f"{save_path}/bild_{j}_schritt_{i}.png", bbox_inches='tight', pad_inches=0)
                        plt.close(fig)
        end_time = time.time()
        print(f"Bildgenerierung mit Zwischenspeicherung in {end_time - start_time:.2f}s abgeschlossen")

        model.train()
        # x = (x.clamp(-1,1)+1)/2
        # x = (x*255).type(torch.uint8)
        return x

    def ddim_sample(self, model, n, fm, num_steps=50, cfg_scale=0, eta=0.0):
        """
        DDIM-Sampling mit optionaler Anzahl von Schritten und Rauschparameter eta

        Args:
            model: Das UNet-Modell
            n: Anzahl der zu generierenden Bilder
            fm: Conditional Embedding
            cfg_scale: Classifier-free guidance scale
            num_steps: Anzahl der DDIM-Schritte (deutlich weniger als bei DDPM)
            eta: Rauschparameter (0 = deterministisch, 1 = wie DDPM)
        """
        model.eval()
        with torch.no_grad():
            # Startrauschen erzeugen
            x = torch.randn((n, 1, self.img_size, self.img_size)).to(device)

            # Erzeuge Sequenz von Zeitschritten für DDIM
            skip = self.noise_steps // num_steps
            seq = list(range(0, self.noise_steps, skip))
            if seq[-1] != self.noise_steps - 1:
                seq.append(self.noise_steps - 1)

            # Iteriere von t=T nach t=0
            for i in range(len(seq) - 1, 0, -1):
                t = (torch.ones(n) * seq[i]).long().to(device)
                next_t = (torch.ones(n) * seq[i - 1]).long().to(device)

                # Vorhersage des Rauschens
                predicted_noise = model(x, t, fm)
                if cfg_scale > 0:
                    empty_condition = torch.zeros((n, 1)).to(device)
                    uncond_predicted_noise = model(x, t, empty_condition)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)

                # Extrahiere x_0 aus x_t und predicted_noise
                alpha_hat_t = self.alpha_hat[t][:, None, None, None]
                predicted_x0 = (x - torch.sqrt(1 - alpha_hat_t) * predicted_noise) / torch.sqrt(alpha_hat_t)

                # DDIM-Formel für den nächsten Zeitschritt
                alpha_hat_next_t = self.alpha_hat[next_t][:, None, None, None]

                # Interpolation zwischen deterministisch und stochastisch
                # eta=0: deterministisch (DDIM), eta=1: stochastisch (wie DDPM)
                sigma_t = eta * torch.sqrt(
                    (1 - alpha_hat_next_t) / (1 - alpha_hat_t) * (1 - alpha_hat_t / alpha_hat_next_t))

                # Rauschen hinzufügen, wenn eta > 0
                noise = torch.zeros_like(x)
                if eta > 0:
                    noise = torch.randn_like(x)

                # DDIM-Update-Schritt
                x = torch.sqrt(alpha_hat_next_t) * predicted_x0 + \
                    torch.sqrt(1 - alpha_hat_next_t - sigma_t ** 2) * predicted_noise + \
                    sigma_t * noise

        model.train()
        return x


device = device = torch.device("cuda:1")
print("Device: ", device)

model = UNet_conditional().to(device)
ema_model = UNet_conditional().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
diffusion = Diffusion(img_size=image_size)

current_dir = os.getcwd()


##################################################################################################
# Change conditions here
##################################################################################################

def sampleManyImages():
    epochs = [  # 600, 580, 560, 540, 500,
        180
    ]

    gen_fm_values = [
        # 2.244, 2.782, 2.879,
        # 3.055, 3.156, 3.464,
        # 3.675, 4.35, 4.045,
        # 4.109, 4.546, 4.644,
        5.2, 6.5,
        # 7.252, 7.481, 8.288,
        # 8.331, 8.556, 8.565
    ]

    sample_batch_size = 16
    n_samples_per_class = 1

    for ep in epochs:
        load_dir = os.path.join(str(current_dir) + '/trained-model-cont-aug-v3/', trained_ddpm_name(ep))
        print("Use Trained DDPM: ", load_dir)
        load_model_mp(load_dir)

        for feed_mod_value in gen_fm_values:
            for _ in range(n_samples_per_class):
                print("Generating image for feed_mod_value: ", feed_mod_value)
                with torch.no_grad():
                    t_fm_normalized = torch.tensor([[normalize_fm(feed_mod_value)]] * sample_batch_size).to(
                        device)  # Normalisiere und forme korrekt
                    # sampled_images = diffusion.sample(model, n_sampled_images, t_shp, t_loc, t_cr, t_sk, t_ht, t_ft, t_mag, cfg_scale=0)
                    # sampled_images = reverse_transforms(sampled_images)
                    ema_sampled_images = diffusion.sample(ema_model, sample_batch_size, t_fm_normalized,
                                                          cfg_scale=0)  # Change conditions here # 1 sample
                    # ema_sampled_images = diffusion.ddim_sample(ema_model, sample_batch_size, t_fm_normalized, num_steps=200, cfg_scale=4)
                    ema_sampled_images = reverse_transforms(ema_sampled_images)
                    # show_grids(sampled_images, test_labels,  e, label_dict)

                    for image in ema_sampled_images:
                        show_grids(image, feed_mod_value, ep)


def sampleWithSteps():
    # Beispielaufruf für die Zwischenschritt-Visualisierung
    load_dir = os.path.join(str(current_dir) + '/trained-model-cont-aug-v3/', trained_ddpm_name(180))
    print("Use Trained DDPM: ", load_dir)
    load_model_mp(load_dir)

    sample_batch_size = 1
    feed_mod_value = 2.782
    print("Sample with steps: ", feed_mod_value)
    t_fm_normalized = torch.tensor([[normalize_fm(feed_mod_value)]] * sample_batch_size).to(
        device)  # Normalisiere und forme korrekt

    save_path = str(current_dir) + '/Generated-Images/ddpm_v3_nocfg_progress'
    ema_sampled_images = diffusion.sample_with_intermediates(
        ema_model,
        sample_batch_size,
        t_fm_normalized,
        cfg_scale=0,
        save_every=1,
        save_path=save_path
    )

    # ema_sampled_images = reverse_transforms(ema_sampled_images)
    # for image in ema_sampled_images:
    #    show_grids(image, feed_mod_value, 100000)


if __name__ == "__main__":
    # sampleWithSteps()
    sampleManyImages()
