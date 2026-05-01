import torch.optim as optim
from torchvision import datasets, transforms
import copy
from .utils import *
from tqdm import tqdm
import time
import math

from .configuration_EBSD import *
from modules_EBSD import *
import torch

# === Environment Settings ===
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# === DDPM Settings ===
start_epoch = 0
load_epoch = 0

# Regularisieren der Input Bilder

# Normalize((0.4433),(0.1038))
# grid_transform = transforms.Compose([
#    transforms.ToTensor(),
#    transforms.Grayscale(),
#    transforms.Normalize((0.5),(0.5))
#    ])

grid_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Grayscale(),
    transforms.Lambda(lambda t: (t * 2) - 1)
])

# grid_dataset = datasets.ImageFolder(grid_dir, transform = grid_transform)
# ---
# whole_transform = transforms.Compose([
#    transforms.ToTensor(),
#    transforms.Grayscale(),
#    transforms.RandomCrop(256),
#    transforms.Normalize((0.5),(0.5))
#    ])

whole_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Grayscale(),
    transforms.RandomCrop(512),
    transforms.Lambda(lambda t: (t * 2) - 1)
])

# === EINSTELLUNGEN ====
train_dataset = datasets.ImageFolder(training_dir, transform=whole_transform)

# ---
aug_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
])
# ---
reverse_transforms = transforms.Compose([
    transforms.Lambda(lambda t: (t + 1) / 2),
    transforms.Lambda(lambda t: t * 255.),
])

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


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

    ##################################################################################################
    # Change conditions here
    ##################################################################################################

    def sample(self, model, n, fm, cfg_scale=0):  # change here !!!!!!!!
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(device)
            for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(n) * i).long().to(device)
                predicted_noise = model(x, t, fm)  # change here !!!!!!!!
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
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
        model.train()
        return x

# Umleiten des Standardoutputs zu Beginn des Programms
sys.stdout = Logger()
sys.stderr = Logger()

# Ab hier werden alle print-Ausgaben in die Datei UND auf die Konsole ausgegeben

import torch
import gc

# GPU leeren
torch.cuda.empty_cache()
gc.collect()

model = UNet_conditional().to(device)
# model.apply(weights_init)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
mse = nn.MSELoss()
diffusion = Diffusion(img_size=image_size)
l = len(train_loader)
ema = EMA(0.995)
ema_model = copy.deepcopy(model).eval().requires_grad_(False)

# Nach dem Definieren deines Modells
if torch.cuda.device_count() > 1:
    print(f"Training auf {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)
    ema_model = nn.DataParallel(ema_model)
model = model.to(device)
ema_model = ema_model.to(device)

if load_epoch > 0:
    load_dir = os.path.join(trained_dir, trained_ddpm_name(120))
    load_model(load_dir, model, ema_model, optimizer)
    print("Use Trained DDPM: ", load_dir)

print('Start training ...')

for e in tqdm(range(start_epoch + 1, n_epoch + 1)):
    loss_epoch = 0
    optimizer.zero_grad()  # Gradienten zu Beginn jeder Epoche zurücksetzen

    for i, (images, labels) in enumerate(train_loader):
        iter_start = time.time()
        # images = aug_transform(images)
        images = images.to(device)

        # fm = torch.tensor([normalize_fm(float(fm)) for fm in all_fm_values]).to(device) # Normalisierung auf 0 bis 1
        fm = torch.zeros(labels.size(0), 1).to(device)  # Ein Wert pro Bild im Batch
        for j, label in enumerate(labels):
            # Hier musst du die Abbildung von Label zu FM-Wert definieren
            # Beispiel: Annahme, dass das Label direkt den Index im all_fm_values darstellt
            fm_value = all_fm_values[label.item() % len(all_fm_values)]  # Sicherstellen, dass es in Grenzen bleibt
            fm[j, 0] = normalize_fm(float(fm_value))

        t = diffusion.sample_timesteps(images.shape[0]).to(device)
        x_t, noise = diffusion.noise_images(images, t)

        torch.cuda.empty_cache()  ### EMPTY Cache

        predicted_noise = model(x_t, t, fm)
        loss = mse(noise, predicted_noise)

        # Verlust für die Akkumulation skalieren
        loss = loss / accumulation_steps

        # Gradientenakkumulation
        loss.backward()

        if (i + 1) % accumulation_steps == 0 or i == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()
            ema.step_ema(ema_model, model)
            print(f"Epoch {e} | AccBatch {(i+1)/accumulation_steps}/{len(train_loader)/accumulation_steps}")
        loss_epoch += loss.item()
        iter_end = time.time()
        if i % 10 == 0:
            print(f"Epoch {e} | Batch {i}/{len(train_loader)} | Iter Time: {iter_end - iter_start:.2f}s")

    print('Epoch: [%d/%d]: Loss: %.3f' % (
        (e), n_epoch, loss_epoch))
    save_loss_to_csv(e, loss_epoch, filepath=trained_dir,
                     filename=f"loss_{trained_ddpm_name(n_epoch).replace('.pth.tar', '.csv')}")

    save_epoch = (n_ax * math.ceil(e / n_ax))
    print("Start saving: ", save_epoch)
    save_dir = os.path.join(trained_dir, trained_ddpm_name(save_epoch))
    save_model(save_dir, model, ema_model, optimizer)
    print("End saving")

    if e % n_ax == 0:
        # t_fm = torch.randfloat([n_sampled_images, 1], min_fm_value, max_fm_value).to(device)
        # Statt torch.randfloat, was nicht existiert:
        t_fm = torch.rand([n_sampled_images, 1]).to(device) * (max_fm_value - min_fm_value) + min_fm_value
        t_fm_normalized = torch.tensor([[normalize_fm(val.item())] for val in t_fm]).to(device)
        ema_sampled_images = diffusion.sample(ema_model, n_sampled_images, t_fm_normalized, cfg_scale=0)
        ema_sampled_images = reverse_transforms(ema_sampled_images)
        show_grids(ema_sampled_images, str(t_fm.item()), e)
        os.makedirs('Generated-Images', exist_ok=True)

print(torch.cuda.memory_summary(device=device))
