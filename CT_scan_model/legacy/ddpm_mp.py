import torch.optim as optim
from torchvision import datasets, transforms
import copy
from tqdm import tqdm
import time
import math

from .utils import *

# NOTE: The original project used a 1x512x512 grayscale model.
# For polar CT training we use a new UNet that supports:
# - non-square polar tensors (R x Theta)
# - 2-channel input (image + mask)
# - categorical + continuous conditioning
from ..modules_polar_ct import UNet_conditional_polar
import torch
import gc


# === Environment Settings ===
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# === DDPM Settings ===
start_epoch = 0
load_epoch = 0

# This script is kept for backwards compatibility.
# The training pipeline for polar CT will be implemented separately.
RUN_TRAINING = False

# Regularisieren der Input Bilder

# Normalize((0.4433),(0.1038))
# grid_transform = transforms.Compose([
#    transforms.ToTensor(),
#    transforms.Grayscale(),
#    transforms.Normalize((0.5),(0.5))
#    ])

grid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(),
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

# NOTE: The training pipeline will be updated later to use the BatteryCTDataset
# and cartesian->polar conversion. The original ImageFolder + RandomCrop setup
# is kept for now as placeholder.
whole_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(),
    transforms.Lambda(lambda t: (t * 2) - 1)
])

if RUN_TRAINING:
    # === EINSTELLUNGEN ====
    train_dataset = datasets.ImageFolder(training_dir, transform=whole_transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

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

    


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        # For polar CT we use a fixed (R, Theta) from configuration.
        from .configuration import POLAR_R_MODEL, POLAR_THETA_BINS
        self.r_size = POLAR_R_MODEL
        self.theta_size = POLAR_THETA_BINS

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat[t])[:, None, None, None]

        # If the model input contains a mask channel (image, mask), we only
        # diffuse the image channel and keep the mask fixed.
        if x.dim() == 4 and x.shape[1] == 2:
            x_img, x_mask = x[:, :1], x[:, 1:]
            epsilon = torch.randn_like(x_img)
            x_noised = sqrt_alpha_hat * x_img + sqrt_one_minus_alpha_hat * epsilon
            return torch.cat([x_noised, x_mask], dim=1), epsilon

        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, cond, cfg_scale=0):
        model.eval()
        with torch.no_grad():
            from .configuration import POLAR_R_MODEL, POLAR_THETA_BINS

            # Diffuse only the image channel; keep the mask fixed.
            x_img = torch.randn((n, 1, POLAR_R_MODEL, POLAR_THETA_BINS), device=device)
            x_mask = torch.ones((n, 1, POLAR_R_MODEL, POLAR_THETA_BINS), device=device)
            for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(n) * i).long().to(device)

                model_in = torch.cat([x_img, x_mask], dim=1)
                predicted_noise = model(model_in, t, cond)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(model_in, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x_img)
                else:
                    noise = torch.zeros_like(x_img)
                x_img = 1 / torch.sqrt(alpha) * (
                        x_img - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(
                    beta) * noise
        model.train()
        return x_img

if RUN_TRAINING:
    # Umleiten des Standardoutputs zu Beginn des Programms
    sys.stdout = Logger()
    sys.stderr = Logger()

    # Ab hier werden alle print-Ausgaben in die Datei UND auf die Konsole ausgegeben

    # GPU leeren
    torch.cuda.empty_cache()
    gc.collect()

    # Gesamten verfügbaren Speicher reservieren
    device = torch.cuda.current_device()

    model = UNet_conditional_polar().to(device)
    # model.apply(weights_init)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    mse = nn.MSELoss()
    diffusion = Diffusion()
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
        load_dir = os.path.join(trained_dir, trained_ddpm_name(load_epoch))
        print("Use Trained DDPM: ", load_dir)
        load_model(load_dir, model, ema_model, optimizer)

    print('Start training ...')

    for e in tqdm(range(start_epoch + 1, n_epoch + 1)):
        loss_epoch = 0
        optimizer.zero_grad()  # Gradienten zu Beginn jeder Epoche zurücksetzen

        for i, (images, labels) in enumerate(train_loader):
            iter_start = time.time()
            # images = aug_transform(images)
            images = images.to(device)

            # Placeholder: conditioning handled later via BatteryCTDataset.

            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)

            torch.cuda.empty_cache()  ### EMPTY Cache

            # TODO(training pipeline): replace `cond=None` with proper (cat, cont) condition tuple.
            predicted_noise = model(x_t, t, None)
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
            # Placeholder: sampling unconditional; conditioning integrated later.
            ema_sampled_images = diffusion.sample(ema_model, n_sampled_images, None, cfg_scale=0)
            ema_sampled_images = reverse_transforms(ema_sampled_images)
            show_grids(ema_sampled_images, "uncond", e)
            os.makedirs('Generated-Images', exist_ok=True)

    print(torch.cuda.memory_summary(device=device))
