import torch
import numpy as np
import os
import matplotlib.pyplot as plt

############################################################################
# Setup
#############################################################################

# Connect to the GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)

# Get the current training directory
current_dir = os.getcwd()
training_dir = str(current_dir) + '/training/'  # change here !!!!!!!!
trained_dir = str(current_dir) + '/trained-model/'  # change here !!!!!!!!
os.makedirs(trained_dir, exist_ok=True)

############################################################################
# Training Parameters
#############################################################################

# Training
accumulation_steps = 8  # 1 = No accumulation
batch_size = 8
n_epoch = 10000  # 400
n_ax = 50 #int(n_epoch / 10)
learning_rate = 8e-5

total_loss_min = np.inf
image_shape = (1, 512, 512)
image_size = 512
image_dim = int(np.prod(image_shape))

# Alte diskrete Konfiguration
# feed_mod = 18 # number of condition values
# embedding_dim = 100
# num_classes = 18  # number of classes per condition
# num_conditions = 1

# class_table = torch.tensor([[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]])

# label_dict = {
#   0: 'FM_2_244', 1: 'FM_2_782',2: 'FM_2_879',
#   3: 'FM_3_055', 4: 'FM_3_156',5: 'FM_3_464',
#   6: 'FM_3_675', 7: 'FM_4_35',8: 'FM_4_045',
#   9: 'FM_4_109', 10: 'FM_4_546',11: 'FM_4_644',
#   12: 'FM_7_252', 13: 'FM_7_481',14: 'FM_8_288',
#   15: 'FM_8_331', 16: 'FM_8_556', 17: 'FM_8_565'
# }

# Neue kontinuierliche Konfiguration
num_conditions = 1
condition_dim = 1  # Dimension des kontinuierlichen Parameters
min_fm_value = 2.244  # Minimaler Feed-Modulation-Wert
max_fm_value = 8.565  # Maximaler Feed-Modulation-Wert

all_fm_values = [
    2.244, 2.782, 2.879,
    3.055, 3.156, 3.464,
    3.675, 4.35, 4.045,
    4.109, 4.546, 4.644,
    7.252, 7.481, 8.288,
    8.331, 8.556, 8.565
]

# Optional: Normalisierungsfunktion für den kontinuierlichen Parameter
def normalize_fm(fm_value):
    return (fm_value - min_fm_value) / (max_fm_value - min_fm_value)

##################################################################################################
# Image Generation
##################################################################################################
# Create a figure with a large size to accommodate multiple images
fig = plt.figure(figsize=(100, 100))

image_shape = (1, 512, 512)
image_size = 512
image_dim = int(np.prod(image_shape))

# Select process parameters from valid enteries for each parameter below
# Feedmod {valid enteries: "2.244", "8.565"}}
feed_mod_values = [ #2.244,
    2.782, 2.879,
    3.055, 3.156, 3.464,
    3.675, 4.35, 4.045,
    4.109, 4.546, 4.644, #5.2, 6.5,
    7.252, 7.481, 8.288,
    8.331, 8.556, 8.565
]


n_sampled_images = 1
# fm_dict = {
#   '2.244': 0, '2.782': 1, '2.879': 2,
#   '3.055': 3, '3.156': 4, '3.464': 5,
#   '3.675': 6, '4.35': 7, '4.045': 8,
#   '4.109': 9, '4.546': 10, '4.644': 11,
#   '7.252': 12, '7.481': 13, '8.288': 14,
#   '8.331': 15, '8.556': 16, '8.565': 17
# }



def trained_ddpm_name(ep):
    """
    Generate the name of the trained DDPM model based on the epoch number.
    """
    return 'ddpm_ebsd_fm_18_ep_' + str(ep) + '_bs_' + str(batch_size) + '.pth.tar'


