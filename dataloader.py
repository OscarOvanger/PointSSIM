#This file contains the datasets
import numpy as np
#from medmnist import PneumoniaMNIST, BreastMNIST
import gstools as gs
from skimage.filters import threshold_otsu
import os
import cv2

def load_nature_data():
    # Try except for mac vs windows
    windows_path = r'C:\Users\oo4663\OneDrive - NTNU\PhD\Paper_3\Experiment\nature_pics'
    mac_path = '/Users/oscaro/Library/CloudStorage/OneDrive-NTNU/PhD/Paper_3/Experiment/nature_pics'
    
    try:
        folder_path = windows_path
        os.listdir(folder_path)
    except FileNotFoundError:
        folder_path = mac_path

    # load all the .jpg files in the folder
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(folder_path, filename))
            # Convert the image to grayscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Use otsu thresholding to binarize the image
            thresh = threshold_otsu(img)
            img = img > thresh
            images.append(img)
    return images


def load_ragnar_data():
    windows_path = r'C:\Users\oo4663\OneDrive - NTNU\Ragnar_data\single_realiz\real_1.multipoint_large_const_ti_1.inc'
    mac_path = '/Users/oscaro/Library/CloudStorage/OneDrive-NTNU/Ragnar_data/single_realiz/real_1.multipoint_large_const_ti_1.inc'
    try:
        file_path = windows_path
        os.listdir(r'C:\Users\oo4663\OneDrive - NTNU\Ragnar_data\single_realiz')
    except FileNotFoundError:
        file_path = mac_path
    # Initialize a list to store the numbers
    data = []
    # Open and read the file
    with open(file_path, 'r') as file:
        for line in file:
            # Strip whitespace and check if the line contains numbers
            stripped_line = line.strip()        
            if stripped_line.isdigit() or stripped_line.replace(' ', '').isdigit():
                # Split the line into numbers, removing spaces
                numbers = [int(num) for num in stripped_line.split()]
                data.append(numbers)

    data_array = np.array(data)
    data_array_reshaped = data_array.reshape(8000000)
    data_x_y_z = np.zeros((400,400,50))
    for k in range(50):
        for j in range(400):
            for i in range(400):
                data_x_y_z[i,j,k] = data_array_reshaped[k*400*400 + j*400 + i]
    
    binary_data = np.zeros_like(data_x_y_z)
    binary_data[data_x_y_z == 1] = 0
    binary_data[data_x_y_z == 2] = 1

    reshaped = np.transpose(binary_data, (2, 0, 1))

    return reshaped

def load_TGRF_data(n_samples=50,range_short=0.02,range_long=0.05,grid_size=400,seed=150):
    def generate_data(n_samples,range_short,range_long,grid_size,seed):
        """Generate synthetic data for the example."""
        # Create the covariance model
        cov_model_short = gs.Gaussian(dim=2, var=1, len_scale=range_short)
        cov_model_long = gs.Gaussian(dim=2, var=1, len_scale=range_long)

        # Create the grid
        x = np.linspace(0, 1, grid_size)
        y = np.linspace(0, 1, grid_size)
        
        srf_short = gs.SRF(cov_model_short)
        srf_long = gs.SRF(cov_model_long) 
        samples = np.zeros((n_samples,grid_size,grid_size))
        for i in range(n_samples):
            seed = seed + i
            GRF_short = srf_short.structured((x,y),seed=seed)
            GRF_long = srf_long.structured((x,y),seed=seed+n_samples) + 2.0
            sample = np.zeros_like(GRF_short)
            sample[GRF_short>GRF_long] = 1
            samples[i] = sample
        return samples

    samples = generate_data(n_samples,range_short,range_long,grid_size,seed)
    return samples

def load_double_sided_TGRF(n_samples=50,range1=0.04,range2 = 0.04, grid_size=400,seed=1000,anis1 = 2.0, anis2 = 0.5, anlges1 = np.pi/4, angles2 = 0):
    cov_model_y = gs.Matern(dim=2, var=1, len_scale=range1,anis=anis1,angles=anlges1)
    srf_y = gs.SRF(cov_model_y)
    cov_model_x = gs.Matern(dim=2, var=1, len_scale=range2,anis=anis2,angles=angles2)
    srf_x = gs.SRF(cov_model_x)
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    samples = np.zeros((n_samples,grid_size,grid_size))
    for i in range(n_samples):
        seed = seed + i
        GRF_x = srf_x.structured((x,y),seed=seed)
        GRF_y = srf_y.structured((x,y),seed=seed)
        GRF = GRF_x + GRF_y
        sample = np.zeros_like(GRF)
        sample[GRF>2] = 1
        sample[GRF<-2] = 1
        samples[i] = sample
    return samples


def make_TGRF(n_samples,var,range,smoothness,grid_size,seed=2437):
    cov_model = gs.Matern(dim=2, var=var, len_scale=range,nu=smoothness)
    srf = gs.SRF(cov_model)
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    samples = np.zeros((n_samples,grid_size,grid_size))
    for i in range(n_samples):
        seed = seed + i
        GRF = srf.structured((x,y),seed=seed)
        sample = np.zeros_like(GRF)
        sample[GRF>1] = 1
        samples[i] = sample
        print("sample number: ", i, "out of ", n_samples)
    return samples