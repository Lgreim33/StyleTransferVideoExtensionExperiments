import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random
from pytorch_msssim import ssim
import torch.nn.functional as F
import time
from model import *



# Helper function to quickly print out the results from the metric tests
def print_tests(tupple):
    print(f'SSIM : {tupple[0]}   MSE : {tupple[1]}     Average Time: {tupple[2]}')


'''
    This is a brute force method of testing, mostly because I wanted to keep tests for metrics seperate. 
    Particuarlly the time test, as I want that data point to be as pure as possible
'''

# test average inference time
def time_test(styleTransferModel, loader,device):
    
    total_time = 0
    #calculate the total ssim from the random sample
    with torch.no_grad():
        for content,style in loader:
            # Generate stylized image
            time_now = time.time()
            _,_,_ = styleTransferModel(content.to(device), style.to(device))
            total_time += time.time() - time_now


    return total_time / loader.__len__()

# Test for calculating average SSIM and MSE, take the model, device, and the random dataloader
def test_ssim_mse(styleTransferModel, loader,device):
    
    total_ssim = 0
    total_mse = 0

    # Calculate the total ssim and mse from the random sample
    with torch.no_grad():
        for content,style in loader:
            # Generate stylized image
            generated_image,_,_ = styleTransferModel(content.to(device), style.to(device))

            total_ssim += ssim(generated_image, content.to(device), data_range=1.0, size_average=True)
            total_mse += F.mse_loss(generated_image, content.to(device), reduction='mean').item()

    # Return average ssim and mse
    return total_ssim / loader.__len__() , total_mse / loader.__len__()




# Performs both the graphing for the subjective test, and the tests to retreive the metrics for all images    
def test(model_name, loader, loader_loss, device):

    # Retrieve desired model
    styleTransferModel = StyleTransferModel()
    # Load checkpoint and filter unexpected keys (e.g. 'mean','std') that can appear in some saved state dicts
    checkpoint = torch.load(model_name, map_location='cpu')
    # support checkpoints that either store a dict under 'model_state_dict' or are the raw state_dict
    saved_state = checkpoint.get("model_state_dict", checkpoint)
    model_state = styleTransferModel.state_dict()
    # Keep only keys that exist in the model; this prevents "Unexpected key(s)" errors
    filtered_state = {k: v for k, v in saved_state.items() if k in model_state}
    unexpected_keys = set(saved_state.keys()) - set(filtered_state.keys())
    if unexpected_keys:
        print(f"Warning: ignoring unexpected keys in checkpoint: {unexpected_keys}")
    # Load filtered state dict with strict=False to allow missing keys if any
    styleTransferModel.load_state_dict(filtered_state, strict=False)
    styleTransferModel.to(device)
    styleTransferModel.eval()


    # Lists for content and style tensors
    content_images = []
    style_images = []
    original_content = []
    original_style = []

    # Save the original content to their own arrays as PIL images for later, otherwise put the style and and content images in their own seperate lists
    for content_tensor, style_tensor in loader:
        content_images.append(content_tensor)
        style_images.append(style_tensor)
        original_content.append(np.array(transforms.ToPILImage()(content_tensor[0])))
        original_style.append(np.array(transforms.ToPILImage()(style_tensor[0])))

    # Generate placeholder grid
    grid_size = len(content_images)
    num_styles = len(style_images)
    grid_images = [[None for _ in range(num_styles)] for _ in range(grid_size)]


    # Generate all combinations of of the passed content and style images
    with torch.no_grad():
        for i, content_tensor in enumerate(content_images):
            for j, style_tensor in enumerate(style_images):
                # Generate stylized image
                generated_image,_,_ = styleTransferModel(content_tensor.to(device), style_tensor.to(device))
                stylized_image = transforms.ToPILImage()(generated_image.squeeze(0))

                # Save to grid
                grid_images[i][j] = np.array(stylized_image)

    # Plot the grid with original content and style images as headers, with one additional row on the x and y for the header images
    fig, axes = plt.subplots(grid_size + 1, num_styles + 1, figsize=(20, 20))

    # Add style images at the x-axis
    for j, img in enumerate(original_style):
        axes[0, j + 1].imshow(img)
        axes[0, j + 1].axis('off')
        axes[0, j + 1].set_title(f"Style {j+1}", fontsize=8)

    # Add content images on the y-axis
    for i, img in enumerate(original_content):
        axes[i + 1, 0].imshow(img)
        axes[i + 1, 0].axis('off')
        axes[i + 1, 0].set_ylabel(f"Content {i+1}", fontsize=8, rotation=90)

    # Add stylized images in the grid
    for i, row in enumerate(grid_images):
        for j, img in enumerate(row):
            axes[i + 1, j + 1].imshow(img)
            axes[i + 1, j + 1].axis('off')


    axes[0,0].axis("off")

    plt.tight_layout()
    plt.show()


    # Get model result metrics
    ssim,mse = test_ssim_mse(styleTransferModel,loader_loss,device)
    t = time_test(styleTransferModel,loader_loss,device)

    return ssim,mse,t


# Given the dataset, select some number of random spamples
def selectRandomImages(dataset, num_samples=10):

    # get x random sample indices
    indices = random.sample(range(len(dataset)), num_samples)
    random_samples = [dataset[idx] for idx in indices]
    return random_samples

# Dataset class for the random samples we retreived in selectRandomImages
class RandomSampleDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
    
def main():

    # Dataset paths
    # Dataset paths, relative to this folder
    coco_path = "DataSets/unlabeled2017/"
    wikiart_path = "DataSets/wikiart/"
    
    # Try setting device to gpu for reduced inference time
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get test dataset
    test_set = StyleContentDataset(content_dir=coco_path,style_dir=wikiart_path)
    
    
    # This will be for displaying 25 combinations of 5 random content and 5 random style images
    random_samples = selectRandomImages(test_set, num_samples=1)
    random_loader = DataLoader(RandomSampleDataset(random_samples), batch_size=1, shuffle=False)

    # This loader will be used to calculate the average ssim and mse of 1000 random images, as well as the average ammount of time it takes to infer a stylized imaage
    random_samples_loss = selectRandomImages(test_set, num_samples=1)
    random_loader_loss = DataLoader(RandomSampleDataset(random_samples_loss), batch_size=1, shuffle=False)




    # The two finalists will now be tested for general performance, however, this should be replaced with paths to whatever model you have trained
    print_tests(test('40000SampleTraining_SSIM=1.pth',random_loader,random_loader_loss,device))
    #print_tests(test('cbam_AdaIN_SSIM=10_StyleScale=10(Epochs=100).pth',random_loader,random_loader_loss,device))  
    
main()