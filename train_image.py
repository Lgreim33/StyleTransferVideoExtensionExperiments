import os
import argparse
import time
import json
import zipfile
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
import torchvision.transforms as transforms
import boto3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models import StyleTransferModel, PatchGAN, StyleTransferLoss, gradient_penalty




def parse_args():
    parser = argparse.ArgumentParser(description="Image Style Transfer Training Job (PatchGAN)")
    
    # Storage args
    parser.add_argument('--bucket_name', type=str, required=True, help="S3 Bucket name")
    parser.add_argument('--s3_output_dir', type=str, default="checkpoints", help="S3 folder to save results")
    
    # Standard Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=8)
    
    # Model Weights (Lambdas) - No temporal lambda needed here!
    parser.add_argument('--lambda_gan', type=float, default=0.1)
    parser.add_argument('--lambda_gp', type=float, default=10.0)
    parser.add_argument('--lambda_content', type=float, default=1.0)
    parser.add_argument('--lambda_style', type=float, default=10.0)
    parser.add_argument('--lambda_ssim', type=float, default=6.0)
    
    # Model tracking
    parser.add_argument('--base_model_name', type=str, default="", help="Optional: Name of a pre-trained model to fine-tune")
    parser.add_argument('--run_name', type=str, default="patchgan_image", help="Prefix for saved files")
    
    return parser.parse_args()


class StyleContentDataset(Dataset):
    """Loads a single content image and a single style image."""
    def __init__(self, content_dir, style_dir):
        self.content_images = [os.path.join(content_dir, img) for img in os.listdir(content_dir) if img.endswith(('.jpg', '.png', '.jpeg'))]
        
        self.style_images = []
        for subdir in os.listdir(style_dir):
            subdir_path = os.path.join(style_dir, subdir)
            if os.path.isdir(subdir_path):
                style_images_in_subdir = [os.path.join(subdir_path, img) for img in os.listdir(subdir_path) if img.endswith(('.jpg', '.png', '.jpeg'))]
                self.style_images.extend(style_images_in_subdir)
        
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.CenterCrop((256,256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return max(len(self.content_images), len(self.style_images))

    def __getitem__(self, idx):
        try:
            content_img_path = self.content_images[idx % len(self.content_images)]
            style_img_path = self.style_images[idx % len(self.style_images)]
            
            content_image = Image.open(content_img_path).convert("RGB")
            style_image = Image.open(style_img_path).convert("RGB")

            content_image = self.transform(content_image)
            style_image = self.transform(style_image)

            return content_image, style_image
        except OSError:
            # Defensive catch for corrupted image streams
            import random
            print(f"Warning: Corrupted image detected at index {idx}. Fetching replacement...")
            return self.__getitem__(random.randint(0, len(self) - 1))


def setup_s3_data(bucket_name):
    """Downloads and extracts the datasets to the local emptyDir."""
    s3 = boto3.client('s3', 
                      endpoint_url=os.environ.get("S3_ENDPOINT"),
                      aws_access_key_id=os.environ.get("S3_ACCESS_KEY"),
                      aws_secret_access_key=os.environ.get("S3_SECRET_KEY"))
    
    data_dir = "/workspace/data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Download data
    for zip_file in ["wikiart.zip", "unlabeled2017.zip"]:
        local_zip = os.path.join(data_dir, zip_file)
        print(f"Downloading {zip_file}...")
        try:
            s3.download_file(bucket_name, zip_file, local_zip)
            print(f"Extracting {zip_file}...")
            with zipfile.ZipFile(local_zip, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            os.remove(local_zip)
        except Exception as e:
            print(f"Could not download {zip_file}: {e}")
        
    return data_dir


def save_and_upload_results(args, generator, discriminator, g_losses, d_losses, val_losses, s3_client):
    """Saves the weights, metadata, convergence JSON, and a PNG graph, then pushes to S3."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = "/workspace/output"
    os.makedirs(out_dir, exist_ok=True)
    
    # Save model weights
    model_filename = f"{args.run_name}_{timestamp}.pth"
    model_path = os.path.join(out_dir, model_filename)
    torch.save({
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'args': vars(args)
    }, model_path)
    
    # Save Metadata
    txt_filename = f"{args.run_name}_{timestamp}_metadata.txt"
    txt_path = os.path.join(out_dir, txt_filename)
    with open(txt_path, 'w') as f:
        f.write(f"Run Name: {args.run_name}\n")
        f.write(f"Date: {timestamp}\n")
        f.write("--- Hyperparameters ---\n")
        f.write(f"Lambda GAN: {args.lambda_gan}\n")
        f.write(f"Lambda GP: {args.lambda_gp}\n")
        f.write(f"Lambda Content: {args.lambda_content}\n")
        f.write(f"Lambda Style: {args.lambda_style}\n")
        f.write(f"Lambda SSIM: {args.lambda_ssim}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write("--- Final Convergence ---\n")
        f.write(f"Final G Loss: {g_losses[-1] if g_losses else 'N/A':.4f}\n")
        f.write(f"Final D Loss: {d_losses[-1] if d_losses else 'N/A':.4f}\n")
        f.write(f"Best Val Loss: {min(val_losses) if val_losses else 'N/A'}\n")

    # Save JSON Array
    json_filename = f"{args.run_name}_{timestamp}_convergence.json"
    json_path = os.path.join(out_dir, json_filename)
    with open(json_path, 'w') as f:
        json.dump({"G_losses": g_losses, "D_losses": d_losses, "Val_losses": val_losses}, f)

    # Generate Graph
    graph_filename = f"{args.run_name}_{timestamp}_loss_curve.png"
    graph_path = os.path.join(out_dir, graph_filename)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(g_losses, label='Generator Loss', color='blue', alpha=0.7)
    ax1.plot(d_losses, label='Discriminator Loss', color='red', alpha=0.7)
    ax1.set_title(f"Training Loss (Run: {args.run_name} | GAN $\lambda$: {args.lambda_gan})")
    ax1.set_xlabel('Batches')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    if val_losses:
        ax2.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', color='green', marker='o')
        ax2.set_title('Validation Loss per Epoch')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.5)
        
    plt.tight_layout()
    plt.savefig(graph_path, dpi=150)
    plt.close()

    # Upload to bucket
    print("\nUploading results to S3...")
    upload_files = [
        (model_path, model_filename), 
        (txt_path, txt_filename), 
        (json_path, json_filename),
        (graph_path, graph_filename) 
    ]
    
    for file_path, file_name in upload_files:
        s3_client.upload_file(file_path, args.bucket_name, f"{args.s3_output_dir}/{file_name}")
        print(f"Uploaded {file_name}")


def train_PatchGAN_image(
    generator, discriminator, criterion,
    dataloader, val_dataloader,
    optimizer_G, optimizer_D, scheduler_G,
    device, args
):
    epochs = args.epochs
    lambda_gan = args.lambda_gan
    lambda_gp = args.lambda_gp
    training_ratio = 5

    scaler_D = GradScaler()
    scaler_G = GradScaler()

    G_losses = []
    D_losses = []
    val_loss_history = []

    generator.train()
    discriminator.train()
    min_val_loss = float('inf')

    print(f"\n==============================================")
    print(f"Starting Image Run: {args.run_name}")
    print(f"==============================================\n")

    for epoch in range(epochs):
        count = 0
        start_time = time.time()

        print(f"--- Starting Epoch {epoch+1}/{epochs} ---")
        for content, style in dataloader:
            
            if count % 50 == 0:
                print(f"Epoch {epoch+1} [{count}/{len(dataloader)}]")
                
            content = content.to(device)
            style = style.to(device)

            # -----------------------------------------------
            # Train Discriminator
            # -----------------------------------------------
            optimizer_D.zero_grad()

            with autocast():
                generated_image, style_feats, content_feats = generator(content, style)
                pred_real = discriminator(style)
                pred_fake_d = discriminator(generated_image.detach())
                adv_loss_D = -torch.mean(pred_real) + torch.mean(pred_fake_d)

            with autocast(enabled=False):
                gp = gradient_penalty(discriminator, style.float(), generated_image.float(), device)

            loss_D = adv_loss_D + (lambda_gp * gp)

            scaler_D.scale(loss_D).backward()
            scaler_D.step(optimizer_D)
            scaler_D.update()
            
            D_losses.append(loss_D.item())

            # -----------------------------------------------
            # Train Generator
            # -----------------------------------------------
            if count % training_ratio == 0:
                optimizer_G.zero_grad()

                with autocast():
                    # Forward pass already done above, but we need gradients for G
                    generated_image, style_feats, content_feats = generator(content, style)
                    
                    loss_c_s_ssim, content_loss, edge_loss, style_loss = criterion(
                        generated_image, content, content_feats, style_feats
                    )

                    pred_fake = discriminator(generated_image)
                    loss_gan = -torch.mean(pred_fake)

                    total_loss_G = loss_c_s_ssim + (lambda_gan * loss_gan)

                scaler_G.scale(total_loss_G).backward()
                scaler_G.step(optimizer_G)
                scaler_G.update()
                
                G_losses.append(total_loss_G.item())

            count += 1

        scheduler_G.step()

        # -----------------------------------------------
        # Validation
        # -----------------------------------------------
        avg_val = float('nan')
        if val_dataloader is not None:
            print(f"Running validation for Epoch {epoch+1}...")
            generator.eval()
            val_losses_epoch = []
            
            with torch.no_grad():
                for v_content, v_style in val_dataloader:
                    v_content = v_content.to(device); v_style = v_style.to(device)
                    gen, s_feats, c_feats = generator(v_content, v_style)
                    v_loss, _, _, _ = criterion(gen, v_content, c_feats, s_feats)
                    val_losses_epoch.append(v_loss.item())
            
            avg_val = sum(val_losses_epoch) / max(1, len(val_losses_epoch))
            val_loss_history.append(avg_val)
            
            print(f"Epoch {epoch+1} Summary: Val Loss = {avg_val:.6f} | Time = {(time.time()-start_time)/60:.2f}m\n")

            if avg_val < min_val_loss:
                min_val_loss = avg_val
                
            generator.train()

    print("Training Complete!")
    return G_losses, D_losses, val_loss_history


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    data_dir = setup_s3_data(args.bucket_name)
    
    # Dataloaders
    coco_path = os.path.join(data_dir, "unlabeled2017")
    wikiart_path = os.path.join(data_dir, "wikiart")
    dataset = StyleContentDataset(coco_path, wikiart_path)
    train_dataset, validation_dataset, _ = random_split(dataset, [0.8, 0.1, 0.1])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    # Initialize Models
    generator = StyleTransferModel(DeepDecoderTrue=True, standardize_encoder_inputs=True, UseAttention=True).to(device)
    discriminator = PatchGAN().to(device)
    
    # Freeze Encoder in Generator
    for param in generator.encoder.model.parameters():
        param.requires_grad = False

    # Initialize Criterion and Optimizers
    criterion = StyleTransferLoss(
        generator.encoder, 
        lambda_content=args.lambda_content,
        lambda_style=args.lambda_style,
        lambda_ssim=args.lambda_ssim
    ).to(device)
    
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.0, 0.9))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.0, 0.9))
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=10, gamma=0.5)

    # Train
    g_losses, d_losses, val_losses = train_PatchGAN_image(
        generator, discriminator, criterion, train_loader, val_loader, 
        optimizer_G, optimizer_D, scheduler_G, device, args
    )
    
    # Save and Upload
    s3_client = boto3.client('s3', 
                      endpoint_url=os.environ.get("S3_ENDPOINT"),
                      aws_access_key_id=os.environ.get("S3_ACCESS_KEY"),
                      aws_secret_access_key=os.environ.get("S3_SECRET_KEY"))
    
    save_and_upload_results(args, generator, discriminator, g_losses, d_losses, val_losses, s3_client)
    print("Job complete!")