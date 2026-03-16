
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
from PIL import Image
import os
import random




def load_raft_model(device):
    """Load pretrained RAFT-Small for optical flow estimation. Frozen."""
    weights = Raft_Small_Weights.DEFAULT
    raft = raft_small(weights=weights).to(device).eval()
    for param in raft.parameters():
        param.requires_grad = False
    return raft, weights.transforms()


def warp_with_flow(image, flow):
    """
    Warp an image using optical flow (backward warp via grid_sample).
    
    Args:
        image: (B, C, H, W) tensor
        flow: (B, 2, H, W) optical flow
    
    Returns:
        warped: (B, C, H, W) warped image
    """
    B, C, H, W = image.shape
    
    # Create base grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=image.device, dtype=torch.float32),
        torch.arange(W, device=image.device, dtype=torch.float32),
        indexing='ij'
    )
    grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)  # (B, H, W)
    grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)

    # Add flow to grid
    new_x = grid_x + flow[:, 0, :, :]  # flow[:,0] is horizontal displacement
    new_y = grid_y + flow[:, 1, :, :]  # flow[:,1] is vertical displacement

    # Normalize to [-1, 1] for grid_sample
    new_x = 2.0 * new_x / (W - 1) - 1.0
    new_y = 2.0 * new_y / (H - 1) - 1.0

    grid = torch.stack([new_x, new_y], dim=-1)  # (B, H, W, 2)

    warped = F.grid_sample(image, grid, mode='bilinear', padding_mode='border', align_corners=True)
    return warped




class FramePairDataset(Dataset):
    """
    Loads consecutive frame pairs (frame_t, frame_t+1) from YouTube-VIS video sequences,
    paired with a random style image each time.

    Expected structure:
        video_root/
            JPEGImages/
                <video_id>/
                    00000.jpg
                    00001.jpg
                    ...
                <video_id>/
                    ...

    If your frames are directly under video_root/<video_id>/, set jpeg_subdir=None.

    Style images use the wikiart folder structure (artist subfolders).
    """

    def __init__(self, video_root, style_dir, jpeg_subdir="JPEGImages", transform=None):
        super().__init__()

        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        # Locate the JPEGImages folder
        if jpeg_subdir is not None:
            frames_root = os.path.join(video_root, jpeg_subdir)
        else:
            frames_root = video_root

        assert os.path.isdir(frames_root), f"Frames root not found: {frames_root}"

        # Collect all consecutive frame pairs from all video sequences
        self.frame_pairs = []
        for video_id in sorted(os.listdir(frames_root)):
            video_dir = os.path.join(frames_root, video_id)
            if not os.path.isdir(video_dir):
                continue

            # Sort frames by filename to guarantee temporal order
            frames = sorted([
                os.path.join(video_dir, f)
                for f in os.listdir(video_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])

            # Create consecutive pairs
            for i in range(len(frames) - 1):
                self.frame_pairs.append((frames[i], frames[i + 1]))

        # Collect all style images (wikiart structure: subfolders per artist)
        self.style_images = []
        for subdir in os.listdir(style_dir):
            subdir_path = os.path.join(style_dir, subdir)
            if os.path.isdir(subdir_path):
                for img in os.listdir(subdir_path):
                    if img.lower().endswith(('.jpg', '.png', '.jpeg')):
                        self.style_images.append(os.path.join(subdir_path, img))

        assert len(self.frame_pairs) > 0, f"No frame pairs found in {frames_root}"
        assert len(self.style_images) > 0, f"No style images found in {style_dir}"

        print(f"YouTubeVISFramePairDataset: {len(self.frame_pairs)} frame pairs from {frames_root}")
        print(f"  Style images: {len(self.style_images)}")

    def __len__(self):
        return len(self.frame_pairs)

    def __getitem__(self, idx):
        frame_t_path, frame_tp1_path = self.frame_pairs[idx]
        style_path = self.style_images[random.randint(0, len(self.style_images) - 1)]

        frame_t = self.transform(Image.open(frame_t_path).convert("RGB"))
        frame_tp1 = self.transform(Image.open(frame_tp1_path).convert("RGB"))
        style = self.transform(Image.open(style_path).convert("RGB"))

        return frame_t, frame_tp1, style
    

def estimate_bidirectional_flow(raft_model, raft_transforms, img1, img2):
    """
    Estimate forward (img1->img2) and backward (img2->img1) optical flow.

    Args:
        img1, img2: (B, 3, H, W) tensors in [0, 1]
    Returns:
        flow_fwd: (B, 2, H, W)  forward flow
        flow_bwd: (B, 2, H, W)  backward flow
    """
    img1_uint8 = (img1 * 255).to(torch.uint8)
    img2_uint8 = (img2 * 255).to(torch.uint8)

    img1_t, img2_t = raft_transforms(img1_uint8, img2_uint8)

    with torch.no_grad():
        flow_fwd = raft_model(img1_t, img2_t)[-1]
        flow_bwd = raft_model(img2_t, img1_t)[-1]

    return flow_fwd, flow_bwd


def compute_flow_confidence(flow_fwd, flow_bwd, alpha=0.01, beta=0.5):
    """
    Compute a per-pixel confidence mask based on forward-backward flow consistency.

    The idea: if we warp the backward flow using the forward flow, a reliable pixel
    should satisfy:  flow_fwd(x) + flow_bwd(x + flow_fwd(x)) ≈ 0

    Pixels where this does NOT hold are likely occluded, out-of-bounds, or have
    unreliable flow estimates.

    Args:
        flow_fwd: (B, 2, H, W) forward optical flow
        flow_bwd: (B, 2, H, W) backward optical flow
        alpha: scaling factor for squared magnitude threshold
        beta: constant offset threshold

    Returns:
        confidence: (B, 1, H, W) in [0, 1], higher = more reliable
    """
    B, _, H, W = flow_fwd.shape

    # Create sampling grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=flow_fwd.device, dtype=torch.float32),
        torch.arange(W, device=flow_fwd.device, dtype=torch.float32),
        indexing='ij'
    )
    grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)
    grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)

    # Warp backward flow to forward frame using forward flow
    warp_x = grid_x + flow_fwd[:, 0]
    warp_y = grid_y + flow_fwd[:, 1]

    # Normalize to [-1, 1] for grid_sample
    warp_x_norm = 2.0 * warp_x / (W - 1) - 1.0
    warp_y_norm = 2.0 * warp_y / (H - 1) - 1.0
    grid = torch.stack([warp_x_norm, warp_y_norm], dim=-1)

    # Sample backward flow at warped locations
    flow_bwd_warped = F.grid_sample(
        flow_bwd, grid, mode='bilinear', padding_mode='border', align_corners=True
    )

    # Forward-backward consistency error
    # If flow is perfect: flow_fwd + flow_bwd_warped ≈ 0
    fb_sum = flow_fwd + flow_bwd_warped
    fb_error = torch.sum(fb_sum ** 2, dim=1, keepdim=True)  # (B, 1, H, W)

    # Adaptive threshold: error should be small relative to flow magnitude
    flow_fwd_mag = torch.sum(flow_fwd ** 2, dim=1, keepdim=True)
    flow_bwd_mag = torch.sum(flow_bwd_warped ** 2, dim=1, keepdim=True)

    threshold = alpha * (flow_fwd_mag + flow_bwd_mag) + beta

    # Confidence: 1 where consistent, 0 where inconsistent
    # Use soft mask with sigmoid for differentiability
    confidence = torch.exp(-fb_error / (threshold + 1e-6))

    return confidence.clamp(0, 1)


def extract_high_frequency(image):
    """
    Extract high-frequency features using Sobel filtering.
    Returns gradient magnitude per channel.

    Args:
        image: (B, C, H, W) tensor
    Returns:
        edges: (B, C, H, W) gradient magnitude
    """
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0)

    C = image.size(1)
    sobel_x = sobel_x.repeat(C, 1, 1, 1).to(image.device)
    sobel_y = sobel_y.repeat(C, 1, 1, 1).to(image.device)

    edges_x = F.conv2d(image, sobel_x, padding=1, groups=C)
    edges_y = F.conv2d(image, sobel_y, padding=1, groups=C)

    edges = torch.sqrt(edges_x ** 2 + edges_y ** 2 + 1e-6)
    return edges


def warp_with_flow(image, flow):
    """Backward-warp image using optical flow."""
    B, C, H, W = image.shape

    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=image.device, dtype=torch.float32),
        torch.arange(W, device=image.device, dtype=torch.float32),
        indexing='ij'
    )
    grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)
    grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)

    new_x = 2.0 * (grid_x + flow[:, 0]) / (W - 1) - 1.0
    new_y = 2.0 * (grid_y + flow[:, 1]) / (H - 1) - 1.0

    grid = torch.stack([new_x, new_y], dim=-1)
    return F.grid_sample(image, grid, mode='bilinear', padding_mode='border', align_corners=True)

def confidence_multifreq_temporal_loss(stylized_t, stylized_tp1, flow_fwd, confidence, 
                                        lambda_lf=1.0, lambda_hf=1.0):
    """
    Confidence-weighted temporal consistency loss on BOTH low and high frequency features.
    
    Low-frequency:  direct RGB warping comparison (catches color/tone flicker)
    High-frequency: Sobel edge warping comparison (catches texture/edge flicker)
    
    Args:
        stylized_t:    (B, 3, H, W)
        stylized_tp1:  (B, 3, H, W)
        flow_fwd:      (B, 2, H, W)
        confidence:    (B, 1, H, W)
        lambda_lf:     weight for low-frequency term
        lambda_hf:     weight for high-frequency term
    
    Returns:
        loss: scalar
        loss_lf: scalar (for logging)
        loss_hf: scalar (for logging)
    """
    C = stylized_t.size(1)
    conf_sum = confidence.sum() * C + 1e-6

    # ---- Low-frequency: raw RGB temporal consistency ----
    warped_t = warp_with_flow(stylized_t, flow_fwd)
    lf_error = (warped_t - stylized_tp1) ** 2 * confidence
    loss_lf = lf_error.sum() / conf_sum

    # ---- High-frequency: Sobel edge temporal consistency ----
    hf_t = extract_high_frequency(stylized_t)
    hf_tp1 = extract_high_frequency(stylized_tp1)
    hf_t_warped = warp_with_flow(hf_t, flow_fwd)
    hf_error = (hf_t_warped - hf_tp1) ** 2 * confidence
    loss_hf = hf_error.sum() / conf_sum

    # ---- Combined ----
    loss = lambda_lf * loss_lf + lambda_hf * loss_hf

    return loss, loss_lf, loss_hf

# Confidence-weighted high-frequency temporal consistency
def confidence_hf_temporal_loss(stylized_t, stylized_tp1, flow_fwd, confidence):
    """
    Confidence-weighted temporal consistency loss on HIGH-FREQUENCY features.

    Instead of naively comparing warped vs. target across all pixels and channels,
    we:
      1. Extract Sobel edges (high-frequency detail) from both stylized frames
      2. Warp the edge map of frame t using optical flow
      3. Compute per-pixel MSE between warped edges and frame t+1 edges
      4. Weight by confidence mask — only penalize where flow is reliable

    This avoids:
      - Enforcing consistency in occluded regions (wrong flow)
      - Blurring high-frequency details in fast-motion areas
      - Propagating flow noise into the style transfer output

    Args:
        stylized_t:   (B, 3, H, W) stylized frame t
        stylized_tp1:  (B, 3, H, W) stylized frame t+1
        flow_fwd:      (B, 2, H, W) optical flow from content_t -> content_t+1
        confidence:    (B, 1, H, W) flow confidence mask in [0, 1]

    Returns:
        loss: scalar, confidence-weighted high-frequency temporal loss
    """
    # Extract high-frequency features
    hf_t = extract_high_frequency(stylized_t)        # (B, 3, H, W)
    hf_tp1 = extract_high_frequency(stylized_tp1)    # (B, 3, H, W)

    # Warp frame t's edges using the flow
    hf_t_warped = warp_with_flow(hf_t, flow_fwd)     # (B, 3, H, W)

    # Per-pixel squared error
    sq_error = (hf_t_warped - hf_tp1) ** 2           # (B, 3, H, W)

    # Weight by confidence (broadcasts over channels)
    weighted_error = sq_error * confidence            # (B, 3, H, W)

    # Normalize by sum of confidence to avoid bias toward low-confidence frames
    loss = weighted_error.sum() / (confidence.sum() * hf_t.size(1) + 1e-6)

    return loss