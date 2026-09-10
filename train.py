import argparse
import glob
import os
import time

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from model import kmireg


def get_training_files(source_dir, target_dir):
    source_files = sorted(glob.glob(os.path.join(source_dir, "*.nii.gz")))
    target_files = sorted(glob.glob(os.path.join(target_dir, "*.nii.gz")))

    if not source_files or not target_files:
        raise FileNotFoundError("No training volumes were found in the data directories.")
    if len(source_files) != len(target_files):
        raise ValueError("Source and target datasets must contain the same number of volumes.")

    return source_files, target_files


def load_batch(files, indices, device):
    volumes = [nib.load(files[index]).get_fdata(dtype=np.float32) for index in indices]
    array = np.stack(volumes)[:, None]
    return torch.from_numpy(array).to(device)


def ncc_loss(moving, fixed, window_size=9):
    window = (window_size,) * 3
    padding = window_size // 2
    kernel = torch.ones((1, 1, *window), device=moving.device, dtype=moving.dtype)
    window_volume = float(window_size**3)

    moving_sum = F.conv3d(moving, kernel, padding=padding)
    fixed_sum = F.conv3d(fixed, kernel, padding=padding)
    moving_sq_sum = F.conv3d(moving.square(), kernel, padding=padding)
    fixed_sq_sum = F.conv3d(fixed.square(), kernel, padding=padding)
    product_sum = F.conv3d(moving * fixed, kernel, padding=padding)

    moving_mean = moving_sum / window_volume
    fixed_mean = fixed_sum / window_volume
    cross = (
        product_sum
        - fixed_mean * moving_sum
        - moving_mean * fixed_sum
        + moving_mean * fixed_mean * window_volume
    )
    moving_var = moving_sq_sum - 2 * moving_mean * moving_sum + moving_mean.square() * window_volume
    fixed_var = fixed_sq_sum - 2 * fixed_mean * fixed_sum + fixed_mean.square() * window_volume
    return -(cross.square() / (moving_var * fixed_var + 1e-5)).mean()


def gradient_loss(flow):
    dy = (flow[:, :, 1:] - flow[:, :, :-1]).square().mean()
    dx = (flow[:, :, :, 1:] - flow[:, :, :, :-1]).square().mean()
    dz = (flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]).square().mean()
    return (dx + dy + dz) / 3


def train(args):
    if args.gpu == "-1" or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}")

    source_files, target_files = get_training_files(
        args.source_dir, args.target_dir
    )
    source_shape = nib.load(source_files[0]).shape
    target_shape = nib.load(target_files[0]).shape
    if source_shape != target_shape:
        raise ValueError(
            f"Source and target shapes must match, got {source_shape} and {target_shape}."
        )

    inshape = source_shape
    model = kmireg(inshape).to(device)

    if args.load_model:
        state_dict = torch.load(args.load_model, map_location=device)
        model.load_state_dict(state_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    os.makedirs(args.model_dir, exist_ok=True)

    for epoch in range(args.initial_epoch, args.epochs):
        start_time = time.time()
        model.train()
        epoch_losses = []

        for _ in range(args.steps_per_epoch):
            indices = np.random.randint(len(source_files), size=args.batch_size)
            moving = load_batch(source_files, indices, device)
            fixed = load_batch(target_files, indices, device)
            warped, flow = model(moving, fixed)

            image_loss = ncc_loss(warped, fixed) * args.image_loss_weight
            grad_loss = gradient_loss(flow) * args.grad_loss_weight
            total_loss = image_loss + grad_loss

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            epoch_losses.append(
                (total_loss.item(), image_loss.item(), grad_loss.item())
            )

        mean_total, mean_image, mean_grad = np.mean(epoch_losses, axis=0)
        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch + 1}/{args.epochs} - {elapsed:.2f} sec - "
            f"loss: {mean_total:.4f} ({mean_image:.4f}, {mean_grad:.4f})",
            flush=True,
        )

        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.model_dir, f"{epoch + 1:04d}.pt")
            torch.save(model.state_dict(), checkpoint_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Train KMI-Reg.")
    parser.add_argument("--source-dir", required=True, help="Source image directory.")
    parser.add_argument("--target-dir", required=True, help="Target image directory.")
    parser.add_argument("--model-dir", default="models", help="Checkpoint directory.")
    parser.add_argument("--load-model", help="Checkpoint used to initialize the model.")
    parser.add_argument("--gpu", default="0", help="GPU ID, or -1 for CPU.")
    parser.add_argument("--initial-epoch", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--image-loss-weight", type=float, default=1.0)
    parser.add_argument("--grad-loss-weight", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
