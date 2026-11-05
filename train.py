import argparse
import glob
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from model.gaze_trak_model import GazeTrakModel, finetune
from signal_processing.gaze_processor import GazeTrakProcessor


SPEAKER_BANDS = ((18000, 21000), (21500, 24500))


@dataclass
class TrainingConfig:
	data_dir: str = "collected_data"
	output_path: str = "checkpoints/gaze_trak_model.pt"
	fs: int = 50000
	microphones: int = 8
	channels: int = 16
	chirp_len: int = 600
	buffer_size: int = 26
	feature_width: int = 60
	screen_width: int = 800
	screen_height: int = 600
	screen_width_cm: float = 34.5
	screen_height_cm: float = 19.4
	eye_screen_distance_cm: float = 60.0
	batch_size: int = 32
	epochs: int = 30
	lr: float = 1e-2
	val_split: float = 0.2
	device: str = "cpu"
	pretrained: bool = False
	random_crop: bool = True
	run_calibration_finetune: bool = True
	calibration_seconds: float = 15.0
	calibration_epochs: int = 5
	calibration_lr: float = 1e-2


def _find_h5_files(data_dir: str) -> List[str]:
	pattern = os.path.join(data_dir, "*.h5")
	files = sorted(glob.glob(pattern))
	if not files:
		raise FileNotFoundError(f"No .h5 files found in '{data_dir}'.")
	return files


def _nearest_target(screen_data: np.ndarray, timestamp: float) -> np.ndarray:
	times = screen_data[:, 0]
	idx = int(np.argmin(np.abs(times - timestamp)))
	return screen_data[idx, 1:3]


def _crop_width(matrix_2d: np.ndarray, width: int, random_crop: bool) -> np.ndarray:
	current_width = matrix_2d.shape[1]
	if current_width < width:
		raise ValueError(f"Cannot crop width {width} from matrix width {current_width}.")
	if current_width == width:
		return matrix_2d
	if random_crop:
		start = np.random.randint(0, current_width - width + 1)
	else:
		start = (current_width - width) // 2
	return matrix_2d[:, start : start + width]


def _build_processors(cfg: TrainingConfig) -> List[List[GazeTrakProcessor]]:
	processor_grid: List[List[GazeTrakProcessor]] = []
	for _ in range(cfg.microphones):
		per_mic_processors: List[GazeTrakProcessor] = []
		for f_start, f_end in SPEAKER_BANDS:
			per_mic_processors.append(
				GazeTrakProcessor(
					fs=cfg.fs,
					chirp_len=cfg.chirp_len,
					f_start=f_start,
					f_end=f_end,
					buffer_size=cfg.buffer_size,
					band_low=f_start,
					band_high=f_end,
				)
			)
		processor_grid.append(per_mic_processors)
	return processor_grid


def _build_samples_from_file(
	file_path: str, cfg: TrainingConfig
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
	main_inputs: List[np.ndarray] = []
	main_targets: List[np.ndarray] = []
	calib_inputs: List[np.ndarray] = []
	calib_targets: List[np.ndarray] = []

	with h5py.File(file_path, "r") as f:
		if "audio" not in f or "screen_coordinates" not in f:
			print(f"Skipping {file_path}: missing 'audio' or 'screen_coordinates'.")
			return main_inputs, main_targets, calib_inputs, calib_targets

		audio = np.asarray(f["audio"], dtype=np.float32)
		screen = np.asarray(f["screen_coordinates"], dtype=np.float32)

	if audio.ndim != 2 or audio.shape[1] != cfg.microphones:
		print(
			f"Skipping {file_path}: expected audio shape (N, {cfg.microphones}), "
			f"got {audio.shape}."
		)
		return main_inputs, main_targets, calib_inputs, calib_targets

	if screen.ndim != 2 or screen.shape[1] < 3:
		print(f"Skipping {file_path}: invalid screen coordinate shape {screen.shape}.")
		return main_inputs, main_targets, calib_inputs, calib_targets

	total_frames = audio.shape[0] // cfg.chirp_len
	if total_frames <= cfg.buffer_size:
		print(f"Skipping {file_path}: not enough audio frames ({total_frames}).")
		return main_inputs, main_targets, calib_inputs, calib_targets

	processors = _build_processors(cfg)

	for frame_idx in range(total_frames):
		start = frame_idx * cfg.chirp_len
		end = start + cfg.chirp_len
		frame = audio[start:end, :]

		channel_mats: List[np.ndarray] = []
		for mic_idx in range(cfg.microphones):
			for proc in processors[mic_idx]:
				mat = proc.process_frame(frame[:, mic_idx])
				channel_mats.append(mat)

		if not all(m is not None for m in channel_mats):
			continue

		frame_time = end / cfg.fs
		raw_target = _nearest_target(screen, frame_time)
		target = np.array(
			[
				np.clip(raw_target[0] / cfg.screen_width, 0.0, 1.0),
				np.clip(raw_target[1] / cfg.screen_height, 0.0, 1.0),
			],
			dtype=np.float32,
		)

		x = np.stack(
			[_crop_width(m, cfg.feature_width, cfg.random_crop) for m in channel_mats], axis=0
		).astype(np.float32)

		if frame_time <= cfg.calibration_seconds:
			calib_inputs.append(x)
			calib_targets.append(target)
		else:
			main_inputs.append(x)
			main_targets.append(target)

	# Fallback for very short recordings without a full calibration segment.
	if not main_inputs:
		main_inputs = calib_inputs
		main_targets = calib_targets
		calib_inputs = []
		calib_targets = []

	return main_inputs, main_targets, calib_inputs, calib_targets


def _as_dataset(inputs: List[np.ndarray], targets: List[np.ndarray]) -> Optional[TensorDataset]:
	if not inputs:
		return None
	x = torch.from_numpy(np.stack(inputs, axis=0))
	y = torch.from_numpy(np.stack(targets, axis=0))
	return TensorDataset(x, y)


def build_datasets(cfg: TrainingConfig) -> Tuple[TensorDataset, Optional[TensorDataset]]:
	all_inputs: List[np.ndarray] = []
	all_targets: List[np.ndarray] = []
	calib_inputs: List[np.ndarray] = []
	calib_targets: List[np.ndarray] = []

	for file_path in _find_h5_files(cfg.data_dir):
		file_inputs, file_targets, file_calib_inputs, file_calib_targets = _build_samples_from_file(file_path, cfg)
		all_inputs.extend(file_inputs)
		all_targets.extend(file_targets)
		calib_inputs.extend(file_calib_inputs)
		calib_targets.extend(file_calib_targets)
		print(
			f"Loaded {len(file_inputs)} train samples and {len(file_calib_inputs)} calibration samples "
			f"from {os.path.basename(file_path)}"
		)

	train_dataset = _as_dataset(all_inputs, all_targets)
	if train_dataset is None:
		raise RuntimeError(
			"No valid training samples were produced. "
			"Check data quality or collection settings."
		)

	calibration_dataset = _as_dataset(calib_inputs, calib_targets)
	x_shape = tuple(train_dataset.tensors[0].shape)
	y_shape = tuple(train_dataset.tensors[1].shape)
	print(f"Dataset ready: X={x_shape}, y={y_shape}")
	return train_dataset, calibration_dataset


def _mgae_deg(pred_xy_norm: torch.Tensor, gt_xy_norm: torch.Tensor, cfg: TrainingConfig) -> float:
	pred_px = pred_xy_norm.clone()
	gt_px = gt_xy_norm.clone()
	pred_px[:, 0] *= cfg.screen_width
	pred_px[:, 1] *= cfg.screen_height
	gt_px[:, 0] *= cfg.screen_width
	gt_px[:, 1] *= cfg.screen_height

	cm_per_px_x = cfg.screen_width_cm / float(cfg.screen_width)
	cm_per_px_y = cfg.screen_height_cm / float(cfg.screen_height)

	pred_cm_x = (pred_px[:, 0] - cfg.screen_width / 2.0) * cm_per_px_x
	pred_cm_y = (pred_px[:, 1] - cfg.screen_height / 2.0) * cm_per_px_y
	gt_cm_x = (gt_px[:, 0] - cfg.screen_width / 2.0) * cm_per_px_x
	gt_cm_y = (gt_px[:, 1] - cfg.screen_height / 2.0) * cm_per_px_y

	z = torch.full_like(pred_cm_x, fill_value=cfg.eye_screen_distance_cm)
	pred_vec = torch.stack((pred_cm_x, pred_cm_y, z), dim=1)
	gt_vec = torch.stack((gt_cm_x, gt_cm_y, z), dim=1)

	dot = torch.sum(pred_vec * gt_vec, dim=1)
	pred_norm = torch.norm(pred_vec, dim=1)
	gt_norm = torch.norm(gt_vec, dim=1)
	cos_theta = torch.clamp(dot / (pred_norm * gt_norm + 1e-8), -1.0, 1.0)
	theta_deg = torch.rad2deg(torch.acos(cos_theta))
	return float(torch.mean(theta_deg).item())


def _evaluate_mgae(model: GazeTrakModel, loader: DataLoader, cfg: TrainingConfig) -> float:
	model.eval()
	all_pred = []
	all_gt = []
	with torch.no_grad():
		for xb, yb in loader:
			xb = xb.to(cfg.device)
			yb = yb.to(cfg.device)
			pred = model(xb)
			all_pred.append(pred.detach().cpu())
			all_gt.append(yb.detach().cpu())
	pred_cat = torch.cat(all_pred, dim=0)
	gt_cat = torch.cat(all_gt, dim=0)
	return _mgae_deg(pred_cat, gt_cat, cfg)


def train(cfg: TrainingConfig) -> None:
	dataset, calibration_dataset = build_datasets(cfg)

	val_size = max(1, int(len(dataset) * cfg.val_split))
	train_size = len(dataset) - val_size
	if train_size <= 0:
		raise RuntimeError("Dataset too small for train/val split.")

	train_set, val_set = random_split(dataset, [train_size, val_size])
	train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
	val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False)

	model = GazeTrakModel(in_channels=cfg.channels, output_dim=2, pretrained=cfg.pretrained)
	model = model.to(cfg.device)

	criterion = torch.nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

	best_val = float("inf")
	os.makedirs(os.path.dirname(cfg.output_path) or ".", exist_ok=True)

	for epoch in range(1, cfg.epochs + 1):
		model.train()
		train_loss = 0.0

		for xb, yb in train_loader:
			xb = xb.to(cfg.device)
			yb = yb.to(cfg.device)

			optimizer.zero_grad()
			pred = model(xb)
			loss = criterion(pred, yb)
			loss.backward()
			optimizer.step()

			train_loss += loss.item() * xb.size(0)

		model.eval()
		val_loss = 0.0
		with torch.no_grad():
			for xb, yb in val_loader:
				xb = xb.to(cfg.device)
				yb = yb.to(cfg.device)
				pred = model(xb)
				loss = criterion(pred, yb)
				val_loss += loss.item() * xb.size(0)

		train_loss /= len(train_set)
		val_loss /= len(val_set)
		print(
			f"Epoch {epoch}/{cfg.epochs} | train_loss={train_loss:.6f} "
			f"| val_loss={val_loss:.6f}"
		)

		if val_loss < best_val:
			best_val = val_loss
			torch.save(
				{
					"state_dict": model.state_dict(),
					"config": cfg.__dict__,
					"best_val_loss": best_val,
				},
				cfg.output_path,
			)
			print(f"Saved best model to {cfg.output_path}")

	val_mgae = _evaluate_mgae(model, val_loader, cfg)
	print(f"Validation MGAE (deg): {val_mgae:.4f}")

	if cfg.run_calibration_finetune and calibration_dataset is not None and len(calibration_dataset) > 0:
		calib_loader = DataLoader(calibration_dataset, batch_size=cfg.batch_size, shuffle=True)
		model = finetune(
			model,
			calib_loader,
			epochs=cfg.calibration_epochs,
			learning_rate=cfg.calibration_lr,
			device=cfg.device,
		)
		finetuned_output = cfg.output_path.replace(".pt", "_calibrated.pt")
		torch.save(
			{
				"state_dict": model.state_dict(),
				"config": cfg.__dict__,
				"note": "Fine-tuned with calibration samples",
			},
			finetuned_output,
		)
		val_mgae_after = _evaluate_mgae(model, val_loader, cfg)
		print(f"Saved calibrated model to {finetuned_output}")
		print(f"Validation MGAE after calibration fine-tune (deg): {val_mgae_after:.4f}")


def _build_argparser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description="Train EchoEyeTracker gaze regression model.")
	p.add_argument("--data-dir", default="collected_data", help="Directory with .h5 training files.")
	p.add_argument("--output", default="checkpoints/gaze_trak_model.pt", help="Checkpoint file path.")
	p.add_argument("--epochs", type=int, default=30)
	p.add_argument("--batch-size", type=int, default=32)
	p.add_argument("--lr", type=float, default=1e-2)
	p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
	p.add_argument("--pretrained", action="store_true", help="Use ImageNet-pretrained ResNet18 backbone.")
	p.add_argument("--val-split", type=float, default=0.2)
	p.add_argument("--screen-width", type=int, default=800)
	p.add_argument("--screen-height", type=int, default=600)
	p.add_argument("--screen-width-cm", type=float, default=34.5)
	p.add_argument("--screen-height-cm", type=float, default=19.4)
	p.add_argument("--eye-screen-distance-cm", type=float, default=60.0)
	p.add_argument("--disable-random-crop", action="store_true", help="Use center crop instead of random crop.")
	p.add_argument("--disable-calibration-finetune", action="store_true")
	p.add_argument("--calibration-seconds", type=float, default=15.0)
	p.add_argument("--calibration-epochs", type=int, default=5)
	p.add_argument("--calibration-lr", type=float, default=1e-2)
	return p


def main() -> None:
	args = _build_argparser().parse_args()
	cfg = TrainingConfig(
		data_dir=args.data_dir,
		output_path=args.output,
		batch_size=args.batch_size,
		epochs=args.epochs,
		lr=args.lr,
		device=args.device,
		pretrained=args.pretrained,
		val_split=args.val_split,
		screen_width=args.screen_width,
		screen_height=args.screen_height,
		screen_width_cm=args.screen_width_cm,
		screen_height_cm=args.screen_height_cm,
		eye_screen_distance_cm=args.eye_screen_distance_cm,
		random_crop=not args.disable_random_crop,
		run_calibration_finetune=not args.disable_calibration_finetune,
		calibration_seconds=args.calibration_seconds,
		calibration_epochs=args.calibration_epochs,
		calibration_lr=args.calibration_lr,
	)

	print("Training config:", cfg)
	train(cfg)


if __name__ == "__main__":
	main()
