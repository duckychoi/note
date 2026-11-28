"""Depth Pro 2-S comparison with single-image inference and paired training."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from depth_full_data import Hypersim, NyuDepth


SOURCES = Path(__file__).resolve().parents[3] / "research-sources"
GPU = "cuda"


def trimmed_mean(values, fraction):
    if values.numel() == 0:
        return values.sum()
    k = max(1, int(values.numel() * (1 - fraction)))
    return torch.topk(values, k, largest=False, sorted=False).values.mean()


def normalize_depth(depth, valid):
    valid_values = depth[valid]
    center = valid_values.median()
    return (depth-center) / (valid_values-center).abs().mean().clamp_min(1e-6)


def first_gradient_error(residual, valid):
    horizontal = (residual[:, 1:] - residual[:, :-1]).abs()[valid[:, 1:] & valid[:, :-1]]
    vertical = (residual[1:] - residual[:-1]).abs()[valid[1:] & valid[:-1]]
    return (horizontal.sum()+vertical.sum()) / max(1, horizontal.numel()+vertical.numel())


def supervised(prediction, depth, valid, focal, synthetic, silog=False):
    target = focal / (depth.shape[1] * depth)
    if not valid.any():
        return prediction.sum()*0
    if silog:
        # DA V2 metric loss, lambda=.5. Inverse-depth log errors reverse the
        # sign of depth log errors, preserving this norm. Norm has a finite
        # subgradient at exact agreement.
        difference = prediction[valid].log()-target[valid].log()
        mean = difference.mean()
        terms = torch.cat((difference-mean, mean.reshape(1)*(.5*difference.numel())**.5))
        return torch.linalg.vector_norm(terms)/difference.numel()**.5
    trim = .01 if synthetic else .10
    metric = trimmed_mean((prediction-target).abs()[valid], trim)
    residual = normalize_depth(prediction, valid) - normalize_depth(target, valid)
    shape = trimmed_mean(residual.abs()[valid], trim)
    if not synthetic:
        return metric+shape
    gaussian = prediction.new_tensor([1, 4, 6, 4, 1])
    kernel = (gaussian[:, None] * gaussian / 256)[None, None]
    residual = residual.masked_fill(~valid, 0)
    edge = first_gradient_error(residual, valid)
    for _ in range(2):
        residual = F.conv2d(residual[None, None], kernel, stride=2, padding=2)[0, 0]
        valid = F.conv2d(valid.float()[None, None], kernel, stride=2, padding=2)[0, 0] > .9999
        edge = edge + first_gradient_error(residual, valid)
    return metric+shape+edge/3


@lru_cache(maxsize=8)
def pixel_grid(height, width):
    y, x = torch.meshgrid(torch.arange(height, device=GPU), torch.arange(width, device=GPU), indexing="ij")
    return torch.stack((x, y, torch.ones_like(x)), dim=-1).float()


def transform_points(points, pose):
    return points @ pose[:, :3].T + pose[:, 3]


def project(points, matrix, height, width):
    projection = points @ matrix.T
    pixel = projection[..., :2] / projection[..., 2:].clamp_min(1e-5)
    inside = ((pixel[..., 0] >= 0) & (pixel[..., 0] <= width-1) &
              (pixel[..., 1] >= 0) & (pixel[..., 1] <= height-1) & (points[..., 2] > 0))
    factor = projection.new_tensor((width, height))
    grid = (pixel+.5)*2/factor-1
    return grid, inside


def sample_at(values, grid, mode="bilinear"):
    return F.grid_sample(values[None, None], grid[None], mode=mode, align_corners=False)[0, 0]


class ErrorCovariance:
    """Per-bin exponentially weighted centered prediction-error moments."""

    def __init__(self):
        self.moments = torch.zeros((4, 5), device=GPU)
        self.count = torch.zeros(4, device=GPU)

    @torch.no_grad()
    def lookup(self, bins):
        p, r, pp, rr, pr = self.moments.T
        p_var, r_var, cross = (pp-p.square()).clamp_min(0), (rr-r.square()).clamp_min(0), pr-p*r
        gain = ((p_var-cross)/(p_var+r_var-2*cross).clamp_min(1e-5)).clamp(0., 1.)
        initialized = self.count >= 1024
        gain = torch.where(initialized, gain, .5)
        bias_shift = torch.where(initialized, r-p, 0.)
        return gain[bins], bias_shift[bins]

    @torch.no_grad()
    def update(self, prior_error, observation_error, bins, valid):
        for group in range(4):
            keep = (bins == group) & valid
            a, b = prior_error[keep], observation_error[keep]
            if a.numel():
                batch = torch.stack((a, b, a.square(), b.square(), a*b)).mean(dim=1)
                weight = min(1., .03 + 1024/(1024+float(self.count[group])))
                self.moments[group].lerp_(batch, weight)
                self.count[group] += a.numel()


def fusion_supervision(log_depth, neighbor_log_depth, bias_shift, adaptive_gain, keep, arm, diagnostics=False):
    prior = log_depth.detach()
    innovation = (neighbor_log_depth - prior - bias_shift).detach()
    gain = adaptive_gain.detach() if arm == "kf" else .5
    target = prior + gain*innovation
    error = (log_depth-target)[keep]
    # d/d(log_depth) = -K * innovation / n. Detached-target L1 loses the
    # magnitude of every strictly positive gain.
    loss = .5*error.square().sum()/max(1, error.numel())
    stats = {}
    if diagnostics and error.numel():
        k = adaptive_gain.detach()[keep]
        stats = dict(adaptive_gain_mean=float(k.mean()), adaptive_gain_zero_fraction=float((k == 0).float().mean()),
                     adaptive_gain_one_fraction=float((k == 1).float().mean()),
                     eligible_fraction=float(keep.float().mean()), mean_abs_innovation=float(innovation[keep].abs().mean()),
                     mean_abs_fusion_step=float(error.detach().abs().mean()))
    return loss, stats


def paired_loss(first, second, pose, arm, covariance, diagnostics=False, held_out=False):
    rgb, depth, valid, k, q = first
    neighbor_rgb, neighbor_depth, neighbor_valid, neighbor_k, neighbor_q = second
    h, w = depth.shape
    nh, nw = neighbor_depth.shape
    rays = pixel_grid(h, w)[::4, ::4] @ torch.linalg.inv(k).T
    truth = depth[::4, ::4]
    truth_in_neighbor = transform_points(rays*truth[..., None], pose)
    gt_grid, visible = project(truth_in_neighbor, neighbor_k, nh, nw)
    observed_truth = sample_at(neighbor_depth, gt_grid, "nearest")
    keep = (valid[::4, ::4] & visible & (sample_at(neighbor_valid.float(), gt_grid, "nearest") > .5) &
            ((observed_truth - truth_in_neighbor[..., 2]).abs() <= .02*observed_truth))
    predicted = k[0, 0]/(w*q[::4, ::4].clamp_min(1e-5))
    neighbor_prediction = neighbor_k[0, 0]/(nw*neighbor_q.clamp_min(1e-5))
    predicted_in_neighbor = transform_points(rays*predicted[..., None], pose)
    pred_grid, pred_visible = project(predicted_in_neighbor, neighbor_k, nh, nw)
    from_first = predicted_in_neighbor[..., 2].clamp_min(.01).reciprocal()
    from_second = sample_at(neighbor_prediction, pred_grid).clamp_min(.01).reciprocal()
    mvc_mask = keep & pred_visible
    mvc_pixels = int(mvc_mask.sum())
    loss = trimmed_mean((from_first-from_second).abs()[mvc_mask], .01)
    if arm in ("d2", "silog"):
        return loss, int(keep.sum()), {"mvc_valid_pixels": mvc_pixels}

    # A second prediction, reprojected using a training-only visible correspondence.
    with torch.no_grad():
        neighbor_ray = truth_in_neighbor / truth_in_neighbor[..., 2:].clamp_min(.01)
        observed_points = neighbor_ray * sample_at(neighbor_prediction, gt_grid)[..., None]
        observation = ((observed_points-pose[:, 3]) @ pose[:, :3])[..., 2]
        keep = keep & torch.isfinite(observation) & (observation >= .25) & (observation <= 30.)
        prior = predicted.detach().log()
        observed_log = observation.clamp(.25, 30.).log()
        luma = rgb.mean(0)[::4, ::4]
        other_luma = sample_at(neighbor_rgb.mean(0), gt_grid)
        bins = 2*(luma < .28).long() + ((luma-other_luma).abs() > .18).long()
        gain, bias_shift = covariance.lookup(bins)
        if not held_out:
            covariance.update(prior-truth.log(), observed_log-truth.log(), bins, keep)
    fusion, stats = fusion_supervision(predicted.log(), observed_log, bias_shift, gain, keep, arm, diagnostics)
    stats["mvc_valid_pixels"] = mvc_pixels
    if held_out and keep.any():
        target = truth[keep].log()
        innovation = (observed_log-prior-bias_shift)[keep]
        errors = dict(prior=prior[keep]-target, neighbor=observed_log[keep]-target,
                      fixed=prior[keep]+.5*innovation-target,
                      adaptive=prior[keep]+gain[keep]*innovation-target)
        stats["mean_squared_log_error"] = {label: float(error.square().mean()) for label, error in errors.items()}
        stats["by_bin"] = {}
        for group in range(4):
            mask = bins[keep] == group
            if mask.any():
                stats["by_bin"][group] = dict(count=int(mask.sum()), gain=float(gain[keep][mask].mean()),
                    **{label: float(error[mask].square().mean()) for label, error in errors.items()})
    return loss + .25*fusion, int(keep.sum()), stats


def input_sample(sample, transform, exponent):
    rgb, depth, valid, intrinsics = sample
    if exponent != 1.:
        lookup = np.rint(255*(np.arange(256)/255)**exponent).astype(np.uint8)
        rgb = lookup[rgb]
    appearance, z, mask, k = [torch.as_tensor(x, device=GPU) for x in (rgb, depth, valid, intrinsics)]
    return transform(rgb), (appearance.permute(2, 0, 1).float()/255, z, mask, k)


def inference(sample, model, transform, exponent=1.):
    pixels, fields = input_sample(sample, transform, exponent)
    return (*fields, model.predict_inverse_depth(pixels[None])[0].float().clamp_min(1e-5))


def predict_independent(model, pixels):
    """Batch the exact T=1 attention topology without sharing tokens across photos."""
    from depth_pro_2.model import INPUT_MULTIPLE
    from depth_pro_2.network.encoder import autocast_for

    height, width = pixels.shape[-2:]
    pixels = F.pad(pixels, (0, -width % INPUT_MULTIPLE, 0, -height % INPUT_MULTIPLE), mode="replicate")
    encoder = model.encoder
    grid = (pixels.shape[-2] // encoder.patch_size, pixels.shape[-1] // encoder.patch_size)
    with autocast_for(pixels.device, model.precision)():
        rope = encoder.rope.get_embed(shape=grid)
        tokens, features = encoder.embed(pixels), []
        for index, block in enumerate(encoder.blocks):
            result = block(tokens, rope=rope)
            if index in encoder.cross_frame_blocks:
                prefix = encoder.num_prefix_tokens
                result = torch.cat((tokens[:, :prefix], result[:, prefix:]), dim=1)
            tokens = result
            if index in encoder.reassembled_blocks:
                features.append(tokens[:, encoder.num_prefix_tokens:])
    return model.decode_frame(features, grid)[:, 0, :height, :width].float().clamp_min(1e-5)


def paired_inference(sample, second, model, transform, gamma):
    a, first_fields = input_sample(sample, transform, float(gamma[0]))
    b, second_fields = input_sample(second, transform, float(gamma[1]))
    qa, qb = predict_independent(model, torch.stack((a, b)))
    return (*first_fields, qa), (*second_fields, qb)


@torch.inference_mode()
def validation(model, transform, data):
    model.eval()
    rows = []
    for sample, _, _ in data:
        _, depth, mask, k, q = inference(sample, model, transform)
        if mask.any():
            ratio = (k[0, 0]/(q.shape[1]*q[mask]))/depth[mask]
            rows.append([float((ratio-1).abs().mean()), float((torch.maximum(ratio, 1/ratio)<1.25).float().mean())])
    return np.mean(rows, 0).tolist()


def prefetched_samples(dataset, indices):
    """Read the next RGB-D pair while the current pair uses the GPU."""
    indices = iter(indices)
    first = next(indices, None)
    if first is None:
        return
    with ThreadPoolExecutor(max_workers=1) as worker:
        future = worker.submit(dataset.__getitem__, int(first))
        for index in indices:
            sample = future.result()
            future = worker.submit(dataset.__getitem__, int(index))
            yield sample
        yield future.result()


def main(args):
    sys.path.insert(0, str(SOURCES / "ml-depth-pro-2" / "src"))
    from dataclasses import replace
    from depth_pro_2 import PRESETS, create_model_and_transforms

    training_real = args.arm == "real"
    stereo = None
    if hasattr(args, "stereo_mode"):
        from stereo_student import StereoTargets, native_rgb_validation, memory_pressure
        if hasattr(args, "pair_manifest"):
            import ctypes
            if not ctypes.windll.kernel32.SetPriorityClass(ctypes.c_void_p(-1),0x4000):
                raise OSError("Cannot lower shared-workstation CPU scheduling priority")
        if args.arm not in ("d2", "silog"):
            raise ValueError("Stereo students require a matched D2 or SiLog backbone recipe")
        calibration=Path(args.stereo_calibration) if hasattr(args,"stereo_calibration") else None
        stereo = StereoTargets(args.stereo_mode, args.stride, calibration)
    nyu = SOURCES / "encoder42_eval" / "nyu-depthv2-wds"
    if training_real:
        train = NyuDepth(nyu, "train", args.stride, smoke=args.smoke)
        dev = None if args.smoke else NyuDepth(nyu, "val", args.stride)
    else:
        train = Hypersim("train_full", args.stride, pair=True, smoke=args.smoke)
        if hasattr(args,"pair_manifest"):
            if not stereo:
                raise ValueError("Selected geometry pairs require a matched stereo experiment")
            selected=json.loads(Path(args.pair_manifest).read_text(encoding="utf8"))
            neighbors={tuple(row[:3]):row[3] for row in selected}
            if len(neighbors)!=len(selected) or (not args.smoke and len(neighbors)!=len(train.rows)):
                raise ValueError("Selected-pair manifest must cover each official anchor exactly once")
            selected_rows=[]
            for s,c,frame,_ in train.rows:
                if (s,c,frame) not in neighbors or neighbors[s,c,frame] not in train.groups[s,c]:
                    raise ValueError(f"Invalid selected pair for {s}/{c}/{frame}")
                selected_rows.append((s,c,frame,neighbors[s,c,frame]))
            train.rows=selected_rows
        dev = Hypersim("validation_24", 1 if stereo else args.stride, pair=False, smoke=args.smoke)
    if len(train) < 2 or (dev is not None and len(dev) == 0):
        raise ValueError("The training and validation datasets have no usable samples")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.hub.set_dir(str(SOURCES / "ml-depth-pro-2-cache" / "hub"))
    preset = PRESETS["depthpro2-s"]
    model, transform = create_model_and_transforms(preset if stereo else replace(preset, fov_checkpoint_uri=None), device=GPU)
    encoder = list(model.encoder.parameters())
    head = list(model.decoder.parameters()) + list(model.head.parameters())
    model.requires_grad_(True)
    if stereo:
        model.fov.requires_grad_(False)
    opt = torch.optim.AdamW([{"params": encoder, "lr": args.lr}, {"params": head, "lr": args.lr*3}], weight_decay=.01)
    covariance = ErrorCovariance()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "training.jsonl"
    checkpoint_path = output / "latest.train.pt"
    if log_path.exists() and not args.resume:
        raise FileExistsError(f"Use a new run directory: {output}")
    started = time.monotonic()

    def record(**values):
        values.update(seconds=round(time.monotonic()-started, 2))
        line = json.dumps(values, allow_nan=False)
        with log_path.open("a", encoding="utf8") as stream:
            stream.write(line+"\n")
        print(line, flush=True)

    def save_checkpoint(path, state):
        pending = path.with_suffix(".partial")
        torch.save(state, pending)
        pending.replace(path)

    total = min(args.steps or len(train)*args.epochs, len(train)*args.epochs)
    config = {key: value for key, value in vars(args).items() if key not in ("output", "resume")}
    if stereo:
        config.update(stereo_calibration_sha256=stereo.digest, development_mode="native_rgb_scene_mean")
    if args.arm in ("fixed", "kf"):
        config["fusion_objective"] = "half_squared_log_v1"
    selection_digest = hashlib.sha256(repr(train.rows).encode("utf8")).hexdigest()
    step, best = 0, float("inf")
    coverage = dict(since_step=0, pairs=0, invalid_pose_pairs=0, sampled_pixels=0,
                    mvc_valid_pixels=0, mvc_empty_pairs=0, fusion_valid_pixels=0)
    def checkpoint_training():
        save_checkpoint(checkpoint_path,dict(model=model.state_dict(),optimizer=opt.state_dict(),step=step,
                        best=best,moments=covariance.moments,count=covariance.count,training_rows=len(train),
                        selection_digest=selection_digest,config=config,pair_coverage=coverage,
                        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()))
    if args.resume:
        checkpoint = torch.load(checkpoint_path, map_location=GPU, weights_only=True)
        if checkpoint["config"] != config or checkpoint["selection_digest"] != selection_digest:
            raise ValueError("Resume needs the original training settings and data selection")
        model.load_state_dict(checkpoint["model"])
        opt.load_state_dict(checkpoint["optimizer"])
        covariance.moments.copy_(checkpoint["moments"])
        covariance.count.copy_(checkpoint["count"])
        step, best = checkpoint["step"], checkpoint["best"]
        coverage = checkpoint.get("pair_coverage", {**coverage, "since_step": step})
        if stereo and "stereo_measured_pairs" not in coverage:
            coverage["stereo_measured_pairs"]=coverage["pairs"]-coverage["invalid_pose_pairs"]
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        torch.cuda.set_rng_state(checkpoint["cuda_rng"].cpu())
        del checkpoint
    else:
        if hasattr(args,"pair_manifest") and not args.smoke:
            checkpoint_training()
    if not args.resume or not log_path.exists():
        record(config=config, training_rows=len(train), selection_digest=selection_digest,
               development_rows=0 if dev is None else len(dev),
               encoder_parameters=sum(x.numel() for x in encoder), decoder_parameters=sum(x.numel() for x in head))
    if args.resume:
        record(resumed_at=step)
    if step >= total:
        return
    model.train()
    first_epoch, offset = divmod(step, len(train))
    for epoch in range(first_epoch, args.epochs):
        order = np.random.default_rng([args.seed, epoch]).permutation(len(train))
        order = order[offset if epoch == first_epoch else 0:][:total-step]
        for sample, second_sample, pose in prefetched_samples(train, order):
            # The same color/exposure draws and sample order are used for all synthetic arms.
            gamma = np.exp(np.random.default_rng([args.seed, step]).uniform(-.45, .6, size=2))
            if stereo:
                lights = tuple(np.random.default_rng([args.seed, step]).integers(2, size=2).tolist())
                gamma = [1.4 if light else 1. for light in lights]
            if second_sample is None:
                first = inference(sample, model, transform, float(gamma[0]))
            else:
                first, second = paired_inference(sample, second_sample, model, transform, gamma)
            _, depth, mask, k, q = first
            loss = supervised(q, depth, mask, k[0, 0], not training_real, silog=args.arm == "silog")
            eligible, details = 0, {}
            if second_sample is not None:
                _, z2, m2, k2, q2 = second
                loss = .5*(loss+supervised(q2, z2, m2, k2[0, 0], True, silog=args.arm == "silog"))
                if pose is not None:
                    cross, eligible, details = paired_loss(first, second, torch.as_tensor(pose, device=GPU), args.arm,
                                                           covariance, step == 0 or (step+1) % args.log_every == 0)
                    loss = loss + cross
                    # Off arms have zero stereo gradient. Retain the first development
                    # interval for coverage diagnostics without matching unused images forever.
                    if stereo and (stereo.mode!="off" or step<args.val_every or not hasattr(args,"pair_manifest")):
                        extra, eligible, grid_count = stereo.loss((sample, second_sample), pose, lights, (first, second))
                        loss = loss+.25*extra
                        details["stereo_auxiliary_loss"] = float(extra.detach())
                        coverage["stereo_valid_pixels_with_pose"] = coverage.get("stereo_valid_pixels_with_pose", 0)+grid_count
                        coverage["stereo_measured_pairs"]=coverage.get("stereo_measured_pairs",0)+1
                    elif stereo:
                        eligible=None
                        details["stereo_measurement_skipped"]=True
                else:
                    details["invalid_pair_pose"] = True
                coverage["pairs"] += 1
                coverage["invalid_pose_pairs"] += int(pose is None)
                coverage["sampled_pixels"] += depth[::4, ::4].numel()
                mvc_valid = details.get("mvc_valid_pixels", 0)
                coverage["mvc_valid_pixels"] += mvc_valid
                coverage["mvc_empty_pairs"] += int(mvc_valid == 0)
                if eligible is not None and (args.arm in ("fixed","kf") or stereo):
                    coverage["fusion_valid_pixels"] += eligible
            step += 1
            scale = float(min(1., step/min(1000, max(1, total//10))) * (.2+.8*(1+np.cos(np.pi*step/total))/2))
            for group, ratio in zip(opt.param_groups, (1., 3.)):
                group["lr"] = args.lr*ratio*scale
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite loss at {step}")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opt.step()
            if step == 1 or step % args.log_every == 0:
                if args.arm in ("fixed", "kf"):
                    details.update(conditional_error_moments=covariance.moments.tolist(),
                                   conditional_counts=covariance.count.tolist())
                record(step=step, epoch=epoch+1, loss=float(loss.detach()), eligible=eligible,
                       grad_norm=float(grad), **details)
            if second_sample is not None and (step % args.val_every == 0 or step == total):
                record(step=step, pair_coverage=coverage)
            if (not args.smoke and step % args.val_every == 0) or step == total:
                if dev is not None:
                    val_details={}
                    if stereo:
                        val,val_details = native_rgb_validation(model, transform, dev, args.smoke)
                    else:
                        val = validation(model, transform, dev if not args.smoke else [dev[i] for i in range(min(8, len(dev)))])
                    record(step=step, development_absrel=val[0], development_delta1=val[1],**val_details)
                    if val[0] < best:
                        best = val[0]
                        save_checkpoint(output / "best.pt", model.state_dict())
                    model.train()
            pause = stereo and step % 32 == 0 and step < total and memory_pressure()
            if step % 1000 == 0 or step == total or pause:
                checkpoint_training()
            if pause:
                record(step=step, paused_for_memory=True)
                raise SystemExit(75)
            if step == total:
                return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("d2", "silog", "fixed", "kf", "real"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=4243)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--val-every", type=int, default=4000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stereo-mode", choices=("off", "fixed", "adaptive"), default=argparse.SUPPRESS)
    parser.add_argument("--stereo-calibration",default=argparse.SUPPRESS)
    parser.add_argument("--pair-manifest",default=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.stride, args.epochs, args.val_every, args.log_every, args.steps or 1) < 1 or args.lr <= 0:
        parser.error("Training dimensions and learning rate must be positive")
    if hasattr(args, "pair_manifest") and not args.smoke:
        hold = Path(args.pair_manifest).with_suffix(".hold")
        if hold.is_file():
            print(f"Paired training deferred before GPU allocation. See {hold}", flush=True)
            raise SystemExit(76)
    from run_stereo_students import gpu_slot, yield_previous_training
    if args.arm!="real" and not hasattr(args,"pair_manifest"):
        yield_previous_training(args.output)
    with gpu_slot():
        if hasattr(args,"stereo_mode"):
            from run_stereo_students import capacity
            if not capacity():
                raise SystemExit(75)
        try:
            main(args)
        except torch.cuda.OutOfMemoryError:
            if not hasattr(args,"pair_manifest"):
                raise
            print("GPU memory pressure; release device and resume from the last atomic checkpoint.",flush=True)
            raise SystemExit(75)
