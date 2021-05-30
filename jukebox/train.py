"""
Ability to train vq-vae and prior
First try for random inputs
Then from maestros
"""
import sys
import fire
import warnings
import numpy as np
import torch as t
import jukebox.utils.dist_adapter as dist
from torch.nn.parallel import DistributedDataParallel

from jukebox.hparams import setup_hparams
from jukebox.make_models import make_nvae, make_prior, restore_opt, save_checkpoint
from jukebox.utils.logger import init_logging
from jukebox.utils.audio_utils import audio_preprocess, audio_postprocess
from jukebox.utils.torch_utils import zero_grad, count_parameters
from jukebox.utils.dist_utils import print_once, allreduce, allgather
from jukebox.utils.ema import CPUEMA, FusedEMA, EMA
from jukebox.utils.fp16 import FP16FusedAdam, FusedAdam, LossScalar, clipped_grad_scale, backward
from jukebox.data.data_processor import DataProcessor


def prepare_aud(x, hps):
    x = audio_postprocess(x.detach().contiguous(), hps)
    return allgather(x)

def log_aud(logger, tag, x, hps):
    logger.add_audios(tag, prepare_aud(x, hps), hps.sr, max_len=hps.max_len, max_log=hps.max_log)
    logger.flush()

def log_labels(logger, labeller, tag, y, hps):
    y = y.cpu().numpy()
    txt = ''
    for item in range(y.shape[0]):
        description = labeller.describe_label(y[item])
        artist, genre, lyrics = description['artist'], description['genre'], description['lyrics']
        txt += f'{item} artist:{artist}, genre:{genre}, lyrics:{lyrics}\n'
    logger.add_text(tag, txt)
    logger.flush()

def get_ddp(model, hps):
    rank = dist.get_rank()
    local_rank = rank % 8
    ddp = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, bucket_cap_mb=hps.bucket)
    return ddp

def get_ema(model, hps):
    mu = hps.mu or (1. - (hps.bs * hps.ngpus/8.)/1000)
    ema = None
    if hps.ema and hps.train:
        if hps.cpu_ema:
            if dist.get_rank() == 0:
                print("Using CPU EMA")
            ema = CPUEMA(model.parameters(), mu=mu, freq=hps.cpu_ema_freq)
        elif hps.ema_fused:
            ema = FusedEMA(model.parameters(), mu=mu)
        else:
            ema = EMA(model.parameters(), mu=mu)
    return ema


def get_optimizer(model, hps):
    # Optimizer
    betas = (hps.beta1, hps.beta2)
    
    
    if hps.fp16_opt:
        opt = FP16FusedAdam(model.parameters(), lr=hps.lr, weight_decay=hps.weight_decay, betas=betas, eps=hps.eps)
    else:
        opt = FusedAdam(model.parameters(), lr=hps.lr, weight_decay=hps.weight_decay, betas=betas, eps=hps.eps)

    # lr scheduler
    if hps.lr_use_linear_decay:
      shd = t.optim.lr_scheduler.CosineAnnealingLR(
        opt, float(hps.epochs - hps.lr_warmup - 1), eta_min=hps.learning_rate_min)

    restore_path = hps.restore_prior if hps.prior else hps.restore_vqvae
    restore_opt(opt, shd, restore_path)

    # fp16 dynamic loss scaler
    scalar = None
    if hps.fp16:
        rank = dist.get_rank()
        local_rank = rank % 8
        scalar = LossScalar(hps.fp16_loss_scale, scale_factor=2 ** (1./hps.fp16_scale_window))
        if local_rank == 0: print(scalar.__dict__)

    zero_grad(model)
    return opt, shd, scalar

def log_inputs(orig_model, logger, x_in, y, x_out, hps, tag="train"):
    print(f"Logging {tag} inputs/ouputs")
    x_ds = x_out.sample().permute(0,2,1)
    log_aud(logger, f'{tag}_x_in', x_in, hps)
    log_aud(logger, f'{tag}_x_out', x_ds, hps)
    bs = x_in.shape[0]
    logger.flush()

def sample_prior(orig_model, ema, logger, x_in, y, hps):
    if ema is not None: ema.swap()
    orig_model.eval()

    x_in = x_in[:hps.bs_sample]
    bs = x_in.shape[0]
    zs_in = orig_model.encode(x_in, start_level=0, bs_chunks=bs)
    assert len(zs_in) == hps.levels
    x_ds = [orig_model.decode(zs_in[level:], start_level=level, bs_chunks=bs) for level in range(0, hps.levels)]

    if not hps.labels:
        y = None
    elif hps.level == (hps.levels - 1):
        # Topmost level labels in order
        y = y[:hps.bs_sample]  # t.ones((hps.bs_sample, 1), device=y.device, dtype=t.long) * dist.get_rank()
    else:
        # Other levels keep labels to match x_cond
        y = y[:hps.bs_sample]

    # Temp 1.0
    _, *z_conds = orig_model.encode(x_in, bs_chunks=bs)
    z = orig_model.sample(hps.bs_sample, z_conds=z_conds, y=y, fp16=False, temp=1.0)
    x_sample = orig_model.decode([z, *z_conds], bs_chunks=bs)

    log_aud(logger, 'sample_x_T1', x_sample, hps)
    if hps.prior and hps.labels:
        log_labels(logger, orig_model.labeller, f'sample_x_T1', allgather(y.cuda()), hps)

    # Recons
    for i in range(len(x_ds)):
        log_aud(logger, f'x_ds_start_{i}', x_ds[i], hps)
    orig_model.train()
    if ema is not None: ema.swap()
    logger.flush()

def evaluate(model, orig_model, logger, metrics, data_processor, hps):
    model.eval()
    orig_model.eval()
    if hps.prior:
        _print_keys = dict(l="loss", bpd="bpd")
    else:
        _print_keys = dict(l="loss", rl="recons_loss", sl="spectral_loss")

    with t.no_grad():
        for i, x in logger.get_range(data_processor.test_loader):
            if isinstance(x, (tuple, list)):
                x, y = x
            else:
                y = None

            x = x.to('cuda', non_blocking=True)
            if y is not None:
                y = y.to('cuda', non_blocking=True)

            x_in = x = audio_preprocess(x, hps)
            log_input_output = (i==0)
            
            
            if hps.prior:
                forw_kwargs = dict(y=y, fp16=hps.fp16, decode=log_input_output)
            else:
                forw_kwargs = dict(global_step=global_step, args=hps)

            x_out, loss, _metrics = model(x, **forw_kwargs)

            # Logging
            for key, val in _metrics.items():
                _metrics[key] = val.item()
            _metrics["loss"] = loss = loss.item() # Make sure to call to free graph

            # Average and log
            for key, val in _metrics.items():
                _metrics[key] = metrics.update(f"test_{key}", val, x.shape[0])

            with t.no_grad():
                if log_input_output:
                    log_inputs(orig_model, logger, x_in, y, x_out, hps)

            logger.set_postfix(**{print_key:_metrics[key] for print_key, key in _print_keys.items()})

    for key, val in _metrics.items():
        logger.add_scalar(f"test_{key}", metrics.avg(f"test_{key}"))

    logger.close_range()
    return {key: metrics.avg(f"test_{key}") for key in _metrics.keys()}

def train(model, orig_model, global_step, opt, shd, scalar, ema, logger, metrics, data_processor, hps):
    model.train()
    orig_model.train()
    if hps.prior:
        _print_keys = dict(l="loss", bpd="bpd", kl_loss="kl_loss")
    else:
        _print_keys = dict(l="loss", rl="recon_loss", gn="gn", bn_loss ="bn_loss", norm_loss="norm_loss", wdn_coeff="wdn_coeff", kl_all="kl_all", kl_coeff="kl_coeff")

    for i, x in logger.get_range(data_processor.train_loader):
        if isinstance(x, (tuple, list)):
            x, y = x
        else:
            y = None

        x = x.to('cuda', non_blocking=True)
        if y is not None:
            y = y.to('cuda', non_blocking=True)

        x_in = x = audio_preprocess(x, hps)
        log_input_output = (logger.iters % hps.save_iters == 0)

        if hps.prior:
            forw_kwargs = dict(y=y, fp16=hps.fp16, decode=log_input_output)
        else:
            forw_kwargs = dict(global_step=global_step, args=hps)

        # Forward
        x_out, loss, _metrics = model(x, **forw_kwargs)

        # Backward
        loss, scale, grad_norm, overflow_loss, overflow_grad = backward(loss=loss, params=list(model.parameters()),
                                                                         scalar=scalar, fp16=hps.fp16, logger=logger)
        # Skip step if overflow
        grad_norm = allreduce(grad_norm, op=dist.ReduceOp.MAX)
        if overflow_loss or overflow_grad or grad_norm > hps.ignore_grad_norm > 0:
            zero_grad(orig_model)
            continue

        # Step opt. Divide by scale to include clipping and fp16 scaling
        logger.step()
        opt.step(scale=clipped_grad_scale(grad_norm, hps.clip, scale))
        zero_grad(orig_model)
        lr = hps.lr if shd is None else shd.get_lr()[0]
        if shd is not None: shd.step()
        if ema is not None: ema.step()
        next_lr = hps.lr if shd is None else shd.get_lr()[0]
        finished_training = (next_lr == 0.0)

        # Logging
        for key, val in _metrics.items():
            _metrics[key] = val.item()
        _metrics["loss"] = loss = loss.item() * hps.iters_before_update # Make sure to call to free graph
        _metrics["gn"] = grad_norm
        _metrics["lr"] = lr
        _metrics["lg_loss_scale"] = np.log2(scale)

        # Average and log
        for key, val in _metrics.items():
            _metrics[key] = metrics.update(key, val, x.shape[0])
            if logger.iters % hps.log_steps == 0:
                logger.add_scalar(key, _metrics[key])

        # Save checkpoint
        with t.no_grad():
            if hps.save and (logger.iters % hps.save_iters == 1 or finished_training):
                if ema is not None: ema.swap()
                orig_model.eval()
                name = 'latest' if hps.prior else f'step_{logger.iters}'
                if dist.get_rank() % 8 == 0:
                    save_checkpoint(logger, name, orig_model, opt, dict(step=logger.iters), hps)
                orig_model.train()
                if ema is not None: ema.swap()

        # Sample
        with t.no_grad():
            if (logger.iters % 12000) in list(range(1, 1 + hps.iters_before_update)) or finished_training:
                if hps.prior:
                    sample_prior(orig_model, ema, logger, x_in, y, hps)

        # Input/Output
        with t.no_grad():
            if log_input_output:
                log_inputs(orig_model, logger, x_in, y, x_out, hps)

        logger.set_postfix(**{print_key:_metrics[key] for print_key, key in _print_keys.items()})
        if finished_training:
            dist.barrier()
            exit()
    logger.close_range()
    return {key: metrics.avg(key) for key in _metrics.keys()}

def run(hps="teeny", port=29500, **kwargs):
    from jukebox.utils.dist_utils import setup_dist_from_mpi
    rank, local_rank, device = setup_dist_from_mpi(port=port)
    hps = setup_hparams(hps, kwargs)
    hps.ngpus = dist.get_world_size()
    hps.argv = " ".join(sys.argv)
    hps.bs_sample = hps.nworkers = hps.bs

    # Setup dataset
    data_processor = DataProcessor(hps)
    global_step=0
    hps.num_total_iter = len(data_processor.train_loader) * hps.epochs

    # Setup models
    nvae = make_nvae(hps, device)
    print_once(f"Parameters NVAE:{count_parameters(nvae)}")
    if hps.prior:
        prior = make_prior(hps, vqvae, device)
        print_once(f"Parameters Prior:{count_parameters(prior)}")
        model = prior
    else:
        model = nvae

    # Setup opt, ema and distributed_model.
    opt, shd, scalar = get_optimizer(model, hps)
    ema = get_ema(model, hps)
    distributed_model = get_ddp(model, hps)

    logger, metrics = init_logging(hps, local_rank, rank)
    logger.iters = model.step

    # Run training, eval, sample
    for epoch in range(hps.curr_epoch, hps.epochs):
        metrics.reset()
        data_processor.set_epoch(epoch)
        if epoch > hps.warmup_epochs:
            shd.step()
        if hps.train:

            train_metrics = train(distributed_model, model, global_step, opt, shd, scalar, ema, logger, metrics, data_processor, hps)
            train_metrics['epoch'] = epoch
            if rank == 0:
                print('Train',' '.join([f'{key}: {val:0.4f}' for key,val in train_metrics.items()]))
            dist.barrier()

        if hps.test:
            if ema: ema.swap()
            test_metrics = evaluate(distributed_model, model, logger, metrics, data_processor, hps)
            test_metrics['epoch'] = epoch
            if rank == 0:
                print('Ema',' '.join([f'{key}: {val:0.4f}' for key,val in test_metrics.items()]))
            dist.barrier()
            if ema: ema.swap()
        dist.barrier()

if __name__ == '__main__':
    fire.Fire(run)
