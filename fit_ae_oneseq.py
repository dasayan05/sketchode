import os
import math
import random
import typing
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.distributions as dist
import torch.optim as optimizer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.cli import (LightningArgumentParser,
                                             LightningCLI,
                                             instantiate_class)
from wandb.util import generate_id
from wandb import Image

# local imports should go below this
from data.batching import BatchedStrokes
from data.dm import GenericDM, ReprType
from model.arch import (CDEOneSeqEncoder,
                        CDEStrokewiseEncoder,
                        ODEOneSeqDecoder,
                        GRUOneSeqEncoder,
                        GRUOneSeqDecoder, ODEStrokewiseDecoder)

# without this, torch.multiprocessing compaints about opening 'too many files'
# got from here: https://stackoverflow.com/questions/48250053/
torch.multiprocessing.set_sharing_strategy('file_system')


class FitAEOneSeq(pl.LightningModule):

    class EncType(str, Enum):
        cde = "cde"
        rnn = "rnn"

    class DecType(str, Enum):
        ode = "ode"
        rnn = "rnn"

    class DynamicsOrder(int, Enum):
        first = 1
        second = 2

    class Periodicity(str, Enum):
        SIN = 'sin'
        COS = 'cos'
        RELU = 'relu'
        TANH = 'tanh'

    def __init__(self,
                 opt_kwargs: dict,
                 sched_kwargs: dict,
                 rep: ReprType = ReprType.ONESEQ,
                 encoder_arch: typing.List[int] = [32, 32],
                 decoder_arch: typing.List[int] = [32, 32],
                 encoder_type: EncType = EncType.cde,
                 decoder_type: DecType = DecType.ode,
                 dec_dyn_order: DynamicsOrder = DynamicsOrder.second,
                 time_dependent: bool = True,
                 cos_annealing_period: int = 100,
                 dec_append_latent: bool = True,
                 vae: typing.Dict[str, typing.Union[int, float]] = {
                     'weight': 0.,
                     'start': 5000,
                     'end': 15000
                 },
                 n_hidden: int = 64,
                 n_latent: int = 48,
                 bounded_latent: bool = False,
                 ode_solver: str = 'rk4',
                 learn_freq: bool = False,
                 freq: typing.Dict[str, float] = {
                     'enc': 3.0,
                     'dec': 3.0
                 },
                 periodicity: Periodicity = Periodicity.COS,
                 scale_time: float = 5.0,
                 learn_time: bool = False,
                 observation_model: bool = True,
                 viz_grid: typing.Optional[int] = 24
                 ):
        """SketchODE model for continuous time free-form modelling

        Args:
            rep: data representation (oneseq or strokewise)
            encoder_arch: Architecture of the MLP for encoder dynamics
            decoder_arch: Architecture of the MLP for decoder dynamics
            encoder_type: Type of encoder
            decoder_type: Type of encoder
            dec_dyn_order: Order of the decoder dynamics, if 'ode'
            time_dependent: Decoder dynamics is time dependent ?
            cos_annealing_period: Annealing period of the optimizer scheduler
            dec_append_latent: Append latent to the dynamics function
            vae: VAE configurations (weight, start and end)
            n_hidden: Dimension of the hidden state (both rnn and ODE)
            n_latent: Dimension of the latent state
            bounded_latent: if the latent vectors are bounded by tanh()
            learning_rate: Learning rate of training
            ode_solver: Solver of ODE used in ODE/CDE (for now stick to 'rk4')
            learn_freq: Should frequency be learned ?
            periodicity: periodicity function (sin or cos)
            freq: fixed freq value or max freq in case of learn_freq=True {'enc': 2., 'dec': 2.}
            scale_time: T for ODE time period 0->T when learn_time=False, otherwise max time
            learn_time: whether to learn the time range
            weight_decay: L2 Regularization
            observation_model: Separate observation model or augmented state for ODEs
            viz_grid: How many samples to vizualize in a grid
        """
        super().__init__()
        self.save_hyperparameters()

        self.hp = self.hparams  # just an easier name to type

        if self.hp.rep == ReprType.ONESEQ:
            Encoder = GRUOneSeqEncoder if self.hp.encoder_type == 'rnn' else CDEOneSeqEncoder
            Decoder = GRUOneSeqDecoder if self.hp.decoder_type == 'rnn' else ODEOneSeqDecoder
        else:
            assert (self.hp.encoder_type == 'cde') and (self.hp.decoder_type == 'ode'), \
                "StrokeWise mode only supports CDE-ODE structure"
            Encoder = CDEStrokewiseEncoder
            Decoder = ODEStrokewiseDecoder

        self.encoder = Encoder(3 if self.hp.rep == ReprType.ONESEQ else 2,  # STROKEWISE do not require pen state
                               self.hp.n_hidden, self.hp.n_latent,
                               self.hp.encoder_arch,
                               ode_method=self.hp.ode_solver,
                               fixed_freq=not self.hp.learn_freq,
                               freq=self.hp.freq['enc'],
                               # if zero, no need for doing reparameterization
                               vae=(self.hp.vae['weight'] != 0.),
                               bounded_latent=self.hp.bounded_latent,
                               periodicity=self.hp.periodicity.value)

        self.decoder = Decoder(3 if self.hp.rep == ReprType.ONESEQ else 2,  # STROKEWISE do not require pen state
                               self.hp.n_hidden, self.hp.n_latent,
                               arch=self.hp.decoder_arch,
                               ode_method=self.hp.ode_solver,
                               order=self.hp.dec_dyn_order,
                               time_dependent=self.hp.time_dependent,
                               observation_model=self.hp.observation_model,
                               append_latent=self.hp.dec_append_latent,
                               fixed_freq=not self.hp.learn_freq,
                               freq=self.hp.freq['dec'],
                               periodicity=self.hp.periodicity.value)

        self.start_time = torch.nn.Parameter(torch.tensor(0.),
                                             requires_grad=self.hp.learn_time)
        self.duration_logit = torch.nn.Parameter(torch.tensor(0.),
                                                 requires_grad=self.hp.learn_time)

    @property
    def duration(self):
        if not self.hp.learn_time:
            return self.hp.scale_time
        else:
            return torch.sigmoid(self.duration_logit) * self.hp.scale_time

    def convert_time_range(self, ts):
        return (ts * self.duration) + self.start_time

    @property
    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def on_fit_start(self):
        # log parameter count as hyperparameter
        self.logger.log_hyperparams({'parameters': self.param_count})

        # Figure for producing visualizations in validation stage
        if self.hp.viz_grid is not None:
            self.figs, self.axes = [], []
            for _ in range(self.hp.viz_grid):
                fig, axis = plt.subplots(1, 1, figsize=(5, 5), frameon=False)
                self.figs.append(fig)
                self.axes.append(axis)

    def forward_oneseq(self, batch):
        x, ts = batch.points, batch.t_union
        latent = self.encoder(x, self.convert_time_range(ts))
        return self.decoder(latent, self.convert_time_range(ts),
                            # differentiating teacher-forcing and inference for RNN
                            x if self.training else x[0, ...].unsqueeze(0),
                            )

    def forward_strokewise(self, batch):
        lens, batches = batch
        ts = batches[0].t_union
        latent = self.encoder(batches, self.convert_time_range(ts))  # time-range for each stroke
        self.decoder.max_strokes = max(lens)  # TODO: this can be removed later
        return self.decoder(latent, self.convert_time_range(ts), ...)

    def forward(self, batch):
        if self.hp.decoder_type == 'ode':
            # if vae regularizer is 0, no need to compute the expensive CNF term
            self.decoder.dynamics.no_dlogp_compute = (self.vae_annealed_weight == 0.)

        if self.hp.rep == ReprType.ONESEQ:
            return self.forward_oneseq(batch)
        else:
            return self.forward_strokewise(batch)

    def loss_oneseq(self, out, batch):
        (traj_x, traj_p), logp_cnf = out
        x_loss = F.smooth_l1_loss(traj_x, batch.points[..., :-1], reduction='none')
        x_loss = x_loss.sum(-1).mean()
        p_loss = (- dist.Bernoulli(traj_p).log_prob(batch.points[..., -1])).mean()
        cnf_loss = - logp_cnf.mean()

        kld_loss = self.kld_loss()
        return x_loss, p_loss, kld_loss, cnf_loss, \
            (x_loss + p_loss + (kld_loss + cnf_loss) * self.vae_annealed_weight)

    def loss_strokewise(self, out, batch):
        lens, batches = batch
        (outx, outp), logp_cnf = out

        stroke_mask = torch.stack([lens - i for i in range(lens.max())], 0)
        torch.clamp_(stroke_mask, 0., 1)
        finish_mask = ((stroke_mask.cumsum(0) * stroke_mask) ==
                       lens.unsqueeze(0).repeat(max(lens), 1)).type(torch.int64)

        padded_gt_points = torch.stack([b.points for b in batches], 1)
        padded_out_x = torch.stack(outx, 1)
        padded_out_p = torch.stack([p.T for p in outp], 1).permute(1, 2, 0)

        x_loss = F.smooth_l1_loss(padded_out_x, padded_gt_points, reduction='none')
        x_loss = x_loss.sum(-1)
        x_loss = x_loss * stroke_mask.unsqueeze(0)  # broadcast in the seqlen dim
        x_loss = x_loss.mean()

        p_loss = (- dist.Categorical(logits=padded_out_p).log_prob(finish_mask) * stroke_mask).mean()
        cnf_loss = - logp_cnf.mean()

        kld_loss = self.kld_loss()
        return x_loss, p_loss, kld_loss, cnf_loss, \
            (x_loss + p_loss + (kld_loss + cnf_loss) * self.vae_annealed_weight)

    def loss(self, out, batch):
        if self.hp.rep == ReprType.ONESEQ:
            return self.loss_oneseq(out, batch)
        else:
            return self.loss_strokewise(out, batch)

    def kld_loss(self):
        if self.encoder.vae:
            mu, logvar = self.encoder.latent_mu, self.encoder.latent_logvar
            kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        else:
            kld_loss = 0.
        return kld_loss

    @property
    def vae_annealed_weight(self):
        annealing_start = self.hp.vae['start']
        annealing_stops = self.hp.vae['end']
        annealing_period = annealing_stops - annealing_start
        return self.hp.vae['weight'] * \
            min(1, max(self.global_step - annealing_start, 0) * (1. / annealing_period))

    def training_step(self, batch, _):
        out = self(batch)
        x_loss, p_loss, kld_loss, cnf_loss, loss = self.loss(out, batch)

        logged_quantities = {
            'train/loss': loss,
            'train/x_loss': x_loss,
            'train/p_loss': p_loss
        }
        if self.hp.vae['weight'] != 0.:
            logged_quantities.update({
                'train/kld_loss': kld_loss,
                'train/cnf_loss': cnf_loss,
                'train/kld_anneal': self.vae_annealed_weight
            })
        if self.hp.learn_time:
            logged_quantities.update({
                'time/start': self.start_time,
                'time/duration': self.duration,
                'time/end': self.start_time + self.duration
            })
        self.log_dict(logged_quantities,
                      prog_bar=False, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, _):
        out, logp = self(batch)
        *_, loss = self.loss((out, logp), batch)

        if self.hp.rep == ReprType.STROKEWISE:
            samples = BatchedStrokes.from_prediction_strokewise(*out)  # TODO: ts required ?
        else:
            samples = BatchedStrokes.from_prediction(*out, ts=batch.t_union)

        return {
            'loss': loss,
            'batch': samples
        }

    def validation_epoch_end(self, outputs):
        avg_loss = sum([out['loss'] for out in outputs]) / len(outputs)
        self.log('valid/loss', avg_loss, prog_bar=False, on_step=False, on_epoch=True)

        # choose a random validation batch
        batch: BatchedStrokes = random.choice([out['batch'] for out in outputs])

        for j, (fig, ax) in enumerate(zip(self.figs, self.axes)):
            ax.cla()
            if j < len(batch):
                batch[j].draw(axes=ax)
            fig.canvas.draw()

        self.logger.experiment.log({
            'valid/samples': [Image(fig) for fig in self.figs]
        })

    def configure_optimizers(self):
        lr_ratio = 1. / 50
        factor = 4.

        def annealing_policy(e):
            # sort-of inspired by https://math.stackexchange.com/a/1709348
            T = self.hp.cos_annealing_period * 2
            wt = (math.pi / T) * e
            gamma = math.log(factor) / math.pi
            return (1. - lr_ratio) * math.exp(-gamma * wt) * (math.cos(wt) ** 2) + lr_ratio

        optim = instantiate_class(self.parameters(), self.hparams.opt_kwargs)
        scheduler_name = self.hparams.sched_kwargs['class_path'].split('.')[-1]
        if scheduler_name == 'LambdaLR':
            self.hparams.sched_kwargs['init_args']['lr_lambda'] = annealing_policy
        sched = {
            'scheduler': instantiate_class(optim, self.hparams.sched_kwargs),
            'interval': 'step' if scheduler_name in ['CyclicLR', 'OneCycleLR'] else 'epoch',
            'frequency': 1
        }
        return [optim, ], [sched, ]


class CustomWandbLogger(WandbLogger):

    def __init__(self,
                 name: typing.Optional[str],
                 save_dir: typing.Optional[str] = 'logs',
                 group: typing.Optional[str] = 'common',
                 project: typing.Optional[str] = 'XXXXX',
                 log_model: typing.Optional[bool] = True,
                 offline: bool = False,
                 tags: typing.List[str] = ['common'],
                 entity: typing.Optional[str] = 'XXXXX'):
        rid = generate_id()
        name_rid = '-'.join([name, rid])
        super().__init__(name=name_rid, id=rid, offline=offline, tags=tags,
                         save_dir=os.path.join(save_dir, name_rid), project=project,
                         log_model=log_model, group=group, entity=entity)


class CustomCLI(LightningCLI):

    def fit(self) -> None:
        pass

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_optimizer_args(optimizer.AdamW,
                                  link_to='model.opt_kwargs')
        parser.add_lr_scheduler_args(
            (
                optimizer.lr_scheduler.CyclicLR,
                optimizer.lr_scheduler.LambdaLR,
                optimizer.lr_scheduler.OneCycleLR
            ),
            link_to='model.sched_kwargs')

        parser.link_arguments('model.rep', 'data.init_args.repr', apply_on='parse')


if __name__ == '__main__':
    cli = CustomCLI(FitAEOneSeq, GenericDM, subclass_mode_data=True,
                    trainer_defaults={
                        'flush_logs_every_n_steps': 10,
                        'log_every_n_steps': 1,
                        'num_sanity_val_steps': 0,
                        'benchmark': True,
                        'deterministic': False,
                        'terminate_on_nan': True,
                        'callbacks': [
                            LearningRateMonitor('step'),
                            ModelCheckpoint(monitor='valid/loss', save_top_k=3, save_last=True)
                        ]
                    })
    cli.trainer.fit(cli.model, cli.datamodule)
