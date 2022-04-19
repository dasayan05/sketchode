import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from umap import UMAP
from pytorch_lightning.utilities.cli import LightningArgumentParser

from data.dm import GenericDM, ReprType
from data.batching import BatchedStrokes
from fit_ae_oneseq import FitAEOneSeq, CustomCLI


class InferAEOneSeq(FitAEOneSeq):

    def on_test_start(self):
        if self.hp.interpolate or self.hp.latent_visualize or self.hp.conditional_sampling:
            self.fig, self.ax = plt.subplots(1, 1, figsize=(2, 2), frameon=True)

        if self.hp.decoder_type == 'ode':
            # if vae regularizer is 0, no need to compute the expensive CNF term
            self.decoder.dynamics.no_dlogp_compute = (self.vae_annealed_weight == 0.)

        # if self.hp.abstraction:
        #     self.fig2, self.ax2 = plt.subplots(5, 5, figsize=(6, 6), frameon=True)
        #     self.abs_dir = os.path.join(self.trainer.log_dir, 'abstraction')
        #     os.makedirs(self.abs_dir, exist_ok=True)

    def test_step(self, batch, batch_idx):
        if self.hp.rep == ReprType.ONESEQ:
            x, ts = batch.points, batch.t_union
            batch_size = batch.batch_size
            latent = self.encoder(x, self.convert_time_range(ts))
        else:
            lens, batches = batch
            ts = batches[0].t_union
            batch_size = batches[0].batch_size
            latent = self.encoder(batches, self.convert_time_range(ts))
            self.decoder.max_strokes = max(lens)

        if self.hp.conditional_sampling:
            batch_dir = os.path.join(self.trainer.log_dir, f'batch_{batch_idx}')
            os.makedirs(batch_dir, exist_ok=True)
            for n in range(self.hp.conditional_sampling_n):  # number of samples
                latent_jittered = latent + \
                    torch.randn_like(latent, device=latent.device) * \
                    self.hp.conditional_sampling_strength
                out_jittered, _ = self.decoder(latent_jittered, self.convert_time_range(ts),
                                               # its basically all zeros but just to be correct ..
                                               x[0, ...].unsqueeze(0) if self.hp.decoder_type == 'rnn' else ...)
                if self.hp.rep == ReprType.ONESEQ:
                    out_jittered = BatchedStrokes.from_prediction(*out_jittered)
                else:
                    out_jittered = BatchedStrokes.from_prediction_strokewise(*out_jittered)
                for b in range(batch_size):
                    sample_dir = os.path.join(batch_dir, f'sample_{b}')
                    os.makedirs(sample_dir, exist_ok=True)
                    self.ax.cla()
                    out_jittered[b].draw(self.ax, lims=[-1.5, 1])
                    self.fig.canvas.draw()
                    self.fig.savefig(
                        os.path.join(sample_dir, f'{n}.svg'),
                        bbox_inches='tight',
                        pad_inches=0.,
                        transparent=True
                    )

        # if self.hp.abstraction:
        #     for scale_dyn in np.linspace(2., 1.5, 10):
        #         out = self.decoder(latent, self.convert_time_range(self.ts),
        #                            # its basically all zeros but just to be correct ..
        #                            x[0, ...].unsqueeze(0) if self.hp.decoder_type == 'rnn' else ...,
        #                            scale_dyn=scale_dyn)
        #         out: BatchedStrokes = BatchedStrokes.from_prediction(*out)
        #         out.draw(self.ax2)
        #         self.fig2.canvas.draw()
        #         self.fig2.savefig(
        #             os.path.join(self.abs_dir, f'{batch_idx}_{scale_dyn}.png'),
        #             bbox_inches='tight',
        #             pad_inches=0.,
        #             transparent=False
        #         )

        if self.hp.interpolate:
            batch_perms = [torch.randperm(batch_size)
                           for _ in range(self.hp.interpolate_perm + 1)]
            batch_perms.append(batch_perms[0])  # for loopy animations

            for i in range(len(batch_perms) - 1):
                latent_batch1 = latent[batch_perms[i], :]
                latent_batch2 = latent[batch_perms[i+1], :]

                set_dir = os.path.join(self.trainer.log_dir, f'batch_{batch_idx}', f'set_{i}')
                os.makedirs(set_dir, exist_ok=True)
                for al, alpha in enumerate(np.linspace(0., 1., 20)):
                    interp_latent = latent_batch1 * (1. - alpha) + latent_batch2 * alpha
                    out_interp, _ = self.decoder(interp_latent, self.convert_time_range(ts),
                                                 # its basically all zeros but just to be correct ..
                                                 x[0, ...].unsqueeze(0) if self.hp.decoder_type == 'rnn' else ...)
                    if self.hp.rep == ReprType.ONESEQ:
                        batch_interp = BatchedStrokes.from_prediction(*out_interp)
                    else:
                        batch_interp = BatchedStrokes.from_prediction_strokewise(*out_interp)

                    for b in range(batch_size):
                        sample_dir = os.path.join(set_dir, f'{b}')
                        os.makedirs(sample_dir, exist_ok=True)
                        self.ax.cla()
                        batch_interp[b].draw(self.ax, lims=[-1.5, 1])
                        self.fig.canvas.draw()
                        self.fig.savefig(
                            os.path.join(set_dir, f'{b}/{i}_{al}.svg'),
                            bbox_inches='tight',
                            pad_inches=0.,
                            transparent=True
                        )

        return latent

    def test_epoch_end(self, outputs):
        if self.hp.latent_visualize:
            umap = UMAP()
            latents = torch.cat(outputs, 0).cpu().numpy()
            print('Running UMAP projections ..')
            proj_latents = umap.fit_transform(latents)
            self.ax.cla()
            self.ax.scatter(proj_latents[:, 0], proj_latents[:, 1], s=2, color='blue')
            self.fig.canvas.draw()
            save_dir = os.path.join(self.trainer.log_dir, 'umap')
            os.makedirs(save_dir, exist_ok=True)
            self.fig.savefig(
                os.path.join(save_dir, 'umap.png'),
                bbox_inches='tight',
                pad_inches=0.,
                transparent=False
            )

        return None


class TestCustomCLI(CustomCLI):

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        class TestSpecific:
            def __init__(self,
                         interpolate: bool = True,
                         interpolate_perm: int = 1,
                         latent_visualize: bool = False,
                         conditional_sampling: bool = False,
                         conditional_sampling_n: int = 10,
                         conditional_sampling_strength: float = 0.1) -> None:
                """Test specific hyperparams

                Args:
                    interpolate: play with latent space interpolation
                    latent_visualize: visualize latent space with UMAP
                    unconditional_sampling: unconditionally sample from latent
                    conditional_sampling: conditional sampling
                    conditional_sampling_n: how many samples
                    conditional_sampling_strength: std of gaussian for sampling
                """
                pass

        parser.add_class_arguments(TestSpecific, "test", instantiate=False)
        super().add_arguments_to_parser(parser)

    def instantiate_classes(self) -> None:
        super().instantiate_classes()
        self.model.hparams.update(self.config['test'])


if __name__ == '__main__':
    cli = TestCustomCLI(InferAEOneSeq, GenericDM, subclass_mode_data=True,
                        save_config_overwrite=True)
    cli.trainer.fit(cli.model, cli.datamodule)
    cli.trainer.test(cli.model, cli.datamodule)
