# Semi-supervised learning with FixMatch

This repository includes different mini-batch sampling approaches for the semi-supervised setting
[[1]](#1) and an extension of FixMatch for semantic segmentation [[2]](#2).

Settings for
reproducing experiments in [[1]](#1) and [[2]](#2) are available in [experiments](./experiments/).

## Getting started

The environment can be built from [`conda_env.yaml`](./conda_env.yaml) using
[`conda`](https://conda.io/projects/conda/en/latest/index.html)
or [`mamba`](https://mamba.readthedocs.io/en/latest/).

```bash
mamba env create -f conda_env.yaml
```

We use [hydra](https://hydra.cc/) for configuring training runs and
[Weights&Biases](https://wandb.ai) for running hyperparameter search experiments and
logging results.

System configuration such as output and dataset paths is available in [`cfg/system`](cfg/system/).
Running on a new system requires adding a new config file for it. To point to the new
configuration simply add the environment variable `HYDRA_SYSTEM_CONFIG` with value matching
the name of the new config for the new system.

For example, to train on CIFAR10 with 4000 labeled samples and the uniform sampling:

```bash
PYTHONPATH=. HYDRA_SYSTEM_CONFIG=local_gpu python scripts/train_and_evaluate.py --config-name=cifar10 dataset.num_labels=4000 dataset.use_implicit_setting=true +ssl=fixmatch
```

To evaluate on the test set for a checkpoint with name `<filename>` resulting from a W&B run with path
`<entity>/<project>/<run_id>`:

```bash
python scripts/evaluate.py --config-name=cifar10 evaluation=test +checkpoint=wandb checkpoint.path=<entity>/<project>/<run_id>/<filename> +wandb.project_name=cifar10_ssl
```

Multi-GPU and mixed precision training support are available via
[🤗Accelerate](https://github.com/huggingface/accelerate).
For mixed precision with FP16 use flag `system.fp16=true`.
For multi-GPU training, launch the script using `torch.distributed.run`
or `accelerate launch` launchers:

```bash
python -m torch.distributed.run --nproc_per_node=<num-gpus> scripts/train_and_evaluate.py ... system.fp16=true
```

To run on a SLURM cluster you can use [`scripts/run_cluster.sbatch`](./scripts/run_cluster.sbatch)
for single runs:

```bash
sbatch scripts/run_cluster.sbatch ssl 'python scripts/train_and_evaluate.py ...'
```

and [`scripts/run_wandb_sweep`](./scripts/run_wandb_sweep) for running experiments
with W&B Sweeps:

```bash
wandb sweep <experiment-file-path>  # returns a wandb sweep id
bash scripts/run_wandb_sweep <num-runs> <num-agents> scripts/run_cluster.sbatch ssl <wandb-entity>/<wandb-project>/<sweep-id>
```

## Citation

<a id="1">[1]</a> 
[[NLDL22] An analysis of over-sampling labeled data in semi-supervised learning with FixMatch](https://septentrio.uit.no/index.php/nldl/article/view/6269)

Miquel Martí i Rabadán, Sebastian Bujwid, Alessandro Pieropan, Hossein Azizpour, Atsuto Maki.

```bibtex
@inproceedings{oversampling_ssl_fixmatch,
  title={An analysis of over-sampling labeled data in semi-supervised learning with FixMatch},
  author={Martí, Miquel and Bujwid, Sebastian and Pieropan, Alessandro and Azizpour, Hossein and Maki, Atsuto},
  booktitle={Proceedings of the Northern Lights Deep Learning Workshop},
  volume={3},
  year={2022},
  doi={10.7557/18.6269}
}
```

<a id="2">[2]</a> 
[Dense FixMatch: a simple semi-supervised learning method for pixel-wise prediction tasks (2022)](https://arxiv.org/abs/2210.09919)

Miquel Martí i Rabadán, Alessandro Pieropan, Hossein Azizpour, Atsuto Maki.

```bibtex
@misc{dense_fixmatch,
  title = {Dense FixMatch: a simple semi-supervised learning method for pixel-wise prediction tasks},
  author = {Rabadán, Miquel Martí i and Pieropan, Alessandro and Azizpour, Hossein and Maki, Atsuto},
  publisher = {arXiv},
  year = {2022},
  url = {https://arxiv.org/abs/2210.09919},
}
```

_Copyright &copy; 2022 [Univrses AB](https://univrses.com/)_
