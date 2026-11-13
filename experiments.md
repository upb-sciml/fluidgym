# Experiments

All commands are run from the repository root (the scripts resolve `./output`,
`./local_data` and `./paper` relative to the working directory). Hydra scripts accept
`-m` for sweeps. Quote sweep values like `'seed=range(5)'` and `'case=glob(hartmann_*)'`,
because the shell would otherwise interpret the parentheses. By default (`cluster=local`)
a sweep runs on the current machine. To submit it as SLURM jobs through submitit, fill in
`runscripts/configs/cluster/my_cluster.yaml` and add `cluster=my_cluster` to the command.

## Directory structure

### `runscripts/`

```
runscripts/
├── configs/                     # hydra configs
│   ├── algorithm/               #   sb3_sac, sb3_ppo, shac
│   ├── case/                    #   validation cases (hartmann_*, shercliff_*, hunt_*, temp1_*, temp2_*, duct_2000)
│   ├── cluster/                 #   local (default), my_cluster (submitit SLURM template)
│   ├── env/                     #   benchmark env configs
│   ├── solver_ablation/         #   PICT vs. phiPICT benchmark variants
│   ├── preparation.yaml         #   initial domains + statistics
│   ├── train_sb3.yaml           #   SAC/PPO training  (test_sb3.yaml inherits it)
│   ├── train_bptt.yaml          #   BPTT training     (test_bptt.yaml inherits it)
│   ├── train_mineral.yaml       #   SHAC training     (test_mineral.yaml inherits it)
│   ├── optimize_cw_inverse.yaml #   inverse C_w design
│   └── benchmark.yaml           #   solver throughput benchmark
├── validation/                  # solver validation against analytic/reference solutions
│   ├── validation.py            #   Hartmann, Shercliff, Hunt flows
│   ├── validation_temp.py       #   magnetoconvection (temp1, temp2)
│   ├── validation_duct.py       #   turbulent MHD duct
│   ├── gradcheck.py             #   torch.autograd.gradcheck of the adjoints
│   ├── optim.py                 #   gradient-based recovery of the Stuart number
│   └── eval_*.py                #   plots/tables of the runs above -> paper/
├── preparation/                 # initial domains and uncontrolled baseflow statistics
│   ├── create_initial_domains.py
│   ├── collect_statistics.py
│   ├── aggregate_statistics.py
│   └── download_/upload_initial_domains.py   # HuggingFace dataset sync of local_data/
├── train/                       # training and testing of the controllers
│   ├── train_sb3.py / test_sb3.py           # SAC (Stable Baselines 3)
│   ├── train_bptt.py / test_bptt.py         # BPTT through the differentiable solver
│   ├── train_mineral.py / test_mineral.py   # SHAC (third_party/mineral)
│   └── optimize_cw_inverse.py               # inverse design of wall conductance ratios
├── evaluation/                  # paper figures/tables of the RL and inverse-design runs
│   ├── plot_experiments.py
│   ├── plot_rbc_mhd_actuation.py
│   └── download_/upload_experimental_results.py  # HuggingFace sync of output/
└── benchmark/                   # PICT vs. phiPICT throughput
    ├── benchmark.py
    └── ablate_solver_improvements.py        # solver switches, used by benchmark.py
```

### Outputs

Every hydra script runs inside its own output directory (`hydra.job.chdir: true`), which
also contains the resolved `config.yaml`, `overrides.yaml` and the log file.

```
local_data/initial_domains/<env dir>/          # e.g. rbc_mhd_3d_Ra20000_Q1000_Pr0.025_A5_N32_g2
├── domain_statistics.json                     #   aggregate_statistics.py
└── <domain_idx>/                              #   0..9
    ├── {train,val,test}.json/.npz             #   create_initial_domains.py
    └── {train,val,test}_uncontrolled_episode.csv  # collect_statistics.py

output/
├── validation/
│   ├── {hartmann,shercliff,hunt}/<Ha>/<nx>x<ny>/        # validation.py
│   ├── {temp1,temp2}/<Ha>_<Cw>/<nx>x<ny>x<nz>/          # validation_temp.py
│   ├── duct/<Ha>/<nx>x<ny>x<nz>_<duct>/                 # validation_duct.py
│   └── optim/{hartmann,shercliff,hunt}/                 # optim.py
├── preparation/<env_id>/<domain_idx>/                   # logs of the preparation jobs
├── training/<rl_mode>/<env_id>/<algorithm>/<seed>/      # rl_mode = marl, algorithm = SAC | BPTT10_mlp | SHAC
│   ├── training_log.csv                                 #   training curve
│   ├── <checkpoints>                                    #   ckpt_latest.zip | bptt_policy*.pth | ckpt/
│   └── test/<domain_idx>/                               #   test_*.py, one episode per test domain
├── optimize_cw_inverse/<optimizer>/<seed>/              # optimize_cw_inverse.py
└── benchmark/                                          # benchmark.py
    ├── <env_id>/benchmark.csv                          #   steps/s and memory per solver variant and batch size
    └── speedup.{csv,tex,pdf}                           #   speedup over PICT per env (aggregate)

paper/
├── plots/{,training,test,optim}/                        # figures (eval_*.py, plot_experiments.py)
└── tables/{,test}/                                      # LaTeX tables
```

## Validation

### Hartmann, Shercliff, Hunt

```bash
python runscripts/validation/validation.py -m 'case=glob(hartmann_*)'
python runscripts/validation/validation.py -m 'case=glob(shercliff_*)'
python runscripts/validation/validation.py -m 'case=glob(hunt_*)'
```

```bash
python runscripts/validation/eval_base.py --case hartmann --mode plot
python runscripts/validation/eval_base.py --case hartmann --mode error
python runscripts/validation/eval_base.py --case shercliff --mode plot
python runscripts/validation/eval_base.py --case shercliff --mode error
python runscripts/validation/eval_base.py --case hunt --mode plot
python runscripts/validation/eval_base.py --case hunt --mode error
```

### Magnetoconvection

```bash
python runscripts/validation/validation_temp.py -m 'case=glob(temp1_*)'
python runscripts/validation/validation_temp.py -m 'case=glob(temp2_*)'
```

```bash
python runscripts/validation/eval_temp.py --case temp1 --mode plot
python runscripts/validation/eval_temp.py --case temp1 --mode error
python runscripts/validation/eval_temp.py --case temp2 --mode plot
python runscripts/validation/eval_temp.py --case temp2 --mode error
```

### Duct

```bash
python runscripts/validation/validation_duct.py case=duct_2000
python runscripts/validation/eval_duct.py
```

### Gradcheck

```bash
python runscripts/validation/gradcheck.py --suite all --advection-scheme central
python runscripts/validation/gradcheck.py --suite all --advection-scheme linear_upwind
```

### Gradient Validation

```bash
python runscripts/validation/optim.py domain_type=hartmann
python runscripts/validation/optim.py domain_type=shercliff
python runscripts/validation/optim.py domain_type=hunt
python runscripts/validation/eval_optim.py
```

## RBC-MHD

### Preparation

Either download the prepared initial domains and statistics from HuggingFace into
`local_data/`
```bash
python runscripts/preparation/download_initial_domains.py
```

or generate them. First, simulate the train/val/test initial domains, one job per domain
index (without `domain_idxs`, every job would create all 10 domains)
```bash
python runscripts/preparation/create_initial_domains.py -m \
    env_id=RBCMHD3D-easy-v0,RBCMHD3D-medium-v0 \
    'domain_idx=range(10)' \
    '+domain_idxs=[${domain_idx}]'
```

Then, collect the uncontrolled baseflow episode of every domain
```bash
python runscripts/preparation/collect_statistics.py -m \
    env_id=RBCMHD3D-easy-v0,RBCMHD3D-medium-v0 \
    'domain_idx=range(10)'
```

and aggregate them into `domain_statistics.json`, which provides the reference values of
the reward
```bash
python runscripts/preparation/aggregate_statistics.py -m \
    env_id=RBCMHD3D-easy-v0,RBCMHD3D-medium-v0
```

### SAC

```bash
python runscripts/train/train_sb3.py -m \
    algorithm=sb3_sac \
    env_id=RBCMHD3D-easy-v0,RBCMHD3D-medium-v0 \
    rl_mode=marl \
    'seed=range(5)'
```

```bash
python runscripts/train/test_sb3.py -m \
    algorithm=sb3_sac \
    env_id=RBCMHD3D-easy-v0,RBCMHD3D-medium-v0 \
    rl_mode=marl \
    'seed=range(5)'
```

### BPTT

```bash
python runscripts/train/train_bptt.py -m \
    env_id=RBCMHD3D-easy-v0,RBCMHD3D-medium-v0 \
    rl_mode=marl \
    'seed=range(5)'
```

```bash
python runscripts/train/test_bptt.py -m \
    env_id=RBCMHD3D-easy-v0,RBCMHD3D-medium-v0 \
    rl_mode=marl \
    'seed=range(5)'
```

### SHAC

SHAC comes from the [mineral](https://github.com/etaoxing/mineral) library, included as
a git submodule pinned to the version used for the experiments. Install it once
```bash
git submodule update --init third_party/mineral
pip install -e third_party/mineral
```

```bash
python runscripts/train/train_mineral.py -m \
    algorithm=shac \
    env_id=RBCMHD3D-easy-v0,RBCMHD3D-medium-v0 \
    rl_mode=marl \
    'seed=range(5)'
```

```bash
python runscripts/train/test_mineral.py -m \
    algorithm=shac \
    env_id=RBCMHD3D-easy-v0,RBCMHD3D-medium-v0 \
    rl_mode=marl \
    'seed=range(5)'
```

The test scripts run in the training directory of each seed and, without `domain_idx`,
run one episode on each of the 10 test domains (`test/<domain_idx>/`).

## Inverse $C_w$

```bash
python runscripts/train/optimize_cw_inverse.py -m \
    optimizer=lbfgs,cma_es \
    'seed=range(5)'
```

## Solver Benchmark

Measure the throughput (env steps per second) of PICT, of phiPICT with one improvement
switched off at a time (`ablations` in `runscripts/configs/benchmark.yaml`, switches in
`configs/solver_ablation/`), and of phiPICT with batching, for every environment in
`configs/env/`. phiPICT is swept over the batch sizes in `batch_sizes`, and the best
one (within `max_memory_pct` of the GPU memory) is used for the comparison.
```bash
python runscripts/benchmark/benchmark.py -m 'env=glob(*)'
```

A single environment:
```bash
python runscripts/benchmark/benchmark.py env=cylinder_jet_2d_easy
```

Every environment writes `output/benchmark/<env_id>/benchmark.csv`. Aggregate them into
the speedup table and plot (`output/benchmark/speedup.{csv,tex,pdf}`)
```bash
python runscripts/benchmark/benchmark.py aggregate_only=true
```

## Plots and Tables

Either use the runs above, or download the published results into `output/`
```bash
python runscripts/evaluation/download_experimental_results.py
```

Then create the training curves, test tables and inverse-design figures in `paper/`
```bash
python runscripts/evaluation/plot_experiments.py
```

The figures and tables that are created are selected in the `__main__` block of
`plot_experiments.py`.

To visualize the actuation, plot the electric potential on a Hartmann wall of the
RBC-MHD environment after one random action
```bash
python runscripts/evaluation/plot_rbc_mhd_actuation.py --env-id RBCMHD3D-easy-v0
```

or after 20 steps of a test episode with a trained BPTT policy
```bash
python runscripts/evaluation/plot_rbc_mhd_actuation.py \
    --env-id RBCMHD3D-easy-v0 \
    --policy output/training/marl/RBCMHD3D-easy-v0/BPTT10_mlp/0/bptt_policy_best.pth \
    --max-steps 20
```

Both write `paper/plots/training/rbc_mhd_actuation.pdf` (set another path with
`--output`).
