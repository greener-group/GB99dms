# GB99dms

This repository contains the force field, training scripts, simulation scripts and data described in the paper:

- Greener JG. Differentiable simulation to develop molecular dynamics force fields for disordered proteins, [Chemical Science](https://pubs.rsc.org/en/content/articlelanding/2024/sc/d3sc05230c) 15, 4897-4909 (2024)

Please cite the paper if you use the force field.
The validation trajectories and explicit solvent training trajectories are on [Zenodo](https://zenodo.org/record/8298226).
[Molly.jl](https://github.com/JuliaMolSim/Molly.jl) was used to carry out training.

## Usage

OpenMM is required, with a minimum version of 8.0.0.
Versions before this will give [silently incorrect results](https://github.com/openmm/openmm/pull/3505).

Download the [GB99dms.xml](https://github.com/greener-group/GB99dms/blob/main/GB99dms.xml) file and use it in OpenMM in the standard way for implicit solvent, i.e. write
```python
forcefield = ForceField("GB99dms.xml")
```
instead of something like
```python
forcefield = ForceField("amber14-all.xml", "implicit/gbn2.xml")
```
The protein parameters and implicit solvent parameters are both in the file and should generally be used together as this is how they were trained.

The force field was trained with a Debye-Hückel screening parameter κ of 0.7 nm^-1, corresponding to a salt concentration of about 100 mM at 300 K.
This can be set by passing `implicitSolventKappa=0.7/nanometer` to `forcefield.createSystem`, though it should be okay to use other values too.

A simulation script, used to run the validation simulations, is available at [sim.py](https://github.com/greener-group/GB99dms/blob/main/sim.py).
For example, to run a 2 μs simulation of Ntail starting from an extended conformation:
```bash
python sim.py structures/disordered/ntail.pdb GB99dms.xml traj
```
This will output `traj.dcd` and `traj.chk`.
Adding a final argument `CutoffPeriodic` means that periodic boundary conditions will be used, which can be useful when simulating multiple molecules.
For example, to simulate amyloid peptide aggregation:
```bash
python sim.py structures/ab_16-22/GB99dms/GB99dms_wt_1.pdb GB99dms.xml traj CutoffPeriodic
```

## Issues

If you want to use a small molecule force field alongside GB99dms, GAFF is recommended though you may need to modify the 1-4 scales in the GAFF XML file to match [those in GB99dms](https://github.com/greener-group/GB99dms/blob/main/GB99dms.xml#L3789). See more in [this issue](https://github.com/greener-group/GB99dms/issues/1).

Improper torsions are not applied to ASP and GLU sidechain carboxylates due to an issue in atom ordering in the XML file.

Simulations with phosphorus, for example those using phosphorylated amino acids, may lead to numerical instabilities as these parameters were not updated during training.

## Training

Training was carried out with Julia 1.8.5, Molly commit 7ea9481, Enzyme commit 2ccf4b7 and CUDA commit 654870d42 on the `vc/atomics` branch.
Since then a more stable setup has been found and Julia 1.9 with Molly 0.17.0, which installs Enzyme 0.11.2 and CUDA 4.2.0, should work.

The [grads.jl](https://github.com/greener-group/GB99dms/blob/main/grads.jl) script runs a simulation and calculates the gradient of the loss (based on residue-residue distance match to explicit solvent data) with respect to each of the force field parameters.
See the file for the options.
For example to run a 1000 step simulation for Trp-cage:
```bash
julia -t 1 grads.jl 1 starting_params.txt 0 1000
```
```
Structure file: structures/training/conf_1/trp-cage.pdb
Potential energy before minimisation: -8426.806 kJ/mol
Potential energy after  minimisation: -8441.581 kJ/mol
Primal compilation run
KL distance loss is 1.9348235
KL reverse distance loss is 0.55823725
KL torsion loss is 0.0
Mean Rg for recorded portion is 6.981576 Å
Rg loss is 0.0
Gradient compilation run
KL distance loss is 1.9347692
KL reverse distance loss is 0.55818874
KL torsion loss is 0.0
Mean Rg for recorded portion is 6.9816523 Å
Rg loss is 0.0
Gradient main run
KL distance loss is 1.2576079
KL reverse distance loss is 0.4246072
KL torsion loss is 0.0
Mean Rg for recorded portion is 6.9351954 Å
Rg loss is 0.0
Grad  inter_PT_N/CT/C/N_k_4     0.0024319272
Grad  inter_PT_C/N/CT/C_k_1     -0.00016169374
Grad  atom_C_ϵ                  0.02041649
Grad  inter_PT_C/N/CT/C_k_3     0.001478893
Grad  atom_HO_charge_scale      -0.00026885187
...
Grad  inter_GB_params_H_γ       0.17373642
Grad  atom_H_charge_scale       0.0029483766
Grad  inter_GB_radius_N         -0.1933675919426605
Grad  atom_C9_ϵ                 0.0002936928
Grad  inter_GB_params_N_γ       -0.14865768
  7.349788 seconds (10.68 M allocations: 576.511 MiB, 4.66% gc time)
Effective GPU memory usage: 6.08% (2.888 GiB/47.536 GiB)
Memory pool usage: 3.900 MiB (1.188 GiB reserved)

```
Various other parameters and options not used in the paper are also available in this script including double precision training, torsion loss, radius of gyration loss and finite differencing.

The [train.jl](https://github.com/greener-group/GB99dms/blob/main/train.jl) script trains the method.
Individual jobs calling grads.jl are spawned asynchronously for a cluster configured with Slurm.
The script will require modification to work with the user's compute resources, it is not designed to just work.
The GB99dms parameters were those after 5 epochs of training, which takes about 5 days on 12 GPUs.
Training was repeated 3 times and the run with the best performance on the training set was used.

## Data

This repository also includes:
- GB99dms parameters and parameter changes over training in [CSV](https://github.com/greener-group/GB99dms/blob/main/GB99dms.csv) format, and GB99dms parameters in [txt](https://github.com/greener-group/GB99dms/blob/main/GB99dms.txt) format.
- Starting parameters in [XML](https://github.com/greener-group/GB99dms/blob/main/starting_params.xml) and [txt](https://github.com/greener-group/GB99dms/blob/main/starting_params.txt) format. [a99SB-*disp* in XML format](https://github.com/greener-group/GB99dms/blob/main/a99SB-disp.xml), i.e. the starting parameters without the GBNeck2 implicit solvent model, is also available. This is not an official version of a99SB-*disp* and lacks the modified backbone O-H interaction term. It also has at least one incorrect improper torsion term (see issues above).
- [Starting structures](https://github.com/greener-group/GB99dms/tree/main/structures) used for training and validation.
- [Reference residue-residue distances](https://github.com/greener-group/GB99dms/tree/main/explicit_solv_distances) from explicit solvent simulations with a99SB-*disp*. [Torsions](https://github.com/greener-group/GB99dms/tree/main/explicit_solv_torsions) are also included though these were not used in the paper. 
