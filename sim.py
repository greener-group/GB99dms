# Run implicit solvent MD simulation
# Requires OpenMM v8.0.0 or later
# Arguments are PDB file path, force field XML file path, output prefix and
#   (optional) non-bonded method

from openmm.app import *
from openmm import *
from openmm.unit import *
from openmm.app import element as E
from math import ceil
import os
import sys

pdb_filepath = sys.argv[1]
ff_filepath  = sys.argv[2]
out_prefix   = sys.argv[3]
if len(sys.argv) > 4 and sys.argv[4] == "CutoffPeriodic":
    nonbondedMethod = CutoffPeriodic
else:
    nonbondedMethod = CutoffNonPeriodic

dt = 0.004 # ps
n_steps       = int(ceil(2e6 / dt)) # 2 μs
n_steps_equil = int(ceil(500 / dt)) # 500 ps
n_steps_save  = int(ceil(500 / dt)) # 500 ps
temperature = 300.0 * kelvin
kappa = 0.7 / nanometer
friction = 1 / picosecond
nonbondedCutoff = 2 * nanometer
restraint_fc = 500.0 # kJ/mol

pdb = PDBFile(pdb_filepath)
forcefield = ForceField(ff_filepath)
traj_fp       = out_prefix + ".dcd"
checkpoint_fp = out_prefix + ".chk"

system = forcefield.createSystem(
    pdb.topology,
    nonbondedMethod=nonbondedMethod,
    nonbondedCutoff=nonbondedCutoff,
    constraints=HBonds,
    hydrogenMass=2*amu,
    implicitSolventKappa=kappa,
)

integrator = LangevinMiddleIntegrator(temperature, friction, dt * picoseconds)
simulation = Simulation(pdb.topology, system, integrator)

if os.path.isfile(checkpoint_fp):
    print("Restarting from checkpoint")
    simulation.loadCheckpoint(checkpoint_fp)
    n_steps_to_run = n_steps - simulation.currentStep
    append_dcd = True
else:
    system_equil = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=nonbondedMethod,
        nonbondedCutoff=nonbondedCutoff,
        constraints=HBonds,
        hydrogenMass=2*amu,
        implicitSolventKappa=kappa,
    )

    # Positional restraints for all heavy-atoms for equilibration
    pos_res = CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2;")
    pos_res.addPerParticleParameter("k")
    pos_res.addPerParticleParameter("x0")
    pos_res.addPerParticleParameter("y0")
    pos_res.addPerParticleParameter("z0")

    for ai, atom in enumerate(pdb.topology.atoms()):
        if atom.element is not E.hydrogen:
            x = pdb.positions[ai][0].value_in_unit(nanometers)
            y = pdb.positions[ai][1].value_in_unit(nanometers)
            z = pdb.positions[ai][2].value_in_unit(nanometers)
            pos_res.addParticle(ai, [restraint_fc, x, y, z])

    system_equil.addForce(pos_res)
    integrator_equil = LangevinMiddleIntegrator(temperature, friction, dt * picoseconds)

    equilibration = Simulation(pdb.topology, system_equil, integrator_equil)
    equilibration.context.setPositions(pdb.positions)
    equilibration.minimizeEnergy()

    equilibration.context.setVelocitiesToTemperature(temperature)
    print("Equilibrating for", n_steps_equil, "steps")
    equilibration.step(n_steps_equil)

    simulation.context.setPositions( equilibration.context.getState(getPositions=True ).getPositions( ))
    simulation.context.setVelocities(equilibration.context.getState(getVelocities=True).getVelocities())
    n_steps_to_run = n_steps
    append_dcd = False

simulation.reporters.append(DCDReporter(traj_fp, n_steps_save, append=append_dcd))
simulation.reporters.append(CheckpointReporter(checkpoint_fp, n_steps_save))
simulation.reporters.append(StateDataReporter(
    sys.stdout,
    n_steps_save,
    step=True,
    potentialEnergy=True,
    temperature=True,
))
print("Running for", n_steps_to_run, "steps")
simulation.step(n_steps_to_run)
