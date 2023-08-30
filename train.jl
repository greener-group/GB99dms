# Train the GB99dms force field
# Individual jobs calling grads.jl are spawned asynchronously for a cluster
#   configured with Slurm
# The script will require modification to work with the user's compute
#   resources, it is not designed to just work

using Molly
using EzXML
using Statistics

suffix = "1"
starting_epoch_n = 1
ff_dir = "."
out_dir = "training_$suffix"
grads_script = joinpath(ff_dir, "grads.jl")
log_file = joinpath(out_dir, "train.log")
starting_params_file = joinpath(ff_dir, "starting_params.txt")
starting_xml_fp  = joinpath(ff_dir, "a99SB-disp.xml")
starting_xml_fp2 = joinpath(ff_dir, "his_a99SB-disp.xml")
n_steps = 5_000_000
clip_norm_val = 0.1
loss_weight_rg = 0.0
loss_weight_torsion = 0.0
loss_weight_sep = 10 # 1 for no weight
friction = 0.1
dt = 0.001 # ps
n_steps_chunk = 100
n_epochs = 100
learning_rate = 4E-4
grad_frac_clamp_struc = 0.005
grad_frac_clamp_all = 0.03

structures = [
    # struc   n_reps  gradweight
    ("trp-cage"  , 1, 1.0), # 1 - 20
    ("htt19"     , 2, 1.0), # 2 - 19
    ("histatin-5", 2, 1.0), # 3 - 24
    ("bba"       , 1, 1.0), # 4 - 28
    ("gtt"       , 1, 1.0), # 5 - 35
    ("ntl9"      , 1, 1.0), # 6 - 39
    ("actr_n"    , 2, 1.0), # 7 - 36
    ("actr_c"    , 2, 1.0), # 8 - 35
]
n_structures = length(structures)
n_total_jobs = sum(s -> s[2], structures)

function param_weight(k)
    if endswith(k, "σ")
        return 0.02
    elseif startswith(k, "atom_H"           ) || startswith(k, "inter_GB_params_H") ||
           startswith(k, "inter_GB_radius_H") || startswith(k, "inter_GB_screen_H")
        return 0.02
    else
        return 1.0
    end
end

function report(msg...)
    println(msg...)
    open(log_file, "a") do of
        println(of, msg...)
    end
end

function running_jobs()
    job_lines = [""]
    read_jobs = false
    while !read_jobs
        try
            job_lines = readlines(`squeue -u username`)
            read_jobs = true
        catch
            # Pass
        end
    end
    running_job_list = String[]
    for line in job_lines
        job_name = split(strip(line))[3]
        if !startswith(strip(line), "JOBID") && occursin(Regex("^g$(suffix)_\\d+\$"), job_name)
            jobid = split(strip(line))[1]
            arrayid = split(jobid, "_")[end]
            # Don't need to account for the [1-2] case since we submit jobs individually
            if startswith(arrayid, "[")
                push!(running_job_list, job_name * "_" * arrayid[2:(end - 1)])
            else
                push!(running_job_list, job_name * "_" * arrayid)
            end
        end
    end
    return running_job_list
end

iscompleted(out_file) = count(l -> startswith(l, "Grad "       ), readlines(out_file)) > 50 ||
                        count(l -> startswith(l, "No gradients"), readlines(out_file)) >= 2

prev_params_file(epoch_n) = epoch_n == 1 ? "starting_params.txt" : "params_ep$(epoch_n - 1).txt"

function run_job(epoch_n, struc_n, rep_n)
    sbatch_file = "temp$(suffix)_struc$(struc_n)_ep$(epoch_n)_rep$rep_n.sh"
    params_file = joinpath(out_dir, prev_params_file(epoch_n))
    open(sbatch_file, "w") do of
        println(of, "#!/bin/bash")
        println(of, "#SBATCH --partition=gpu")
        println(of, "#SBATCH --gres=gpu:1")
        println(of, "#SBATCH --output=$out_dir/epoch_$epoch_n/g_$(rep_n)_%a.log")
        println(of, "#SBATCH --time=3-0")
        println(of, "#SBATCH --job-name=g$(suffix)_$rep_n")
        println(of, "#SBATCH --array=$(struc_n)-$(struc_n)")
        println(of, "hostname")
        println(of, "julia -t 1 $grads_script \$SLURM_ARRAY_TASK_ID $params_file 0 $n_steps gpu f32 $clip_norm_val $loss_weight_rg $loss_weight_torsion $loss_weight_sep $friction $dt $n_steps_chunk $rep_n")
    end

    run(`sbatch $sbatch_file`)
    sleep(5)
    rm(sbatch_file)
end

function run_jobs(epoch_n)
    grads_epoch_dir = joinpath(out_dir, "epoch_$epoch_n")
    isdir(grads_epoch_dir) || mkdir(grads_epoch_dir)

    completed = false
    check_counter = 0
    while !completed
        n_completed = 0
        running_job_list = running_jobs()
        # Submit longer-running jobs first
        for struc_n in n_structures:-1:1
            n_reps = structures[struc_n][2]
            for rep_n in 1:n_reps
                is_running = "g$(suffix)_$(rep_n)_$struc_n" in running_job_list
                if !is_running
                    out_file = joinpath(grads_epoch_dir, "g_$(rep_n)_$struc_n.log")
                    if !isfile(out_file)
                        report("-- Submitting struc $struc_n repeat $rep_n")
                        run_job(epoch_n, struc_n, rep_n)
                    elseif iscompleted(out_file)
                        n_completed += 1
                    else
                        # Jobs sometimes fail, so resubmit and keep error log
                        report("-- Resubmitting struc $struc_n repeat $rep_n")
                        ext_n = 1
                        while isfile(out_file * ".err$ext_n")
                            ext_n += 1
                        end
                        mv(out_file, out_file * ".err$ext_n")
                        run_job(epoch_n, struc_n, rep_n)
                    end
                end
            end
        end
        if n_completed == n_total_jobs
            report("---- $n_completed of $n_total_jobs jobs completed")
            completed = true
        else
            if check_counter % 60 == 0
                report("---- $n_completed of $n_total_jobs jobs completed")
            end
            check_counter += 1
            sleep(60)
        end
    end
end

function read_grad_file(fp)
    params = Dict{String, Float64}()
    for line in readlines(fp)
        if startswith(line, "Grad ")
            l, k, v = split(line)
            params[k] = parse(Float64, v)
        end
    end
    return params
end

function combine_grads(epoch_n)
    report("-- Calculating new parameters")
    previous_params_fp = joinpath(out_dir, prev_params_file(epoch_n))
    new_params_fp      = joinpath(out_dir, "params_ep$epoch_n.txt")

    previous_params = Dict{String, Float64}()
    for line in readlines(previous_params_fp)
        k, v = split(line)
        previous_params[k] = parse(Float64, v)
    end

    new_params = deepcopy(previous_params)

    for (si, (structure, n_reps, struc_grad_weight)) in enumerate(structures)
        struc_params = []
        for rep_n in 1:n_reps
            fp = joinpath(out_dir, "epoch_$epoch_n", "g_$(rep_n)_$si.log")
            params = read_grad_file(fp)
            if length(params) > 0
                push!(struc_params, deepcopy(params))
                report("-- Read grads from ", fp)
            else
                report("-- Read zero grads from ", fp)
            end
        end
        if length(struc_params) > 0
            for k in keys(first(struc_params))
                # Average the gradients over repeats of the same structure
                grad = mean(d -> d[k], struc_params)
                size_weight = inv(mean([median(abs.(values(d))) for d in struc_params]))
                grad_weighted = learning_rate * param_weight(k) * struc_grad_weight * size_weight * grad
                # Limit the change from each structure
                change_min, change_max = sort([
                    previous_params[k] * -grad_frac_clamp_struc,
                    previous_params[k] *  grad_frac_clamp_struc,
                ])
                new_params[k] -= clamp(grad_weighted, change_min, change_max)
            end
        end
    end

    if !isnothing(grad_frac_clamp_all)
        for k in keys(new_params)
            # Limit the overall change
            grad_min, grad_max = sort([
                previous_params[k] * (1 - grad_frac_clamp_all),
                previous_params[k] * (1 + grad_frac_clamp_all),
            ])
            new_params[k] = clamp(new_params[k], grad_min, grad_max)
        end
    end

    report("---- Parameter changes for epoch ", epoch_n)
    report("---- Param                         Before         Now        Diff   Frac_diff")
    pad_length = maximum(length.(keys(new_params)))
    ks = sort(collect(keys(new_params)))
    diffs, frac_diffs = Float64[], Float64[]

    open(new_params_fp, "w") do of
        for k in ks
            println(of, rpad(k, pad_length), "  ", new_params[k])
            diff = new_params[k] - previous_params[k]
            frac_diff = abs(diff / previous_params[k])
            push!(diffs, diff)
            push!(frac_diffs, frac_diff)
        end
    end

    for ki in sortperm(frac_diffs, rev=true)
        k = ks[ki]
        report(
            "---- ",
            rpad(k, pad_length), "  ",
            lpad(round(previous_params[k]; sigdigits=4), 10), "  ",
            lpad(round(new_params[k]; sigdigits=4), 10), "  ",
            lpad(round(diffs[ki]; sigdigits=4), 10), "  ",
            lpad(round(frac_diffs[ki]; sigdigits=4), 10),
        )
    end
end

function retrieve_param!(params_used, params, key, default)
    if haskey(params, key)
        params_used[key] = true
        return params[key]
    else
        return default
    end
end

function params_to_xml(epoch_n)
    prefix = joinpath(out_dir, epoch_n == 0 ? "starting_params" : "params_ep$epoch_n")
    params_fp = prefix * ".txt"
    out_fp    = prefix * ".xml"

    params = Dict{String, String}()
    for line in readlines(params_fp)
        k, v = split(line)
        params[k] = v
    end

    ForceFieldType = isdefined(Molly, :MolecularForceField) ? MolecularForceField : OpenMMForceField
    molly_ff = ForceFieldType(starting_xml_fp, starting_xml_fp2; units=false)
    scaled_res_charges = Dict{String, Dict{String, Float64}}()
    for res_name in keys(molly_ff.residue_types)
        charge_sum, scaled_charge_sum, abs_scaled_charge_sum = 0.0, 0.0, 0.0
        res_type = molly_ff.residue_types[res_name]
        for (atom_name, atom_type) in res_type.types
            param = "atom_$(atom_type)_charge_scale"
            ch = res_type.charges[atom_name]
            if haskey(params, param)
                scaled_ch = ch * parse(Float64, params[param])
            else
                scaled_ch = ch
            end
            charge_sum += ch
            scaled_charge_sum += scaled_ch
            abs_scaled_charge_sum += abs(scaled_ch)
        end
        charge_diff = scaled_charge_sum - charge_sum
        scaled_res_charges[res_name] = Dict()
        for (atom_name, atom_type) in res_type.types
            param = "atom_$(atom_type)_charge_scale"
            ch = res_type.charges[atom_name]
            if haskey(params, param)
                scaled_ch = ch * parse(Float64, params[param])
            else
                scaled_ch = ch
            end
            shifted_ch = scaled_ch - charge_diff * abs(scaled_ch) / abs_scaled_charge_sum
            scaled_res_charges[res_name][atom_name] = shifted_ch
        end
    end

    # Track parameters that have been written
    params_used = Dict(k => false for k in keys(params))

    ff_xml = parsexml(read(starting_xml_fp))
    ff = root(ff_xml)
    for entry in eachelement(ff)
        entry_name = entry.name
        if entry_name == "Residues"
            for residue in eachelement(entry)
                res_name = residue["name"]
                for atom_or_bond in eachelement(residue)
                    if atom_or_bond.name == "Atom"
                        atom_name = atom_or_bond["name"]
                        atom_or_bond["charge"] = string(scaled_res_charges[res_name][atom_name])
                    end
                end
            end
        elseif entry_name == "PeriodicTorsionForce"
            for torsion in eachelement(entry)
                torsion_type = join(
                    [torsion["type$i"] == "" ? "-" : torsion["type$i"] for i in 1:4], "/")
                for i in 1:20
                    if haskey(torsion, "k$i")
                        torsion["k$i"] = retrieve_param!(params_used, params,
                                            "inter_PT_$(torsion_type)_k_$i", torsion["k$i"])
                    end
                end
            end
        elseif entry_name == "NonbondedForce"
            entry["coulomb14scale"] = retrieve_param!(params_used, params, "inter_CO_weight_14",
                                                      entry["coulomb14scale"])
            entry["lj14scale"] = retrieve_param!(params_used, params, "inter_LJ_weight_14",
                                                 entry["lj14scale"])
            for atom_or_attr in eachelement(entry)
                if atom_or_attr.name == "Atom"
                    atom_type = atom_or_attr["type"]
                    atom_or_attr["sigma"] = retrieve_param!(params_used, params, "atom_$(atom_type)_σ",
                                                            atom_or_attr["sigma"])
                    atom_or_attr["epsilon"] = retrieve_param!(params_used, params, "atom_$(atom_type)_ϵ",
                                                              atom_or_attr["epsilon"])
                end
            end
        end
    end

    for k in keys(params_used)
        if !params_used[k] && !startswith(k, "inter_GB_") && !endswith(k, "_charge_scale")
            error("parameter $k not written to file, aborting")
        end
    end

    write(out_fp, ff_xml)
    xml_lines = readlines(out_fp)

    open(out_fp, "w") do of
        for line in xml_lines[1:(end - 1)]
            println(of, line)
        end
        println(of, "  <Script>")
        print(of, """
            import openmm
            import openmm.app as app
            import openmm.unit as unit
            from openmm import CustomGBForce, Discrete2DFunction
            from openmm.app import element as E
            from openmm.app.internal.customgbforces import GBSAGBnForce, _NUCLEIC_ACID_RESIDUES, \\
                _get_bonded_atom_list, m0, d0, strip_unit, _is_carboxylateO

            params_dict = {
            """)
        for k in keys(params)
            if startswith(k, "inter_GB_")
                println(of, "    '", k, "': ", params[k], ",")
            end
        end
        print(of, """
            }

            def _mbondi2_radii(topology, all_bonds=None):
                default_radius = 1.5
                element_to_const_radius = {
                    E.nitrogen:   10 * params_dict["inter_GB_radius_N"],
                    E.oxygen:     10 * params_dict["inter_GB_radius_O"],
                    E.fluorine:   1.5,
                    E.silicon:    2.1,
                    E.phosphorus: 1.85,
                    E.sulfur:     1.8,
                    E.chlorine:   1.7,
                }
                radii = [0] * topology.getNumAtoms()
                if all_bonds is None:
                    all_bonds = _get_bonded_atom_list(topology)
                for i, atom in enumerate(topology.atoms()):
                    element = atom.element
                    # Radius of H atom depends on element it is bonded to
                    if element in (E.hydrogen, E.deuterium):
                        bondeds = all_bonds[atom]
                        if bondeds[0].element is E.nitrogen:
                            radii[i] = 10 * params_dict["inter_GB_radius_H_N"]
                        else:
                            radii[i] = 10 * params_dict["inter_GB_radius_H"]
                    # Radius of C atom depends on what type it is
                    elif element is E.carbon:
                        radii[i] = 10 * params_dict["inter_GB_radius_C"]
                    # All other elements have fixed radii for all types/partners
                    else:
                        radii[i] = element_to_const_radius.get(element, default_radius)
                return radii # Converted to nanometers above

            def _mbondi3_radii(topology, all_bonds = None):
                if all_bonds is None:
                    all_bonds = _get_bonded_atom_list(topology)
                radii = _mbondi2_radii(topology, all_bonds=all_bonds)
                for i, atom in enumerate(topology.atoms()):
                    # carboxylate and HH/HE (ARG)
                    if _is_carboxylateO(atom, all_bonds):
                        radii[i] = 10 * params_dict["inter_GB_radius_O_CAR"]
                    elif atom.residue.name == "ARG":
                        if atom.name.startswith("HH") or atom.name.startswith("HE"):
                            radii[i] = 1.17
                return radii # Converted to nanometers above

            def _createEnergyTerms(force, solventDielectric, soluteDielectric, SA, cutoff, kappa, offset):
                cutoff = strip_unit(cutoff, unit.nanometer)
                kappa = strip_unit(kappa, unit.nanometer**-1)
                params = "; solventDielectric=%.16g; soluteDielectric=%.16g; kappa=%.16g; offset=%.16g" % (solventDielectric, soluteDielectric, kappa, offset)
                if cutoff is not None:
                    params += "; cutoff=%.16g" % cutoff
                if kappa &gt; 0:
                    force.addEnergyTerm("-0.5*138.935485*(1/soluteDielectric-exp(-kappa*B)/solventDielectric)*charge^2/B"+params,
                            CustomGBForce.SingleParticle)
                elif kappa &lt; 0:
                    # Do kappa check here to avoid repeating code everywhere
                    raise ValueError("kappa/ionic strength must be &gt;= 0")
                else:
                    force.addEnergyTerm("-0.5*138.935485*(1/soluteDielectric-1/solventDielectric)*charge^2/B"+params,
                            CustomGBForce.SingleParticle)
                if SA=="ACE":
                    force.addEnergyTerm(f"{params_dict['inter_GB_sa_factor']}*(radius+{params_dict['inter_GB_probe_radius']})^2*(radius/B)^6; radius=or+offset"+params, CustomGBForce.SingleParticle)
                elif SA is not None:
                    raise ValueError("Unknown surface area method: "+SA)
                if cutoff is None:
                    if kappa &gt; 0:
                        force.addEnergyTerm("-138.935485*(1/soluteDielectric-exp(-kappa*f)/solventDielectric)*charge1*charge2/f;"
                                            "f=sqrt(r^2+B1*B2*exp(-r^2/(4*B1*B2)))"+params, CustomGBForce.ParticlePairNoExclusions)
                    else:
                        force.addEnergyTerm("-138.935485*(1/soluteDielectric-1/solventDielectric)*charge1*charge2/f;"
                                            "f=sqrt(r^2+B1*B2*exp(-r^2/(4*B1*B2)))"+params, CustomGBForce.ParticlePairNoExclusions)
                else:
                    if kappa &gt; 0:
                        force.addEnergyTerm("-138.935485*(1/soluteDielectric-exp(-kappa*f)/solventDielectric)*charge1*charge2*(1/f-"+str(1/cutoff)+");"
                                            "f=sqrt(r^2+B1*B2*exp(-r^2/(4*B1*B2)))"+params, CustomGBForce.ParticlePairNoExclusions)
                    else:
                        force.addEnergyTerm("-138.935485*(1/soluteDielectric-1/solventDielectric)*charge1*charge2*(1/f-"+str(1/cutoff)+");"
                                            "f=sqrt(r^2+B1*B2*exp(-r^2/(4*B1*B2)))"+params, CustomGBForce.ParticlePairNoExclusions)

            SCREEN_PARAMETERS = { # Non-nucleic nucleic
                    E.hydrogen  : (params_dict["inter_GB_screen_H"], 1.696538 ),
                    E.carbon    : (params_dict["inter_GB_screen_C"], 1.268902 ),
                    E.nitrogen  : (params_dict["inter_GB_screen_N"], 1.4259728),
                    E.oxygen    : (params_dict["inter_GB_screen_O"], 0.1840098),
                    E.fluorine  : (0.5                             , 0.5      ),
                    E.phosphorus: (0.5                             , 1.5450597),
                    E.sulfur    : (-0.703469                       , 0.05     ),
                    None        : (0.5                             , 0.5      ),
            }
            SCREEN_PARAMETERS[E.deuterium] = SCREEN_PARAMETERS[E.hydrogen]

            def screen_parameter(atom):
                return SCREEN_PARAMETERS.get(atom.element, SCREEN_PARAMETERS[None])

            class GBSAGBnDMSForce(GBSAGBnForce):
                OFFSET = params_dict["inter_GB_offset"]

                def __init__(self, solventDielectric=78.5, soluteDielectric=1, SA=None, cutoff=None, kappa=0.0/unit.nanometer):
                    GBSAGBnForce.__init__(self, solventDielectric, soluteDielectric, SA, cutoff, kappa)

                # Note that D shadows H and S shadows O
                _atom_params = {
                    E.hydrogen:   [params_dict["inter_GB_params_H_α"], params_dict["inter_GB_params_H_β"], params_dict["inter_GB_params_H_γ"]],
                    E.deuterium:  [params_dict["inter_GB_params_H_α"], params_dict["inter_GB_params_H_β"], params_dict["inter_GB_params_H_γ"]],
                    E.carbon:     [params_dict["inter_GB_params_C_α"], params_dict["inter_GB_params_C_β"], params_dict["inter_GB_params_C_γ"]],
                    E.nitrogen:   [params_dict["inter_GB_params_N_α"], params_dict["inter_GB_params_N_β"], params_dict["inter_GB_params_N_γ"]],
                    E.oxygen:     [params_dict["inter_GB_params_O_α"], params_dict["inter_GB_params_O_β"], params_dict["inter_GB_params_O_γ"]],
                    E.sulfur:     [params_dict["inter_GB_params_O_α"], params_dict["inter_GB_params_O_β"], params_dict["inter_GB_params_O_γ"]],
                }
                _atom_params_nucleic = {
                    E.hydrogen:   [0.537050, 0.362861, 0.116704 ],
                    E.deuterium:  [0.537050, 0.362861, 0.116704 ],
                    E.carbon:     [0.331670, 0.196842, 0.093422 ],
                    E.nitrogen:   [0.686311, 0.463189, 0.138722 ],
                    E.oxygen:     [0.606344, 0.463006, 0.142262 ],
                    E.sulfur:     [0.606344, 0.463006, 0.142262 ],
                    E.phosphorus: [0.418365, 0.290054, 0.1064245],
                }
                _default_atom_params = [1.0, 0.8, 4.851]

                @classmethod
                def getStandardParameters(cls, topology):
                    natoms = topology.getNumAtoms()
                    radii = [[r / 10, 0, 0, 0, 0] for r in _mbondi3_radii(topology)]
                    for atom, rad in zip(topology.atoms(), radii):
                        if atom.residue.name in _NUCLEIC_ACID_RESIDUES:
                            rad[1] = screen_parameter(atom)[1]
                            for i, p in enumerate(cls._atom_params_nucleic.get(atom.element, cls._default_atom_params)):
                                rad[2 + i] = p
                        else:
                            rad[1] = screen_parameter(atom)[0]
                            for i, p in enumerate(cls._atom_params.get(atom.element, cls._default_atom_params)):
                                rad[2 + i] = p
                    return radii

                def _addEnergyTerms(self):
                    self.addPerParticleParameter("charge")
                    self.addPerParticleParameter("or") # Offset radius
                    self.addPerParticleParameter("sr") # Scaled offset radius
                    self.addPerParticleParameter("alpha")
                    self.addPerParticleParameter("beta")
                    self.addPerParticleParameter("gamma")
                    self.addPerParticleParameter("radindex")

                    n = len(self._uniqueRadii)
                    m0Table = self._createUniqueTable(m0)
                    d0Table = self._createUniqueTable(d0)
                    self.addTabulatedFunction("getd0", Discrete2DFunction(n, n, d0Table))
                    self.addTabulatedFunction("getm0", Discrete2DFunction(n, n, m0Table))

                    self.addComputedValue("I", "Ivdw+neckScale*Ineck;"
                                               "Ineck=step(radius1+radius2+neckCut-r)*getm0(radindex1,radindex2)/(1+100*(r-getd0(radindex1,radindex2))^2+"
                                               "0.3*1000000*(r-getd0(radindex1,radindex2))^6);"
                                               "Ivdw=select(step(r+sr2-or1), 0.5*(1/L-1/U+0.25*(r-sr2^2/r)*(1/(U^2)-1/(L^2))+0.5*log(L/U)/r), 0);"
                                               "U=r+sr2;"
                                               "L=max(or1, D);"
                                               "D=abs(r-sr2);"
                                               "radius1=or1+offset; radius2=or2+offset;"
                                               f"neckScale={params_dict['inter_GB_neck_scale']}; neckCut={params_dict['inter_GB_neck_cut']}; offset={params_dict['inter_GB_offset']}",
                                               CustomGBForce.ParticlePairNoExclusions)

                    self.addComputedValue("B", "1/(1/or-tanh(alpha*psi-beta*psi^2+gamma*psi^3)/radius);"
                                               f"psi=I*or; radius=or+offset; offset={params_dict['inter_GB_offset']}",
                                               CustomGBForce.SingleParticle)
                    _createEnergyTerms(self, self.solventDielectric, self.soluteDielectric, self.SA, self.cutoff, self.kappa, self.OFFSET)

            # Find the NonbondedForce.  We need it to look up charges, and also to set the reaction field dielectric to 1.

            nonbonded = [f for f in sys.getForces() if isinstance(f, openmm.NonbondedForce)]
            if len(nonbonded) != 1:
                raise ValueError('Implicit solvent requires the System to contain a single NonbondedForce')
            nonbonded = nonbonded[0]
            nonbonded.setReactionFieldDielectric(1)

            # Construct the CustomGBForce.

            argMap = {'soluteDielectric':'soluteDielectric', 'solventDielectric':'solventDielectric', 'implicitSolventKappa':'kappa'}
            solventArgs = {'SA':'ACE'}
            for key in argMap:
                if key in args:
                    solventArgs[argMap[key]] = args[key]
            if nonbondedMethod != app.NoCutoff:
                solventArgs['cutoff'] = nonbondedCutoff
            force = GBSAGBnDMSForce(**solventArgs)
            params_gb = GBSAGBnDMSForce.getStandardParameters(topology)
            for i, p in enumerate(params_gb):
                charge, sigma, epsilon = nonbonded.getParticleParameters(i)
                force.addParticle([charge, p[0], p[1], p[2], p[3], p[4]])
            force.finalize()

            # Set the nonbonded method and cutoff distance.

            if nonbondedMethod == app.NoCutoff:
                force.setNonbondedMethod(openmm.CustomNonbondedForce.NoCutoff)
            elif nonbondedMethod == app.CutoffNonPeriodic:
                force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffNonPeriodic)
                force.setCutoffDistance(nonbondedCutoff)
            elif nonbondedMethod == app.CutoffPeriodic:
                force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
                force.setCutoffDistance(nonbondedCutoff)
            else:
                raise ValueError("Illegal nonbonded method for use with implicit solvent")
            sys.addForce(force)
            """)
        println(of, "  </Script>")
        println(of, "</ForceField>")
    end
end

isdir(out_dir) || mkdir(out_dir)
cp(starting_params_file, joinpath(out_dir, "starting_params.txt"))
isfile(log_file) && error("log file $log_file already exists, aborting")

for epoch_n in starting_epoch_n:n_epochs
    report("Starting epoch ", epoch_n)
    params_to_xml(epoch_n - 1)
    run_jobs(epoch_n)
    combine_grads(epoch_n)
    report("Completed epoch ", epoch_n)
end

report("Finished")
