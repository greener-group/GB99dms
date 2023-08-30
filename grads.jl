# Take gradients through a simulation with AD or FD
# See README.md for required software versions
# Required arguments are structure number and parameters txt file
# Optional arguments are below

using Molly
using CUDA
using FiniteDifferences
using Zygote
using DelimitedFiles
using LinearAlgebra
using Random
using Statistics

const structure_n         = parse(Int, ARGS[1])
const params_fp           = ARGS[2]
const device_n            = length(ARGS) >= 3  ? parse(Int, ARGS[3])                    : 0
const n_steps             = length(ARGS) >= 4  ? parse(Int, ARGS[4])                    : 5_000_000
const gpu                 = length(ARGS) >= 5  ? ARGS[5] == "gpu"                       : true
const T                   = length(ARGS) >= 6  ? (ARGS[6] == "f32" ? Float32 : Float64) : Float32
const clip_norm_val       = length(ARGS) >= 7  ? parse(T  , ARGS[7 ])                   : T(0.1)
const loss_weight_rg      = length(ARGS) >= 8  ? parse(T  , ARGS[8 ])                   : T(0.0)
const loss_weight_torsion = length(ARGS) >= 9  ? parse(T  , ARGS[9 ])                   : T(0.0)
const loss_weight_sep     = length(ARGS) >= 10 ? parse(Int, ARGS[10])                   : 10
const friction            = length(ARGS) >= 11 ? parse(T  , ARGS[11])                   : T(0.1)
const dt                  = length(ARGS) >= 12 ? parse(T  , ARGS[12])                   : T(0.001) # ps
const n_steps_chunk       = length(ARGS) >= 13 ? parse(Int, ARGS[13])                   : 100
const repeat_n            = length(ARGS) >= 14 ? parse(Int, ARGS[14])                   : 1
const use_fd              = length(ARGS) >= 15 ? ARGS[15] == "fd"                       : false

const structures = [
    ("trp-cage"  , T(300.0), T( 6.9)), # 1 - 20
    ("htt19"     , T(300.0), T(11.2)), # 2 - 19
    ("histatin-5", T(300.0), T(12.9)), # 3 - 24
    ("bba"       , T(300.0), T( 9.1)), # 4 - 28
    ("gtt"       , T(300.0), T( 9.4)), # 5 - 35
    ("ntl9"      , T(300.0), T( 8.3)), # 6 - 39
    ("actr_n"    , T(300.0), T(16.4)), # 7 - 36
    ("actr_c"    , T(300.0), T(16.2)), # 8 - 35
]
const structure, temp, target_rg = structures[structure_n]

if gpu
    CUDA.allowscalar(true)
    device!(device_n)
    CUDA.limit!(CUDA.CU_LIMIT_MALLOC_HEAP_SIZE, 1*1024^3)
end

const ff_dir = "."
const n_threads = 1
const n_steps_distance = 5000
const print_grad_norm = true
const n_steps_init = n_steps_chunk * 5
const clipping = "norm"
const use_kl_rev_loss = true
const traj_frac_loss = T(0.0)
const traj_frac_loss_rg = T(0.2)
const rg_tolerance = T(1.0) # No loss for this many Å either side of target Rg
const rand_seed_fd = rand(UInt32)
const kappa = T(0.7)
const backbone_atoms = ("N", "CA", "C", "O")

function read_target_dists(fp)
    arrs_flat = readdlm(fp, T)
    n_res = Int(sqrt(size(arrs_flat)[2]))
    dists_mean = reshape(arrs_flat[1, :], n_res, n_res)
    dists_std  = reshape(arrs_flat[2, :], n_res, n_res)
    return dists_mean, dists_std
end

function read_target_torsions(fp)
    arrs_flat = readdlm(fp, T)
    return [arrs_flat[:, i] for i in 1:8]
end

const target_dists_mean, target_dists_std = read_target_dists(joinpath(
    ff_dir, "explicit_solv_distances", "$structure.txt"))
const n_res = size(target_dists_mean, 1)

const target_sinϕs_mean, target_sinϕs_std, target_cosϕs_mean, target_cosϕs_std,
target_sinψs_mean, target_sinψs_std, target_cosψs_mean, target_cosψs_std = read_target_torsions(
    joinpath(ff_dir, "explicit_solv_torsions", "$structure.txt"))

const loss_weight_matrix = [min(abs(i - j) / T(loss_weight_sep), one(T)) for i in 1:n_res, j in 1:n_res]

params_dic = Dict{String, T}()
for line in readlines(params_fp)
    k, v = split(line)
    params_dic[k] = parse(T, v)
end
const pad_length = maximum(length.(keys(params_dic)))

# This was named differently in earlier versions of Molly
const ForceFieldType = isdefined(Molly, :MolecularForceField) ? MolecularForceField : OpenMMForceField

const ff = ForceFieldType(
    T,
    joinpath(ff_dir, "a99SB-disp.xml"),
    joinpath(ff_dir, "his_a99SB-disp.xml");
    units=false,
)

if repeat_n == 2 && isfile(joinpath(ff_dir, "structures", "training", "conf_2", "$structure.pdb"))
    pdb_fp = joinpath(ff_dir, "structures", "training", "conf_2", "$structure.pdb")
else
    pdb_fp = joinpath(ff_dir, "structures", "training", "conf_1", "$structure.pdb")
end
println("Structure file: ", pdb_fp)

s_init = System(
    pdb_fp,
    ff;
    boundary=CubicBoundary(T(500.0)),
    units=false,
    gpu=gpu,
    dist_cutoff=T(240.0),
    dist_neighbors=T(245.0),
    implicit_solvent="gbn2",
    kappa=kappa,
)

const calpha_inds   = [i for i in 1:length(s_init) if s_init.atoms_data[i].atom_name == "CA"]
const backbone_inds = [i for i in 1:length(s_init) if s_init.atoms_data[i].atom_name in backbone_atoms]
const ϕ_inds_i      = [i for i in 1:length(s_init) if s_init.atoms_data[i].atom_name == "C" ][1:(end - 1)]
const ϕ_inds_j      = [i for i in 1:length(s_init) if s_init.atoms_data[i].atom_name == "N" ][2:end]
const ϕ_inds_k      = [i for i in 1:length(s_init) if s_init.atoms_data[i].atom_name == "CA"][2:end]
const ϕ_inds_l      = [i for i in 1:length(s_init) if s_init.atoms_data[i].atom_name == "C" ][2:end]
const ψ_inds_i      = [i for i in 1:length(s_init) if s_init.atoms_data[i].atom_name == "N" ][1:(end - 1)]
const ψ_inds_j      = [i for i in 1:length(s_init) if s_init.atoms_data[i].atom_name == "CA"][1:(end - 1)]
const ψ_inds_k      = [i for i in 1:length(s_init) if s_init.atoms_data[i].atom_name == "C" ][1:(end - 1)]
const ψ_inds_l      = [i for i in 1:length(s_init) if s_init.atoms_data[i].atom_name == "N" ][2:end]

const present_res_names = [
    Set([ad.res_name for ad in s_init.atoms_data])...,
    "N" * s_init.atoms_data[1  ].res_name,
    "C" * s_init.atoms_data[end].res_name,
]

atoms, pairwise_inters, specific_inter_lists, general_inters = inject_gradients(s_init, params_dic)
s_ref = System(
    atoms=atoms,
    atoms_data=s_init.atoms_data,
    pairwise_inters=pairwise_inters,
    specific_inter_lists=specific_inter_lists,
    general_inters=general_inters,
    coords=s_init.coords,
    velocities=s_init.velocities,
    boundary=s_init.boundary,
    neighbor_finder=s_init.neighbor_finder,
    force_units=NoUnits,
    energy_units=NoUnits,
)

E_start = potential_energy(s_ref, find_neighbors(s_ref; n_threads=n_threads); n_threads=n_threads)
println("Potential energy before minimisation: ", E_start, " kJ/mol")
minimizer = SteepestDescentMinimizer(step_size=T(0.01), tol=T(1_000.0))
simulate!(s_ref, minimizer; n_threads=n_threads)
E_min = potential_energy(s_ref, find_neighbors(s_ref; n_threads=n_threads); n_threads=n_threads)
println("Potential energy after  minimisation: ", E_min, " kJ/mol")

const coords_start = deepcopy(s_ref.coords)
random_velocities!(s_ref, temp)
const velocities_start = deepcopy(s_ref.velocities)
const simulator = Langevin(dt=dt, temperature=temp, friction=friction, remove_CM_motion=10)

# 1 is the reference (P), 2 is the model (Q)
function kl_divergence(μ1, σ1, μ2, σ2)
    if Molly.iszero_value(μ1)
        return zero(μ1) # Same residue case, use zero to work with Dual
    else
        return log(σ2 / σ1) + ((σ1 ^ 2 + (μ1 - μ2) ^ 2) / (2 * σ2 ^ 2)) - T(0.5)
    end
end

function inject_charge_gradients(atoms_nocharge, atoms_data, params_dic)
    scaled_res_charges = Dict{String, Dict{String, T}}()
    for res_name in present_res_names
        charge_sum, scaled_charge_sum, abs_scaled_charge_sum = zero(T), zero(T), zero(T)
        res_type = ff.residue_types[res_name]
        for (atom_name, atom_type) in res_type.types
            param = "atom_$(atom_type)_charge_scale"
            ch = res_type.charges[atom_name]
            if haskey(params_dic, param)
                scaled_ch = ch * params_dic[param]
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
            if haskey(params_dic, param)
                scaled_ch = ch * params_dic[param]
            else
                scaled_ch = ch
            end
            shifted_ch = scaled_ch - charge_diff * abs(scaled_ch) / abs_scaled_charge_sum
            scaled_res_charges[res_name][atom_name] = shifted_ch
        end
    end

    atoms_charge = map(Array(atoms_nocharge), atoms_data) do at, at_data
        if at_data.res_number == 1
            res_name = "N" * at_data.res_name
        elseif at_data.res_number == n_res
            res_name = "C" * at_data.res_name
        else
            res_name = at_data.res_name
        end
        ch = scaled_res_charges[res_name][at_data.atom_name]
        Atom(at.index, ch, at.mass, at.σ, at.ϵ, at.solute)
    end
    return typeof(atoms_nocharge)(atoms_charge)
end

function gradient_clip_norm(arr, max_grad_norm)
    if max_grad_norm > clip_norm_val
        return arr .* clip_norm_val / max_grad_norm
    else
        return arr
    end
end

gradient_clip_norm(::Nothing, max_grad_norm) = nothing

function run_sim_chunk(coords, velocities, atoms, pairwise_inters, specific_inter_lists,
                       general_inters, s_ref, n_steps, rand_seed)
    sys = System(
        atoms=atoms,
        pairwise_inters=pairwise_inters,
        specific_inter_lists=specific_inter_lists,
        general_inters=general_inters,
        coords=coords,
        velocities=velocities,
        boundary=s_ref.boundary,
        neighbor_finder=s_ref.neighbor_finder,
        force_units=NoUnits,
        energy_units=NoUnits,
    )

    simulate!(sys, simulator, n_steps; n_threads=n_threads, rng=Xoshiro(rand_seed))

    return sys.coords, sys.velocities
end

sum_abs2(x) = sum(abs2, x)

# Avoid NaN gradients at zero
sqrt_grad_safe(x) = x > zero(x) ? sqrt(x) : zero(x)

Zygote.accum(x::CuArray{<:SVector}, y::Vector{<:SizedVector}, z::CuArray{<:SVector}) = Zygote.accum(
    x, CuArray(convert(Vector{SVector{3, Float32}}, y)), z)

function loss(params_dic, s_ref, n_steps)
    use_fd && Random.seed!(rand_seed_fd)
    coords_native = deepcopy(coords_start)
    coords = deepcopy(coords_native)
    velocities = deepcopy(velocities_start)
    atoms_nocharge, pairwise_inters, specific_inter_lists, general_inters = inject_gradients(s_ref, params_dic)
    atoms = inject_charge_gradients(atoms_nocharge, s_ref.atoms_data, params_dic)

    n_chunks = Int(ceil(n_steps / n_steps_chunk))
    n_tors = n_res - 1
    dists_sum    = zeros(T, n_res, n_res)
    dists_sq_sum = zeros(T, n_res, n_res)
    sinϕs_sum, sinϕs_sq_sum = zeros(T, n_tors), zeros(T, n_tors)
    cosϕs_sum, cosϕs_sq_sum = zeros(T, n_tors), zeros(T, n_tors)
    sinψs_sum, sinψs_sq_sum = zeros(T, n_tors), zeros(T, n_tors)
    cosψs_sum, cosψs_sq_sum = zeros(T, n_tors), zeros(T, n_tors)
    n_steps_dist = n_steps > n_steps_distance ? n_steps_distance : 10 # Allow testing with fewer steps
    rg_sum = zero(T)
    dists_count, rg_count = 0, 0

    for ci in 1:n_chunks
        if clipping == "norm"
            max_grad_norm = max(
                norm(reinterpret(T, coords    )),
                norm(reinterpret(T, velocities)),
            )
            coords     = Zygote.hook(arr -> gradient_clip_norm(arr, max_grad_norm), coords    )
            velocities = Zygote.hook(arr -> gradient_clip_norm(arr, max_grad_norm), velocities)
        end
        if print_grad_norm && !use_fd && ci % 1_000 == 0
            coords = Zygote.hook(x -> println("Norm of gradient ", ci, " - ",
                                    norm(reinterpret(T, x))), coords)
        end

        # Use the same random seed when repeating the run with checkpointing
        rand_seed = rand(UInt32)
        coords, velocities = Zygote.checkpointed(run_sim_chunk, coords, velocities, atoms,
                    pairwise_inters, specific_inter_lists, general_inters, s_ref,
                    n_steps_chunk, rand_seed)

        if ci / n_chunks > traj_frac_loss
            if (ci * n_steps_chunk) % n_steps_dist == 0
                dists_count += 1
                dists_sq = sum_abs2.(Array(displacements(coords[calpha_inds], s_ref.boundary)))
                dists_sum += sqrt_grad_safe.(dists_sq)
                dists_sq_sum += dists_sq
                coords_arr = Array(coords)
                ϕs = torsion_angle.(coords_arr[ϕ_inds_i], coords_arr[ϕ_inds_j], coords_arr[ϕ_inds_k],
                                    coords_arr[ϕ_inds_l], (s_ref.boundary,))
                ψs = torsion_angle.(coords_arr[ψ_inds_i], coords_arr[ψ_inds_j], coords_arr[ψ_inds_k],
                                    coords_arr[ψ_inds_l], (s_ref.boundary,))
                sinϕs, cosϕs = sin.(ϕs), cos.(ϕs)
                sinψs, cosψs = sin.(ψs), cos.(ψs)
                sinϕs_sum    += sinϕs
                sinϕs_sq_sum += sinϕs .^ 2
                cosϕs_sum    += cosϕs
                cosϕs_sq_sum += cosϕs .^ 2
                sinψs_sum    += sinψs
                sinψs_sq_sum += sinψs .^ 2
                cosψs_sum    += cosψs
                cosψs_sq_sum += cosψs .^ 2
            end
        end
        if ci / n_chunks > traj_frac_loss_rg
            if (ci * n_steps_chunk) % n_steps_dist == 0
                rg_count += 1
                rg_sum += radius_gyration(coords[backbone_inds], atoms[backbone_inds]) * 10
            end
        end
    end

    dists_mean = dists_sum ./ dists_count
    dists_std = sqrt_grad_safe.((dists_sq_sum ./ dists_count) .- dists_mean .^ 2)
    loss_vals_PQ_init = kl_divergence.(target_dists_mean, target_dists_std, dists_mean, dists_std)
    loss_vals_QP_init = kl_divergence.(dists_mean, dists_std, target_dists_mean, target_dists_std)
    loss_vals_PQ = log.(loss_vals_PQ_init .+ one(T)) .* loss_weight_matrix
    loss_vals_QP = log.(loss_vals_QP_init .+ one(T)) .* loss_weight_matrix
    loss_val_PQ = mean(loss_vals_PQ)
    loss_val_QP = mean(loss_vals_QP)

    sinϕs_mean = sinϕs_sum ./ dists_count
    cosϕs_mean = cosϕs_sum ./ dists_count
    sinψs_mean = sinψs_sum ./ dists_count
    cosψs_mean = cosψs_sum ./ dists_count
    sinϕs_std = sqrt_grad_safe.((sinϕs_sq_sum ./ dists_count) .- sinϕs_mean .^ 2)
    cosϕs_std = sqrt_grad_safe.((cosϕs_sq_sum ./ dists_count) .- cosϕs_mean .^ 2)
    sinψs_std = sqrt_grad_safe.((sinψs_sq_sum ./ dists_count) .- sinψs_mean .^ 2)
    cosψs_std = sqrt_grad_safe.((cosψs_sq_sum ./ dists_count) .- cosψs_mean .^ 2)
    loss_vals_sinϕ = kl_divergence.(target_sinϕs_mean, target_sinϕs_std, sinϕs_mean, sinϕs_std)
    loss_vals_cosϕ = kl_divergence.(target_cosϕs_mean, target_cosϕs_std, cosϕs_mean, cosϕs_std)
    loss_vals_sinψ = kl_divergence.(target_sinψs_mean, target_sinψs_std, sinψs_mean, sinψs_std)
    loss_vals_cosψ = kl_divergence.(target_cosψs_mean, target_cosψs_std, cosψs_mean, cosψs_std)
    loss_val_torsion = (mean(loss_vals_sinϕ) + mean(loss_vals_cosϕ) + mean(loss_vals_sinψ) +
                        mean(loss_vals_cosψ)) * loss_weight_torsion / 4

    if n_steps > n_steps_distance && !use_fd
        println("Dists mean ", dists_mean)
        println("Dists std ", dists_std)
        println("KL distance losses are ", loss_vals_PQ)
        println("KL reverse distance losses are ", loss_vals_QP)
    end
    !use_fd && println("KL distance loss is ", loss_val_PQ)
    !use_fd && println("KL reverse distance loss is ", loss_val_QP)
    !use_fd && println("KL torsion loss is ", loss_val_torsion)

    rg_mean = rg_sum / rg_count
    loss_val_rg = max(abs(rg_mean - target_rg) - rg_tolerance, zero(T)) * loss_weight_rg
    !use_fd && println("Mean Rg for recorded portion is ", rg_mean, " Å")
    !use_fd && println("Rg loss is ", loss_val_rg)

    if use_kl_rev_loss
        return loss_val_PQ + loss_val_torsion + loss_val_rg + loss_val_QP
    else
        return loss_val_PQ + loss_val_torsion + loss_val_rg
    end
end

function print_grads(io, params_dic, s_ref, n_steps)
    grads = gradient(loss, params_dic, s_ref, n_steps)[1]
    for k in keys(grads)
        println(io, "Grad  ", rpad(k, pad_length), "  ", ustrip(grads[k]))
    end
end

function select_key(key)
    key in ("inter_LJ_weight_14", "inter_CO_coulomb_const", "atom_N_mass",
            "atom_O_σ", "inter_PT_C/N/CT/C_k_1")
end

println("Primal compilation run")
loss(params_dic, s_ref, n_steps_init)

if !use_fd
    println("Gradient compilation run")
    print_grads(devnull, params_dic, s_ref, n_steps_init)
    println("Gradient main run")
    if gpu
        GC.gc(true)
        CUDA.reclaim()
        @time print_grads(stdout, params_dic, s_ref, n_steps)
        CUDA.memory_status()
        println()
    else
        @time print_grads(stdout, params_dic, s_ref, n_steps)
    end
else
    fd_grad_dict = Dict()
    @time for key in keys(params_dic)
        if select_key(key)
            grad = central_fdm(2, 1)(params_dic[key]) do val
                dic = deepcopy(params_dic)
                dic[key] = val
                loss(dic, s_ref, n_steps)
            end
            fd_grad_dict[key] = grad
            println(stdout, "Grad  ", rpad(key, pad_length), "  ", grad)
        end
    end
end
