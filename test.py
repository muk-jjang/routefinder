import argparse
import os
import pickle
import time
import warnings

import torch

from rl4co.data.transforms import StateAugmentation
from rl4co.utils.ops import gather_by_index, unbatchify
from tqdm.auto import tqdm

from routefinder.data.utils import get_dataloader
from routefinder.envs import MTVRPEnv
from routefinder.models import RouteFinderBase, RouteFinderMoE
from routefinder.models.baselines.mtpomo import MTPOMO
from routefinder.models.baselines.mvmoe import MVMoE
from routefinder.models.decoding_strategy import get_decoding_strategy  # NEW

# Tricks for faster inference
try:
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
except AttributeError:
    pass

torch.set_float32_matmul_precision("medium")


def _compute_gap_to_bks_tensor(max_aug_reward: torch.Tensor, costs_bks: torch.Tensor) -> torch.Tensor:
    b1 = max_aug_reward.shape[0]
    b2 = costs_bks.shape[0]
    if b1 == b2:
        c = costs_bks
    elif b2 > b1:
        c = costs_bks[:b1]
    else:
        repeat_factor = (b1 + b2 - 1) // b2
        c = costs_bks.repeat(repeat_factor)[:b1]
    return 100 * (-max_aug_reward - torch.abs(c)) / torch.abs(c)


def decode_with_strategy(policy, td, env, decode_type: str, decoding_params: dict | None = None):
    decoding_params = decoding_params or {}
    # Encoder: get encoder output and initial embeddings from initial state
    hidden, init_embeds = policy.encoder(td)

    # Setup decoding strategy
    decode_strategy = get_decoding_strategy(
        decode_type,
        temperature=getattr(policy, "temperature", 1.0),
        tanh_clipping=getattr(policy, "tanh_clipping", 0.0),
        mask_logits=getattr(policy, "mask_logits", True),
        store_all_logp=False,
        **decoding_params,
    )
    # print(f"decode_with_strategy - After get_decoding_strategy td.shape: {td.shape}")

    # For SMC, pass model reference for greedy rollout
    if decode_type == "smc" and hasattr(decode_strategy, 'model'):
        decode_strategy.model = policy

    # Pre-decoding hook
    td, env, num_starts = decode_strategy.pre_decoder_hook(td, env)
    # ㅁ

    # Decoder pre-hook
    td, env, hidden = policy.decoder.pre_decoder_hook(td, env, hidden, num_starts)

    # Main decoding loop
    step = 0
    max_steps = 1_000_000
    while not td["done"].all():
        logits, mask = policy.decoder(td, hidden, num_starts)
        td = decode_strategy.step(logits, mask, td, env=env, model=policy, hidden=hidden)
        td = env.step(td)["next"]
        step += 1
        if step > max_steps:
            break

    # Post-decoding
    logprobs, actions, td, env = decode_strategy.post_decoder_hook(td, env)
    reward = env.get_reward(td, actions)

    return {"reward": reward, "actions": actions}


def test(
    policy,
    td,
    env,
    num_augment=8,
    augment_fn="dihedral8",  # or symmetric. Default is dihedral8 for reported eval
    num_starts=None,
    device="cuda",
    decode_type: str | None = None,
    decoding_params: dict | None = None,
):

    costs_bks = td.get("costs_bks", None)

    with torch.inference_mode():
        with (
            torch.amp.autocast("cuda")
            if "cuda" in str(device)
            else torch.inference_mode()
        ):  # Use mixed precision if supported
            n_start = env.get_num_starts(td) if num_starts is None else num_starts

            # Record base batch size before any augmentation or multistart/SMC expansion
            base_batch_size = td.shape[0]

            if num_augment > 1:
                td = StateAugmentation(num_augment=num_augment, augment_fn=augment_fn)(td)

            # Evaluate policy
            decoding_params = decoding_params or {}

            # Ensure correct n_start for unbatchify under SMC (n_start == n_particles)
            if decode_type == "smc":
                n_start = decoding_params.get("n_particles", n_start)

            if decode_type == "smc":
                out = decode_with_strategy(policy, td, env, decode_type, decoding_params)
            else:
                policy_kwargs = {
                    "phase": "test",
                    "num_starts": n_start,
                    "return_actions": True,
                }
                if decode_type is not None:
                    policy_kwargs["decode_type"] = decode_type
                    policy_kwargs.update(decoding_params)
                out = policy(td, env, **policy_kwargs)

            # Infer effective augmentation count from reward size: total = base * n_aug_eff * n_start
            total_batch = out["reward"].shape[0]
            denom = max(1, base_batch_size * max(1, n_start))
            n_aug_eff = max(1, total_batch // denom)

            # Unbatchify reward to [batch_size, n_aug_eff, n_start].
            reward = unbatchify(out["reward"], (n_aug_eff, n_start))

            if n_start > 1:
                # max multi-start reward
                max_reward, max_idxs = reward.max(dim=-1)
                out.update({"max_reward": max_reward})

                if out.get("actions", None) is not None:
                    # Reshape batch to [batch_size, n_aug_eff, n_start, ...]
                    actions = unbatchify(out["actions"], (n_aug_eff, n_start))
                    out.update(
                        {
                            "best_multistart_actions": gather_by_index(
                                actions, max_idxs, dim=max_idxs.dim()
                            )
                        }
                    )
                    out["actions"] = actions

            # Get augmentation score only during inference
            if n_aug_eff > 1:
                # If multistart is enabled, we use the best multistart rewards
                reward_ = max_reward if n_start > 1 else reward
                max_aug_reward, max_idxs = reward_.max(dim=1)
                out.update({"max_aug_reward": max_aug_reward})

                # If costs_bks is available, we calculate the gap to BKS
                if costs_bks is not None:
                    # note: torch.abs is here as a temporary fix, since we forgot to
                    # convert rewards to costs. Does not affect the results.
                    gap_to_bks = _compute_gap_to_bks_tensor(max_aug_reward, torch.abs(costs_bks))
                    out.update({"gap_to_bks": gap_to_bks})

                if out.get("actions", None) is not None:
                    actions_ = (
                        out["best_multistart_actions"] if n_start > 1 else out["actions"]
                    )
                    out.update({"best_aug_actions": gather_by_index(actions_, max_idxs)})
            else:
                # Ensure max_aug_reward exists even if there is only 1 augmentation
                if n_start > 1:
                    out.update({"max_aug_reward": max_reward.squeeze(1)})  # [B]
                else:
                    out.update({"max_aug_reward": reward.squeeze(1).squeeze(1)})  # [B]
                # Compute gap_to_bks also for single augmentation if available
                if costs_bks is not None:
                    gap_to_bks = _compute_gap_to_bks_tensor(out["max_aug_reward"], torch.abs(costs_bks))
                    out.update({"gap_to_bks": gap_to_bks})

            if out.get("gap_to_bks", None) is None:
                out.update({"gap_to_bks": torch.full_like(out["max_aug_reward"], 69420.0)})

            return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--problem",
        type=str,
        default="all",
        help="Problem name: cvrp, vrptw, etc. or all",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=100,
        help="Problem size: 50, 100, for automatic loading",
    )
    parser.add_argument(
        "--datasets",
        help="Filename of the dataset(s) to evaluate. Defaults to all under data/{problem}/ dir",
        default=None,
    )
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--remove-mixed-backhaul",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove mixed backhaul instances. Use --no-remove-mixed-backhaul to keep them.",
    )
    parser.add_argument(
        "--save-results",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save results to results/main/{size}/{checkpoint",
    )
    # Decoding options
    parser.add_argument(
        "--decode_type",
        type=str,
        default="greedy",
        choices=[
            "greedy",
            "sampling",
            "multistart_greedy",
            "multistart_sampling",
            "beam_search",
            "smc",
        ],
        help="Decoding strategy to use during evaluation",
    )
    parser.add_argument("--n_particles", type=int, default=64, help="SMC: number of particles")
    parser.add_argument(
        "--ess_threshold",
        type=float,
        default=0.0,
        help="SMC: resampling threshold as fraction of n_particles (0 disables)",
    )
    parser.add_argument(
        "--resample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="SMC: enable/disable resampling",
    )

    # Use load_from_checkpoint with map_location, which is handled internally by Lightning
    # Suppress FutureWarnings related to torch.load and weights_only
    warnings.filterwarnings("ignore", message=".*weights_only.*", category=FutureWarning)

    opts = parser.parse_args()

    if "cuda" in opts.device and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    if opts.datasets is not None:
        data_paths = opts.datasets.split(",")
    else:
        # list recursively all npz files in data/
        data_paths = []
        for root, _, files in os.walk("data"):
            for file in files:
                # print(file)
                if "test" not in root:
                    continue
                if file.endswith(".npz"):
                    if opts.remove_mixed_backhaul and "m" in root:
                        continue
                    # if name in 50 or 100, append
                    if str(opts.size) in file:
                        if file == "50.npz" or file == "100.npz":
                            data_paths.append(os.path.join(root, file))
        assert len(data_paths) > 0, "No datasets found. Check the data directory."
        data_paths = sorted(sorted(data_paths), key=lambda x: len(x))
        print(f"Found {len(data_paths)} datasets on the following paths: {data_paths}")

    # Load model
    print("Loading checkpoint from ", opts.checkpoint)
    if "mvmoe" in opts.checkpoint:
        BaseLitModule = MVMoE
    elif "mtpomo" in opts.checkpoint:
        BaseLitModule = MTPOMO
    elif "moe" in opts.checkpoint:
        BaseLitModule = RouteFinderMoE
    else:
        BaseLitModule = RouteFinderBase

    model = BaseLitModule.load_from_checkpoint(
        opts.checkpoint, map_location="cpu", strict=False
    )

    env = MTVRPEnv()
    policy = model.policy.to(device).eval()  # Use mixed precision if supported

    # Build decoding params from CLI
    decoding_params = {}
    if opts.decode_type == "smc":
        decoding_params.update(
            {
                "n_particles": opts.n_particles,
                "ess_threshold": opts.ess_threshold,
                "resample": opts.resample,
            }
        )

    results = {}
    for dataset in tqdm(data_paths):

        print(f"Loading {dataset}")
        td_test = env.load_data(dataset)  # this also adds the bks cost
        dataloader = get_dataloader(td_test, batch_size=opts.batch_size)

        start = time.time()
        res = []
        for batch in dataloader:
            td_test = env.reset(batch).to(device)
            # print(f"Before SMC - td batch size: {td_test.batch_size}")
            # print(f"Before SMC - td shape: {td_test.shape}")
            o = test(
                policy,
                td_test,
                env,
                device=device,
                decode_type=opts.decode_type,
                decoding_params=decoding_params,
            )
            res.append(o)
        out = {}
        out["max_aug_reward"] = torch.cat([o["max_aug_reward"] for o in res])
        out["gap_to_bks"] = torch.cat([o["gap_to_bks"] for o in res])
            
        inference_time = time.time() - start

        dataset_name = dataset.split("/")[-3].split(".")[0].upper()
        print(
            f"{dataset_name} | Cost: {-out['max_aug_reward'].mean().item():.3f} | Gap: {out['gap_to_bks'].mean().item():.3f}% | Inference time: {inference_time:.3f} s"
        )

        if results.get(dataset_name, None) is None:
            results[dataset_name] = {}
        results[dataset_name]["cost"] = -out["max_aug_reward"].mean().item()
        results[dataset_name]["gap"] = out["gap_to_bks"].mean().item()
        results[dataset_name]["inference_time"] = inference_time

    if opts.save_results:
        # Save results with checkpoint name under results/main/
        checkpoint_name = opts.checkpoint.split("/")[-1].split(".")[0]
        savedir = f"results/main/{opts.size}/"
        os.makedirs(savedir, exist_ok=True)
        pickle.dump(results, open(savedir + checkpoint_name + ".pkl", "wb"))
