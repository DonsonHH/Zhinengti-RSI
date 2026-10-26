"""Bounded GPU engineering run, explicitly not a reproduced paper result."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform

import numpy as np
import torch

from .environment import PatrolEnv, greedy_scores, make_scenario, synthetic_field
from .models import ConditionalVAEUNet, PrioritizedReplay, SharedDuelingQ, ddqn_update, vae_loss
from .versions import AgentVersion, record_proposal


def tensor(array, device):
    return torch.as_tensor(np.asarray(array), dtype=torch.float32, device=device)


def run(output: Path, device_name: str, steps: int = 48, seed: int = 7) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested; refusing silent CPU fallback")
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    torch.set_num_threads(min(4, int(os.environ.get("SLURM_CPUS_PER_TASK", "2"))))
    estimator = ConditionalVAEUNet().to(device)
    online = SharedDuelingQ().to(device)
    target = SharedDuelingQ().to(device)
    target.load_state_dict(online.state_dict())
    target.eval()
    estimator_before = next(estimator.parameters()).detach().clone()
    policy_before = next(online.parameters()).detach().clone()
    estimate_optimizer = torch.optim.Adam(estimator.parameters(), lr=1e-3)
    q_optimizer = torch.optim.Adam(online.parameters(), lr=1e-3)

    losses = []
    # Independent generated fields, distinct from rollout seeds; smoke training only.
    for iteration in range(12):
        fields = np.stack([synthetic_field((24, 24), 10000 + iteration * 4 + i) for i in range(4)])
        mask = rng.random(fields.shape) < 0.15
        inputs = tensor(np.stack([fields * mask, mask], axis=1), device)
        truth = tensor(fields[:, None], device)
        predicted, prior, posterior = estimator(inputs, truth)
        loss, details = vae_loss(predicted, truth, prior, posterior)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite VAE loss")
        estimate_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(estimator.parameters(), 10)
        estimate_optimizer.step()
        losses.append(details)
    estimator.eval()
    replay = PrioritizedReplay(capacity=1024, seed=seed)
    version = AgentVersion()
    results = {}
    q_losses = []
    estimator_calls = 0
    for scenario_name in ("paper_grid", "supercomputer"):
        env = PatrolEnv(make_scenario(scenario_name), seed=seed)
        evaluations = []
        with torch.no_grad():
            prediction = estimator.predict(tensor(env.estimator_input()[None], device), version.estimator_samples)[0, 0].cpu().numpy()
        env.set_estimate(prediction)
        estimator_calls += 1
        for step in range(steps):
            observed = env.policy_observations()
            with torch.no_grad():
                scores = online(tensor(observed, device)).cpu().numpy()
            if rng.random() < 0.25:
                scores = rng.random(scores.shape)
            actions = env.safe_actions(scores, version.coordination_order)
            event = env.step(actions)
            if (step + 1) % version.estimator_refresh_interval == 0:
                with torch.no_grad():
                    prediction = estimator.predict(tensor(env.estimator_input()[None], device),
                                                   version.estimator_samples)[0, 0].cpu().numpy()
                env.set_estimate(prediction)
                estimator_calls += 1
            following = env.policy_observations()
            legal = env.legal_masks()
            for i, a in enumerate(actions):
                if a >= 0:
                    replay.add(observed[i], a, event["rewards"][i], following[i], step == steps - 1, legal[i])
            if len(replay.items) >= 16 and step % 4 == 0:
                q_losses.append(ddqn_update(online, target, q_optimizer, replay, device))
            if step % 12 == 0:
                target.load_state_dict(online.state_dict())
            evaluations.append(env.evaluation())
        metrics = env.evaluation()
        metrics["time_mean_weighted_idleness_sum"] = float(np.mean([v["terminal_weighted_idleness_sum"] for v in evaluations]))
        results[scenario_name] = metrics
        with (output / f"{scenario_name}-events.jsonl").open("w", encoding="utf-8") as handle:
            for event in env.events:
                handle.write(json.dumps(event) + "\n")

    if torch.equal(estimator_before, next(estimator.parameters()).detach()):
        raise AssertionError("VAE parameters did not update")
    if torch.equal(policy_before, next(online.parameters()).detach()):
        raise AssertionError("DDQN parameters did not update")
    if not q_losses:
        raise AssertionError("DDQN update path was not executed")
    checkpoint = output / "engineering-checkpoint.pt"
    torch.save({"vae": estimator.state_dict(), "ddqn": online.state_dict(), "seed": seed,
                "status": "short_smoke_training_not_converged"}, checkpoint)
    record_proposal(output / "proposal-fixtures", version, {"coordination_order": "cyclic"},
                    {"split": "development", "status": "interface_fixture_only"},
                    "Validate recording only; this candidate was not evaluated or accepted", False, "human_fixture")
    summary = {"purpose": "engineering_smoke_not_paper_reproduction", "seed": seed,
               "split": "development", "device": str(device),
               "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
               "gpu_memory_bytes": torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else None,
               "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
               "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "neural_updates": {"vae": len(losses), "ddqn": len(q_losses)},
               "estimator_calls": estimator_calls, "estimator_prior_samples": estimator_calls * version.estimator_samples,
               "vae_losses": losses, "ddqn_losses": q_losses, "scenario_metrics": results,
               "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
               "agent_version": asdict(version), "llm_improvement_executed": False,
               "changes_from_paper": ["small_network_width_and_depth_in_latent_encoder", "perceptual_loss_disabled_for_offline_smoke",
                                      "synthetic_indoor_data", "conservative_no_follow_collision_mask",
                                      "short_training_not_converged", "KL_posterior_to_prior_as_candidate_code"],
               "claims": ["VAE_and_shared_DDQN_forward_backward_and_parameter_updates_executed",
                          "not_an_algorithm_performance_comparison", "not_a_paper_table_reproduction", "not_RSI_effectiveness_evidence"]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "neural_updates": summary["neural_updates"],
                      "device": summary["cuda_device"], "purpose": summary["purpose"]}))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.steps < 16:
        parser.error("Use at least 16 steps to exercise replay updates")
    run(args.output, args.device, args.steps, args.seed)


if __name__ == "__main__":
    main()
