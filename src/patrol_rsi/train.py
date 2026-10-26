"""Fixed-baseline training with validation-only checkpoint selection."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from .environment import PatrolEnv, synthetic_field
from .experiment import evaluate_case, manifest_hash, metrics_mean, room, split_manifest
from .models import ConditionalVAEUNet, PrioritizedReplay, SharedDuelingQ, ddqn_update, vae_loss


class VGGPerceptual(nn.Module):
    def __init__(self, weights: Path):
        super().__init__()
        from torchvision.models import vgg16_bn
        network = vgg16_bn(weights=None)
        network.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self.features = network.features[:23].eval()
        for param in self.features.parameters():
            param.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([.229, .224, .225])[None, :, None, None])

    def forward(self, x):
        return self.features((x-self.mean)/self.std)


def setup(seed: int):
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires allocated CUDA; refusing CPU fallback")
    torch.set_num_threads(min(4, int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))))
    torch.manual_seed(seed)
    np.random.seed(seed)
    return torch.device("cuda:0")


def train_vae(output: Path, weights: Path, iterations: int = 3000):
    device = setup(42)
    output.mkdir(parents=True, exist_ok=False)
    data = split_manifest()
    (output/"splits.json").write_text(json.dumps(data, indent=2))
    model = ConditionalVAEUNet(width=8).to(device)
    features = VGGPerceptual(weights).to(device).eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    rng = np.random.default_rng(42)
    fields = np.stack([synthetic_field((24, 24), seed) for seed in data["vae_train_field_seeds"]])
    val_fields = np.stack([synthetic_field((24, 24), seed) for seed in data["vae_validation_field_seeds"]])
    train_targets = np.stack([room(seed).targets for seed in data['train_layout_seeds']])
    val_targets = np.stack([room(2000+i%8).targets for i in range(len(val_fields))])
    # Deployment observes at most eight inspection points, often none initially.
    # Validation must reflect this sparsity instead of 12% of the whole map.
    val_mask = val_targets & (rng.random(val_fields.shape) < rng.uniform(0, 1, (len(val_fields),1,1)))
    validation_x = torch.as_tensor(np.stack([val_fields*val_mask, val_mask], axis=1), dtype=torch.float32, device=device)
    validation_y = torch.as_tensor(val_fields[:, None], dtype=torch.float32, device=device)
    best = float("inf")
    start = time.monotonic()
    with (output/"learning.jsonl").open("w") as log:
        for i in range(iterations):
            model.train()
            ids = rng.integers(0, len(fields), 16)
            truth = fields[ids]
            if i % 2:
                targets = train_targets[rng.integers(0,len(train_targets),len(ids))]
                mask = targets & (rng.random(truth.shape) < rng.uniform(0,1,(len(ids),1,1)))
            else:
                mask = rng.random(truth.shape) < rng.uniform(.005, .35)
            x = torch.as_tensor(np.stack([truth*mask, mask], axis=1), dtype=torch.float32, device=device)
            y = torch.as_tensor(truth[:, None], dtype=torch.float32, device=device)
            predicted, prior, posterior = model(x, y)
            loss, parts = vae_loss(predicted, y, prior, posterior, perceptual_network=features, perceptual_weight=.1)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite VAE training loss")
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(), 10);optimizer.step()
            if (i+1) % 100 == 0:
                model.eval()
                torch_state=torch.random.get_rng_state();cuda_state=torch.cuda.get_rng_state()
                torch.manual_seed(98765)
                with torch.no_grad():
                    estimate = model.predict(validation_x, samples=2)
                    error = float((estimate-validation_y).square().mean())
                    naive = np.where(val_mask, val_fields, .5)
                    naive_error = float(((naive-val_fields)**2).mean())
                    zero_error = float(validation_y.square().mean())
                    mean_error = float(((validation_y-float(fields.mean()))**2).mean())
                    target_error = float((estimate-validation_y)[:,0][torch.as_tensor(val_targets,device=device)].square().mean())
                torch.random.set_rng_state(torch_state);torch.cuda.set_rng_state(cuda_state)
                row = {"iteration": i+1, **parts, "validation_mse": error,
                       "validation_naive_mse": naive_error,"zero_mse":zero_error,"mean_mse":mean_error,
                       "validation_target_mse":target_error,"prediction_std":float(estimate.std()),"seconds": time.monotonic()-start}
                log.write(json.dumps(row)+"\n");log.flush();print(json.dumps(row), flush=True)
                if error < best:
                    best = error
                    torch.save({"model":model.state_dict(),"iteration":i+1,"validation_mse":error,
                                "split_hash":manifest_hash(data),"width":8},output/"best-vae.pt")
    if best>=mean_error:
        raise RuntimeError("VAE has not surpassed the constant-mean validation reference; preserve outputs and diagnose before policy training")
    summary={"iterations":iterations,"best_validation_mse":best,"seed":42,"learning_rate":1e-4,"loss_weights":{"reconstruction":1,"kl":.01,"perceptual":.1},
             "vgg_source_sha256":hashlib.sha256(weights.read_bytes()).hexdigest(),"split_hash":manifest_hash(data),
             "architecture":"groupnorm_unet_vector_latent_fusion_v3","validation":"sparse_indoor_targets_0_to_8_observations",
             "purpose":"M1_scaled_synthetic_baseline_not_paper_reproduction"}
    (output/"summary.json").write_text(json.dumps(summary,indent=2))


def train_policy(output: Path, estimator_path: Path, seed: int, episodes: int = 1200, horizon: int = 96):
    device = setup(seed)
    output.mkdir(parents=True, exist_ok=False)
    manifest = split_manifest()
    (output/"splits.json").write_text(json.dumps(manifest,indent=2))
    estimator = ConditionalVAEUNet().to(device)
    state=torch.load(estimator_path,map_location=device,weights_only=True)
    if state["split_hash"]!=manifest_hash(manifest):raise ValueError("Estimator split contract mismatch")
    estimator.load_state_dict(state["model"]);estimator.eval()
    online,target=SharedDuelingQ().to(device),SharedDuelingQ().to(device)
    target.load_state_dict(online.state_dict());target.eval()
    optimizer=torch.optim.Adam(online.parameters(),lr=3e-4)
    replay=PrioritizedReplay(capacity=20000,seed=seed)
    rng=np.random.default_rng(seed)
    best=float("inf");updates=0;total_steps=0;start=time.monotonic()
    # Geometry and neighborhood caches are public and independent of hidden fields.
    layouts={k:room(k) for k in manifest["train_layout_seeds"]}
    cached={k:{} for k in layouts}
    with (output/"learning.jsonl").open("w") as log:
        for episode in range(episodes):
            layout_seed=int(rng.choice(manifest["train_layout_seeds"]))
            env=PatrolEnv(layouts[layout_seed],seed=1000000+seed*10000+episode)
            env._neighborhoods=cached[layout_seed]
            epsilon=max(.05,1-(episode/(episodes*.7)))
            rewards=0.;last_loss=None
            with torch.no_grad():
                x=torch.as_tensor(env.estimator_input()[None],dtype=torch.float32,device=device)
                env.set_estimate(estimator.predict(x)[0,0].cpu().numpy())
            for t in range(horizon):
                obs=env.policy_observations()
                with torch.no_grad():scores=online(torch.as_tensor(obs,device=device)).cpu().numpy()
                for agent in range(len(scores)):
                    if rng.random()<epsilon:scores[agent]=rng.random(8)
                actions=env.safe_actions(scores)
                event=env.step(actions);rewards+=sum(event["rewards"])
                with torch.no_grad():
                    x=torch.as_tensor(env.estimator_input()[None],dtype=torch.float32,device=device)
                    env.set_estimate(estimator.predict(x)[0,0].cpu().numpy())
                following=env.policy_observations();legal=env.legal_masks()
                for agent,a in enumerate(actions):
                    if a>=0:replay.add(obs[agent],a,event["rewards"][agent],following[agent],t==horizon-1,legal[agent])
                total_steps+=1
                if total_steps%4==0 and len(replay.items)>=256:
                    last_loss=ddqn_update(online,target,optimizer,replay,device,batch_size=64);updates+=1
                if total_steps%500==0:target.load_state_dict(online.state_dict())
            if (episode+1)%100==0:
                torch_state=torch.random.get_rng_state();cuda_state=torch.cuda.get_rng_state()
                validation=[evaluate_case(estimator,online,case,device,horizon=horizon)[0] for case in manifest["validation"]]
                torch.random.set_rng_state(torch_state);torch.cuda.set_rng_state(cuda_state)
                average=metrics_mean(validation)
                row={"episode":episode+1,"epsilon":epsilon,"last_episode_reward":rewards,"last_loss":last_loss,
                     "validation":average,"steps":total_steps,"seconds":time.monotonic()-start}
                log.write(json.dumps(row)+"\n");log.flush();print(json.dumps(row),flush=True)
                score=average["mean_normalized_weighted_idleness"]
                if score<best:
                    best=score
                    torch.save({"model":online.state_dict(),"seed":seed,"episode":episode+1,"validation_score":score,
                                "split_hash":manifest_hash(manifest)},output/"best-policy.pt")
    # Test is read only once after checkpoint selection and never controls training.
    checkpoint=torch.load(output/"best-policy.pt",map_location=device,weights_only=True)
    online.load_state_dict(checkpoint["model"]);online.eval()
    all_rows=[]
    for method in ("learned","route_greedy","local_greedy","random"):
        rows=[]
        for index,case in enumerate(manifest["test"]):
            row,events=evaluate_case(estimator,online,case,device,method=method,horizon=horizon,record=index==0)
            row["training_seed"]=seed;rows.append(row);all_rows.append(row)
            if events:(output/f"test-example-{method}.json").write_text(json.dumps({"case":case,"events":events},indent=2))
        print(json.dumps({"method":method,"test":metrics_mean(rows)}),flush=True)
    (output/"test-cases.json").write_text(json.dumps(all_rows,indent=2))
    summary={"training_seed":seed,"episodes":episodes,"horizon":horizon,"ddqn_updates":updates,
             "selected_episode":checkpoint["episode"],"best_validation_score":best,"split_hash":manifest_hash(manifest),
             "estimator_sha256":hashlib.sha256(estimator_path.read_bytes()).hexdigest(),
             "test":{m:metrics_mean([r for r in all_rows if r["method"]==m]) for m in ("learned","route_greedy","local_greedy","random")},
             "job_id":os.environ.get("SLURM_JOB_ID"),"gpu":torch.cuda.get_device_name(),
             "rsi_enabled":False,"paper_reproduction":False,"seconds":time.monotonic()-start}
    (output/"summary.json").write_text(json.dumps(summary,indent=2))


def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest="stage",required=True)
    vae=sub.add_parser("vae");vae.add_argument("--output",type=Path,required=True);vae.add_argument("--weights",type=Path,required=True);vae.add_argument("--iterations",type=int,default=3000)
    policy=sub.add_parser("policy");policy.add_argument("--output",type=Path,required=True);policy.add_argument("--estimator",type=Path,required=True);policy.add_argument("--seed",type=int,required=True);policy.add_argument("--episodes",type=int,default=1200)
    args=parser.parse_args()
    if args.stage=="vae":train_vae(args.output,args.weights,args.iterations)
    else:train_policy(args.output,args.estimator,args.seed,args.episodes)


if __name__=="__main__":main()
