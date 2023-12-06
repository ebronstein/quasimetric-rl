#!/bin/bash

NUM_WORKERS=32
INNER_PROD_SPEC="'inner_prod(dim=2048)'"
MLP_SPEC="'mlp(dim=2048,hidden_sizes=[1024])'"
IQE_SPEC="'iqe(dim=2048,components=64)'"
L2_SPEC="'l2(dim=2048)'"

# TODO: set these.
ENV_NAME="maze2d-medium-v1"
VALUE_HEAD_SPEC=$MLP_SPEC

args=(
    env.kind=d4rl
    env.name=$ENV_NAME
    num_workers=$NUM_WORKERS
    # encoder
    agent.quasimetric_critic.model.encoder.arch="[1024,1024,1024]"
    # quasimetric model
    agent.quasimetric_critic.model.quasimetric_model.quasimetric_head_spec=$VALUE_HEAD_SPEC
    agent.quasimetric_critic.model.quasimetric_model.projector_arch="[1024,1024]"
    # dynamics
    agent.quasimetric_critic.model.latent_dynamics.arch="[1024,1024,1024]"
    agent.quasimetric_critic.losses.latent_dynamics.weight=1
    # critic
    agent.quasimetric_critic.losses.critic_optim.lr=5e-4
    agent.quasimetric_critic.losses.global_push.softplus_beta=0.01
    agent.quasimetric_critic.losses.global_push.softplus_offset=500
    # actor
    agent.actor.model.arch="[1024,1024,1024,1024]"
    agent.actor.losses.actor_optim.lr=3e-5
    agent.actor.losses.min_dist.adaptive_entropy_regularizer=False
    agent.actor.losses.min_dist.add_goal_as_future_state=False
    agent.actor.losses.behavior_cloning.weight=0.05
)

python -m offline.main "${args[@]}" "${@}"
