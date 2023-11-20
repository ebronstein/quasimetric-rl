#!/bin/bash

# default args are for the online GCRL setting, so we need to change some of them
# for offline d4rl.

args=(
    env.kind=d4rl
    num_workers=12
    # encoder
    agent.quasimetric_critic.model.encoder.arch="[512,512]"
    # quasimetric model
    agent.quasimetric_critic.model.quasimetric_model.projector_arch="[512]"
    # dynamics
    agent.quasimetric_critic.model.latent_dynamics.arch="[512,512]"
    agent.quasimetric_critic.losses.latent_dynamics.weight=1
    # critic
    agent.quasimetric_critic.losses.critic_optim.lr=5e-4
    agent.quasimetric_critic.losses.global_push.softplus_beta=0.01
    agent.quasimetric_critic.losses.global_push.softplus_offset=500
    # actor
    agent.actor=null
)

exec python -m offline.check_value_func "${args[@]}" "${@}"
