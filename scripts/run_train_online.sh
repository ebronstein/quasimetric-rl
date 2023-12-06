#!/bin/bash

INNER_PROD_SPEC="'inner_prod(dim=2048)'"
MLP_SPEC="'mlp(dim=2048,hidden_sizes=[1024])'"
IQE_SPEC="'iqe(dim=2048,components=64)'"
L2_SPEC="'l2(dim=2048)'"

FETCHREACH="FetchReach"
FETCHREACHIMAGE="FetchReachImage"
FETCHPUSH="FetchPush"
FETCHPUSHIMAGE="FetchPushImage"
FETCHSLIDE="FetchSlide"

# TODO: set these.
ENV_NAME=$FETCHSLIDE
VALUE_HEAD_SPEC=$MLP_SPEC

args=(
    env.kind=gcrl
    env.name=$ENV_NAME
    # quasimetric model
    agent.quasimetric_critic.model.quasimetric_model.quasimetric_head_spec=$VALUE_HEAD_SPEC
)

python -m online.main "${args[@]}" "${@}"
