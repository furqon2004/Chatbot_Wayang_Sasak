#!/usr/bin/env bash
# Called over SSH with a verified GHCR digest; application secrets stay on the VPS.
set -euo pipefail

if [[ $# != 2 ]]; then
    printf 'Usage: deploy_vps.sh ABSOLUTE_DEPLOY_PATH GHCR_IMAGE_DIGEST\n' >&2
    exit 2
fi
target=$1
image=$2
if [[ ! "$target" =~ ^/[a-zA-Z0-9_./-]+$ || "$target" == / ||
      "$target" == *"//"* ||
      "/${target#/}/" == *"/../"* || "/${target#/}/" == *"/./"* ||
      ! "$image" =~ ^ghcr\.io/[a-z0-9_./-]+@sha256:[a-f0-9]{64}$ ]]; then
    printf 'Invalid deployment path or image digest\n' >&2
    exit 2
fi

cd "$target"
target=$(pwd -P)
if [[ "$target" == / ]]; then
    printf 'Deployment path must be an application directory\n' >&2
    exit 2
fi
test -r .env.docker
test -d artifacts
test -r compose.pending.yaml
compose=(docker compose --project-directory "$target" -f "$target/compose.yaml")
if [[ -f .env.image ]]; then
    compose+=(--env-file "$target/.env.image")
fi
WAYANG_IMAGE="$image" docker compose --project-directory "$target" \
    -f "$target/compose.pending.yaml" config --quiet

previous_image=
had_compose=false
if [[ -f compose.yaml ]]; then
    had_compose=true
    previous_container=$("${compose[@]}" ps -a -q app)
    if [[ -n "$previous_container" ]]; then
        previous_image=$(docker inspect --format '{{.Config.Image}}' "$previous_container")
    fi
fi

# A pull failure leaves the current image/configuration running.
docker pull "$image"
if [[ "$had_compose" == true ]]; then
    cp -p compose.yaml .compose.previous.yaml
fi
cp compose.pending.yaml compose.yaml
if WAYANG_IMAGE="$image" "${compose[@]}" up -d --no-build --pull never --wait --wait-timeout 240 app; then
    umask 077
    printf 'WAYANG_IMAGE=%s\n' "$image" > .env.image
    printf 'Deployment healthy: %s\n' "$image"
    exit 0
fi

printf 'Deployment failed health validation; restoring the previous release\n' >&2
if [[ "$had_compose" == true ]]; then
    cp -p .compose.previous.yaml compose.yaml
fi
if [[ -n "$previous_image" ]]; then
    if WAYANG_IMAGE="$previous_image" "${compose[@]}" up -d --no-build --pull never --wait --wait-timeout 240 app; then
        umask 077
        printf 'WAYANG_IMAGE=%s\n' "$previous_image" > .env.image
        printf 'Previous release restored\n' >&2
    else
        printf 'Rollback also failed; inspect the VPS container and its logs\n' >&2
    fi
else
    "${compose[@]}" stop app
    printf 'No previous release was available for rollback\n' >&2
fi
exit 1
