from pathlib import Path
import subprocess

import pytest


OLD_IMAGE = "ghcr.io/example/wayang@sha256:" + "a" * 64
NEW_IMAGE = "ghcr.io/example/wayang@sha256:" + "b" * 64
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "deploy_vps.sh"
BASH = next((path for path in (Path('/bin/bash'), Path('C:/Program Files/Git/bin/bash.exe'))
             if path.is_file()), None)
pytestmark = pytest.mark.skipif(BASH is None, reason="Bash is required for deployment-script tests")

HARNESS = r'''
if command -v cygpath >/dev/null 2>&1; then
    DEPLOY_DIR=$(cygpath -u "$DEPLOY_DIR")
    DEPLOY_SCRIPT=$(cygpath -u "$DEPLOY_SCRIPT")
fi
docker() {
    printf '%s | %s\n' "${WAYANG_IMAGE:-unset}" "$*" >> "$TRACE"
    if [[ "$1" == pull && "$FAIL_PULL" == 1 ]]; then return 1; fi
    if [[ "$1" == inspect ]]; then printf '%s\n' "$OLD_IMAGE"; fi
    if [[ " $* " == *" ps -a -q app "* && "$HAVE_CONTAINER" == 1 ]]; then
        printf '%s\n' old-container
    fi
    if [[ " $* " == *" up "* && "${WAYANG_IMAGE:-}" == "$NEW_IMAGE" && "$FAIL_HEALTH" == 1 ]]; then
        return 1
    fi
    return 0
}
export -f docker
bash "$DEPLOY_SCRIPT" "$DEPLOY_DIR" "$NEW_IMAGE"
'''


def deploy(tmp_path, *, fail_pull=False, fail_health=False, previous=True, image=NEW_IMAGE):
    target = tmp_path / 'server'
    target.mkdir()
    (target / 'artifacts').mkdir()
    (target / '.env.docker').write_text('# application secrets stay on server\n')
    (target / 'compose.pending.yaml').write_text('new-compose\n')
    if previous:
        (target / 'compose.yaml').write_text('old-compose\n')
        (target / '.env.image').write_text('WAYANG_IMAGE=' + OLD_IMAGE + '\n')
    trace = tmp_path / 'docker-trace.txt'
    result = subprocess.run(
        [str(BASH), '--noprofile', '--norc', '-c', HARNESS],
        env={'PATH': '/usr/bin:/bin', 'DEPLOY_SCRIPT': str(SCRIPT), 'DEPLOY_DIR': str(target),
             'TRACE': str(trace), 'OLD_IMAGE': OLD_IMAGE, 'NEW_IMAGE': image,
             'FAIL_PULL': str(int(fail_pull)), 'FAIL_HEALTH': str(int(fail_health)),
             'HAVE_CONTAINER': str(int(previous))},
        capture_output=True, text=True, timeout=20,
    )
    return result, target, trace


def test_success_persists_only_the_healthy_image(tmp_path):
    result, target, _ = deploy(tmp_path)
    assert result.returncode == 0, result.stderr
    assert (target / 'compose.yaml').read_text() == 'new-compose\n'
    assert (target / '.env.image').read_text() == 'WAYANG_IMAGE=' + NEW_IMAGE + '\n'
    assert (target / '.env.docker').read_text() == '# application secrets stay on server\n'


def test_failed_pull_keeps_the_running_release_configuration(tmp_path):
    result, target, _ = deploy(tmp_path, fail_pull=True)
    assert result.returncode == 1, result.stderr
    assert (target / 'compose.yaml').read_text() == 'old-compose\n'
    assert (target / '.env.image').read_text() == 'WAYANG_IMAGE=' + OLD_IMAGE + '\n'


def test_failed_health_restores_previous_compose_and_image(tmp_path):
    result, target, trace = deploy(tmp_path, fail_health=True)
    assert result.returncode == 1, result.stderr
    assert (target / 'compose.yaml').read_text() == 'old-compose\n'
    assert (target / '.env.image').read_text() == 'WAYANG_IMAGE=' + OLD_IMAGE + '\n'
    assert any(line.startswith(OLD_IMAGE + ' | ') and ' up ' in line
               for line in trace.read_text().splitlines())


def test_failed_first_release_does_not_record_it_as_deployed(tmp_path):
    result, target, trace = deploy(tmp_path, fail_health=True, previous=False)
    assert result.returncode == 1, result.stderr
    assert not (target / '.env.image').exists()
    assert 'stop app' in trace.read_text()


@pytest.mark.parametrize('image', ['wayang:latest', 'ghcr.io/a/b;touch /tmp/unsafe'])
def test_invalid_image_is_rejected_before_docker_runs(tmp_path, image):
    result, target, trace = deploy(tmp_path, image=image)
    assert result.returncode == 2, result.stderr
    assert not trace.exists()
    assert (target / 'compose.yaml').read_text() == 'old-compose\n'
