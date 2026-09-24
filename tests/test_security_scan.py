import pytest

from scripts.security_scan import check_audit_report, TRIAGED_ADVISORIES


def report(package="chromadb", version="1.5.9", advisory="PYSEC-2026-311"):
    return {"dependencies": [{"name": package, "version": version, "vulns": [{"id": advisory}]}]}


def test_vex_accepts_only_documented_versions_and_reports_remaining_risk():
    rows = [{"name": name, "version": version, "vulns": [{"id": key} for key in ids]}
            for name, (version, ids) in TRIAGED_ADVISORIES.items()]
    result = check_audit_report({"dependencies": rows})
    assert result["advisory_entries"] == result["unique_advisories"] == 9
    assert result["untriaged"] == []
    assert result["vex"] == "docs/security-hardening.md#dependency-vex"


@pytest.mark.parametrize("package,version,advisory", [
    ("chromadb", "1.5.9", "PYSEC-2099-1"),
    ("pillow", "12.3.0", "PYSEC-2026-311"),
    ("transformers", "4.57.6", "PYSEC-2099-2"),
])
def test_vex_rejects_new_or_wrongly_scoped_advisories(package, version, advisory):
    assert check_audit_report(report(package, version, advisory))["untriaged"]


@pytest.mark.parametrize("value", [{}, {"dependencies": []},
    {"dependencies": [{"name": "torch", "skip_reason": "not audited"}]},
    {"dependencies": [{"name": "chromadb", "version": "1.5.9"}]},
])
def test_vex_rejects_missing_or_incomplete_audit(value):
    with pytest.raises(ValueError):
        check_audit_report(value)


def test_vex_retains_duplicate_entry_count_without_duplicate_unique_risk():
    value = report()
    value["dependencies"][0]["vulns"] *= 2
    result = check_audit_report(value)
    assert result["advisory_entries"] == 2 and result["unique_advisories"] == 1


@pytest.mark.parametrize('vulns', [[], [{'id': 'PYSEC-2026-311'}]])
def test_vex_rejects_changed_version_even_without_findings(vulns):
    value = report(version='1.5.10')
    value['dependencies'][0]['vulns'] = vulns
    with pytest.raises(ValueError):
        check_audit_report(value)


@pytest.mark.parametrize('reason', ['', None, 'not audited'])
def test_vex_rejects_any_skip_reason(reason):
    value = report()
    value['dependencies'][0]['skip_reason'] = reason
    with pytest.raises(ValueError):
        check_audit_report(value)


@pytest.mark.parametrize('name', [
    'docs/example.json', 'requirements.txt', 'requirements-dev.txt',
    'requirements-security.txt', 'pytest.ini', 'scripts/check.ps1', 'scripts/check.sh',
    'docs/example.toml', 'docs/example.cfg', 'docs/example.txt',
    'main.py', 'README.md', 'todolist.md', '.env.example',
    '.github/check.yml', '.github/check.yaml',
])
def test_secrets_in_source_formats_and_root_files(tmp_path, name):
    from scripts.security_scan import scan
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    # Construct only at runtime; never store a credential-shaped literal in source.
    synthetic = 'AIza' + 'x' * 35
    path.write_text('{"key": "' + synthetic + '"}', encoding='utf-8')
    assert scan(tmp_path) == [{'path': name, 'category': 'google_key'}]


def test_excluded_files_are_never_read(tmp_path, monkeypatch):
    from pathlib import Path
    from scripts.security_scan import scan
    forbidden = ['.env', '.venv/a.py', '.git/a.txt', 'model/a.json',
        'dataset/a.json', 'ChromaDB/a.json', 'uploads/a.txt', 'logs/a.txt',
        'app/.env', 'docs/models/a.json', 'scripts/dataset/a.txt',
        'docs/chroma_db_wayang/a.json', 'docs/wayang_db.json', 'app/weights.bin']
    forbidden += ['docs/' + name for name in (
        '.venv/a.py', '.git/a.txt', 'model/a.json', 'dataset/a.json',
        'ChromaDB/a.json', 'upload/a.txt', 'log/a.txt', '.env',
        'weights.pt', 'weights.safetensors', 'archive.zip')]
    for name in forbidden:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('excluded', encoding='utf-8')
    original = Path.read_text
    def guarded(path, *args, **kwargs):
        assert path.relative_to(tmp_path).as_posix() not in forbidden
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', guarded)
    assert scan(tmp_path) == []


@pytest.mark.parametrize('value,expected', [
    (report(), 0),
    (report(advisory='PYSEC-2099-1'), 1),
    (report(version='1.5.10'), 2),
    ({'dependencies': []}, 2),
    ({'dependencies': [{'name': 'torch', 'skip_reason': ''}]}, 2),
    (None, 2),
])
def test_audit_cli_exit_status(tmp_path, value, expected):
    import json
    from pathlib import Path
    import subprocess
    import sys

    audit = tmp_path / 'audit.json'
    audit.write_text(json.dumps(value), encoding='utf-8')
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'security_scan.py'
    result = subprocess.run(
        [sys.executable, str(script), '--audit-json', str(audit)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == expected, result.stdout + result.stderr


def test_windows_junction_is_not_read(tmp_path, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace
    from scripts.security_scan import scan

    junction = tmp_path / 'docs'
    junction.mkdir()
    (junction / 'hidden.txt').write_text('excluded', encoding='utf-8')
    original = Path.lstat
    monkeypatch.setattr(Path, 'lstat', lambda p: (
        SimpleNamespace(st_file_attributes=0x400) if p == junction else original(p)))
    monkeypatch.setattr(Path, 'is_symlink', lambda p: False)
    def forbidden_read(*args, **kwargs):
        pytest.fail('Scanner opened a junction target')
    monkeypatch.setattr(Path, 'read_text', forbidden_read)
    assert scan(tmp_path) == []


@pytest.mark.parametrize('name', ['docs/linked.json', 'docs/linked/nested.txt', 'requirements.txt'])
def test_symlink_files_and_directories_are_not_read(tmp_path, monkeypatch, name):
    from pathlib import Path
    from scripts.security_scan import scan
    target = tmp_path / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('excluded', encoding='utf-8')
    link = target.parent if name.endswith('nested.txt') else target
    # Emulate link metadata on Windows without requiring symlink privileges.
    original = Path.is_symlink
    monkeypatch.setattr(Path, 'is_symlink', lambda p: p == link or original(p))
    def forbidden_read(*args, **kwargs):
        pytest.fail('Scanner opened a symlink target')
    monkeypatch.setattr(Path, 'read_text', forbidden_read)
    assert scan(tmp_path) == []
