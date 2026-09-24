"""Source-only secret scan. Does not open .env, artifacts, logs or Git history."""
import json
from pathlib import Path
import re
import argparse
import os

ROOT = Path(__file__).resolve().parents[1]
SOURCE_EXTENSIONS = {'.py', '.md', '.yml', '.yaml', '.json', '.txt', '.toml',
                     '.ini', '.cfg', '.ps1', '.sh'}
EXCLUDED_NAMES = {'.venv', '.git', '__pycache__', 'model', 'models', 'dataset',
                  'datasets', 'data', 'chromadb', 'chroma', 'chroma_db',
                  'chroma_db_wayang', 'upload', 'uploads', 'log', 'logs',
                  'artifacts', 'artefacts', 'security-scan-results', 'wayang_db.json'}


def is_link(path):
    """Reject symlinks and Windows reparse points (including junctions)."""
    return (path.is_symlink() or os.path.islink(path)
            or bool(getattr(path.lstat(), 'st_file_attributes', 0) & 0x400))


def excluded(path):
    name = path.name.lower()
    return (name in EXCLUDED_NAMES or name.startswith(('.env', 'model_', 'chroma_'))
            or is_link(path))


PATTERNS = {
    "google_key": re.compile(r"AIza[\w-]{35}"),
    "service_key": re.compile(r"sk-[\w-]{20,}|gh[pousr]_[\w]{20,}|AKIA[A-Z0-9]{16}"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "credential_assignment": re.compile(r'''(?im)^[ \t]*(?:GEMINI_API_KEY|WAYANG_SERVICE_TOKEN)[ \t]*=[ \t]*["']?[^\s"'#][^\r\n]*'''),
}

# Version-specific acceptance; rationale/conditions in security-hardening.md#dependency-vex.
# Never pass these as global --ignore-vuln flags: preserve the full scanner report.
TRIAGED_ADVISORIES = {
    "chromadb": ("1.5.9", frozenset({"PYSEC-2026-311", "PYSEC-2026-3813",
                                   "PYSEC-2026-3814", "PYSEC-2026-3815"})),
    "transformers": ("4.57.6", frozenset({"PYSEC-2025-217", "PYSEC-2026-2288",
                                       "PYSEC-2026-2289", "PYSEC-2026-2290", "PYSEC-2026-3929"})),
}


def check_audit_report(report):
    """Fail closed on incomplete reports, new IDs, packages or unreviewed versions."""
    dependencies = report.get("dependencies") if isinstance(report, dict) else None
    if not isinstance(dependencies, list) or not dependencies:
        raise ValueError("Missing dependency audit results")
    entries, unique, untriaged = 0, set(), []
    for dependency in dependencies:
        if (not isinstance(dependency, dict) or "skip_reason" in dependency
                or not isinstance(dependency.get("name"), str)
                or not isinstance(dependency.get("version"), str)
                or not isinstance(dependency.get("vulns"), list)):
            raise ValueError("Incomplete dependency audit results")
        package, version = dependency["name"].lower().replace("_", "-"), dependency["version"]
        reviewed_version, reviewed_ids = TRIAGED_ADVISORIES.get(package, (None, frozenset()))
        if reviewed_version is not None and version != reviewed_version:
            raise ValueError("Unreviewed version of VEX package")
        for vulnerability in dependency["vulns"]:
            if not isinstance(vulnerability, dict) or not isinstance(vulnerability.get("id"), str):
                raise ValueError("Invalid advisory record")
            advisory = vulnerability["id"]
            entries += 1
            unique.add((package, advisory))
            if version != reviewed_version or advisory not in reviewed_ids:
                untriaged.append({"package": package, "version": version, "id": advisory})
    return {"advisory_entries": entries, "unique_advisories": len(unique),
            "untriaged": untriaged, "vex": "docs/security-hardening.md#dependency-vex"}


def source_paths(root=ROOT):
    for folder in ("app", "tests", "scripts", "docs", ".github"):
        base = root / folder
        if not base.exists() or excluded(base):
            continue
        for directory, dirs, files in os.walk(base, followlinks=False):
            dirs[:] = [name for name in dirs if not excluded(Path(directory) / name)]
            for name in files:
                path = Path(directory) / name
                if not excluded(path) and path.is_file() and path.suffix.lower() in SOURCE_EXTENSIONS:
                    yield path
    for name in ("main.py", "README.md", "todolist.md", ".env.example",
                 "requirements.txt", "requirements-dev.txt", "requirements-security.txt", "pytest.ini"):
        path = root / name
        if path.exists() and not is_link(path) and path.is_file():
            yield path


def scan(root=ROOT):
    findings = []
    for path in source_paths(root):
        text = path.read_text(encoding="utf-8-sig")
        for category, pattern in PATTERNS.items():
            if category == "credential_assignment" and path.name != ".env.example":
                continue  # Python os.getenv/field assignments are not literal secrets.
            if pattern.search(text):
                findings.append({"path": path.relative_to(root).as_posix(), "category": category})
    return findings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-json", type=Path, help="Validate complete pip-audit JSON against version-specific VEX")
    args = parser.parse_args()
    findings = scan()
    print(json.dumps({"scope": "source allowlist; excludes .env and artifacts", "findings": findings}, indent=2))
    failed = bool(findings)
    if args.audit_json:
        try:
            if (args.audit_json.suffix != ".json"
                    or any(p.name.lower().startswith('.env') or is_link(p)
                           for p in (args.audit_json, *args.audit_json.parents))):
                raise ValueError("Audit input must be a JSON report")
            result = check_audit_report(json.loads(args.audit_json.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            print("Dependency audit unavailable or incomplete; validation failed.")
            raise SystemExit(2) from None
        print(json.dumps(result, indent=2))
        failed = failed or bool(result["untriaged"])
    raise SystemExit(failed)
