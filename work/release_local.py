"""Install or roll back this repository's two Compose sidecars on the production host."""
import argparse
import ast
from contextlib import closing
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
from tempfile import NamedTemporaryFile
from time import sleep


def run(*args, cwd=None):
    return subprocess.check_output(args, cwd=cwd)


def install_bytes(path, data, mode=0o644):
    with NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp.write(data)
        os.fchmod(tmp.fileno(), mode)
        tmp.flush()
        os.fsync(tmp.fileno())
    os.replace(tmp.name, path)


def local_mapping(directory):
    path = directory / "rooms.json"
    rooms = json.loads(path.read_text())
    if all("temperature" in c and "witness_trial_id" in c for c in rooms.values()):
        return path.read_bytes()
    # Previous installation put only these bindings in the Python source.
    # Resolve constant names and literals without executing production code.
    constants = {}
    for node in ast.parse((directory / "pilot_gate.py").read_text()).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        if name == "LIVING_REFERENCE":
            constants[name] = ast.literal_eval(node.value)
        if name in {"TEMPERATURE", "WITNESSES"} and isinstance(node.value, ast.Dict):
            constants[name] = {ast.literal_eval(key): constants[val.id] if isinstance(val, ast.Name)
                               else ast.literal_eval(val) for key, val in zip(node.value.keys, node.value.values)}
    for room, c in rooms.items():
        c.setdefault("temperature", constants["TEMPERATURE"][room])
        c.setdefault("witness_trial_id", constants["WITNESSES"][room])
    return (json.dumps(rooms, ensure_ascii=False, indent=2)+"\n").encode()


def pending(root):
    pilot = root / "pilot-data"
    states = (p for p in pilot.glob("continuous-*.json") if p.stem.removeprefix("continuous-").isdigit())
    return [(a["room"], a["sent_at"], a["status"]) for file in states
            for a in json.loads(file.read_text())["commands"] if a["status"] in {"prepared", "accepted"}]


def sidecar(root, which, action):
    compose = root / ("compose." + which + ".yaml")
    flags = ("-t", "60") if action=="stop" else ()
    run("docker", "compose", "-f", str(compose), action, *flags, cwd=root)


def quiesce(root):
    for _ in range(3):
        if pending(root):
            sleep(20)
            continue
        sidecar(root, "pilot", "stop")
        if not pending(root):
            try:
                sidecar(root, "shadow", "stop")
            except BaseException:
                sidecar(root, "pilot", "start")
                raise
            return
        sidecar(root, "pilot", "start")
        sleep(20)
    raise RuntimeError("Unsettled instruction journal, left previous sidecars active")


def resume(root):
    sidecar(root, "shadow", "start")
    sidecar(root, "pilot", "start")


def shadow_compose_storage(contents):
    text = contents.decode()
    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.startswith("    tmpfs:"):
            if "size=32m" in line:
                return contents
            lines[index] = '    tmpfs: ["/tmp:size=32m,mode=1777"]\n'
            return "".join(lines).encode()
    index = lines.index("    read_only: true\n")
    lines.insert(index+1, '    tmpfs: ["/tmp:size=32m,mode=1777"]\n')
    return "".join(lines).encode()


def replace_shadow(root):
    run("docker", "compose", "-f", str(root/"compose.shadow.yaml"), "up", "-d", "--no-deps", "--force-recreate",
        "room-climate-shadow", cwd=root)


def backup(root, destination, names):
    destination.mkdir(mode=0o700, parents=True)
    path = destination / "runtime"
    path.mkdir(mode=0o700)
    owned = {*names, "rooms.json", ".release", "topology.json"}
    (destination/"managed-files.json").write_text(json.dumps(sorted(owned)))
    for name in owned:
        source = root / "shadow" / name
        if source.exists():
            shutil.copy2(source, path / name)
    compose = root/"compose.shadow.yaml"
    if compose.is_file():
        shutil.copy2(compose,destination/compose.name)
    journals = destination / "journal-evidence"
    journals.mkdir(mode=0o700)
    for source in (root / "pilot-data").glob("*.json"):
        shutil.copy2(source, journals / source.name)
    db = root / "shadow-data/shadow.sqlite3"
    with closing(sqlite3.connect(db)) as src, closing(
            sqlite3.connect(destination / "shadow-evidence.sqlite3")) as dest:
        src.execute("PRAGMA query_only=ON")
        src.backup(dest)


def restore_code(directory, recorded):
    for source in recorded.iterdir():
        install_bytes(directory / source.name, source.read_bytes(), 0o600 if source.name=="rooms.json" else 0o644)
    managed = recorded.parent/"managed-files.json"
    if managed.exists():
        for name in json.loads(managed.read_bytes()):
            if Path(name).name == name and not (recorded/name).exists():
                (directory/name).unlink(missing_ok=True)
    if not (recorded/".release").exists():
        (directory/".release").unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production", type=Path, required=True, help="directory containing compose.pilot.yaml and shadow/")
    parser.add_argument("--topology", type=Path, help="initial site observation/zone configuration, kept private")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--commit")
    group.add_argument("--rollback", type=Path, help="backup directory printed by the preceding deployment")
    args = parser.parse_args()
    root = args.production.resolve()
    directory = root/"shadow"
    if not (root/"pilot-data").is_dir() or not (directory/"rooms.json").is_file():
        raise ValueError("Expected existing thermostat installation")
    if args.rollback:
        recorded = args.rollback.resolve()/"runtime"
        if not (recorded/"rooms.json").is_file() or not (recorded/"continuous_pilot.py").is_file():
            raise FileNotFoundError("Missing rollback code and configuration")
        quiesce(root)
        try:
            restore_code(directory, recorded)
            compose = recorded.parent/"compose.shadow.yaml"
            if compose.exists() and compose.read_bytes() != (root/compose.name).read_bytes():
                install_bytes(root/compose.name, compose.read_bytes())
                replace_shadow(root)
        finally:
            resume(root)
        print("Rollback complete. Active journals, receipts and database were retained.")
        return

    repo = Path(__file__).resolve().parents[1]
    commit = args.commit
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Use a full tested commit hash")
    remote = run("git", "-c", "credential.helper=", "-c", "credential.helper=!gh auth git-credential",
                 "ls-remote", "--exit-code", "origin", "refs/heads/main", cwd=repo).decode().split()[0]
    if remote != commit:
        raise ValueError("Release does not match remote main")
    prefix = "home-assistant/shadow/"
    files = run("git", "ls-tree", "--name-only", "-r", commit, prefix, cwd=repo).decode().splitlines()
    bundle = {Path(file).name: run("git", "show", f"{commit}:{file}", cwd=repo) for file in files
              if file.endswith(".py") or file==prefix+"house_policy.json"}
    for name, data in bundle.items():
        if name.endswith(".py"):
            compile(data, name, "exec")
    for which in ("shadow","pilot"):
        run("docker", "compose", "-f", str(root/f"compose.{which}.yaml"), "config", "--quiet", cwd=root)
    local_config = local_mapping(directory)
    topology_path = args.topology or directory/"topology.json"
    topology = topology_path.read_bytes() if topology_path.is_file() else None
    if "zone_control.py" in bundle:
        if topology is None:
            raise ValueError("Pass the site topology JSON for a unified-controller deployment")
        import sys
        sys.path.insert(0, str(repo/"home-assistant/shadow"))
        from topology import configure
        configure(json.loads(topology), json.loads(local_config))
    compose = root/"compose.shadow.yaml"
    previous_compose = compose.read_bytes()
    updated_compose = shadow_compose_storage(previous_compose) if topology is not None else previous_compose
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = root / "release-backups" / f"{stamp}-{commit[:12]}"
    quiesce(root)
    installed = False
    try:
        backup(root, destination, bundle)
        installed = True
        for name, data in bundle.items():
            install_bytes(directory/name, data)
        install_bytes(directory/"rooms.json", local_config, 0o600)
        if topology is not None:
            install_bytes(directory/"topology.json", topology, 0o600)
        install_bytes(directory/".release", (commit+"\n").encode())
        assert all((directory/name).read_bytes()==data for name,data in bundle.items())
        if updated_compose != previous_compose:
            install_bytes(compose, updated_compose)
            run("docker", "compose", "-f", str(compose), "config", "--quiet", cwd=root)
            replace_shadow(root)
        resume(root)
    except BaseException:
        if installed:
            # Restore only after the pilot is quiescent. Never replace writable
            # journals with the pre-deployment archive.
            quiesce(root)
            restore_code(directory, destination/"runtime")
            if compose.read_bytes() != previous_compose:
                install_bytes(compose, previous_compose)
                replace_shadow(root)
        resume(root)
        raise
    print("Commit", commit)
    settings = [local_config, bundle["house_policy.json"]]
    if topology is not None:
        settings.append(topology)
    print("Config SHA256", sha256(b"\0".join(settings)).hexdigest())
    print("Backup", destination)


if __name__ == "__main__":
    main()
