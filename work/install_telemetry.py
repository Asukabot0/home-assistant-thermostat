"""Install the thermostat evidence LaunchAgent for this Docker Desktop user."""
import argparse
from pathlib import Path
import json
import os
import plistlib
import secrets
import shutil
import subprocess
import sys

from release_local import install_bytes


LABEL = "local.codex.home-assistant.thermostat-telemetry"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production",type=Path,required=True)
    parser.add_argument("--repo",required=True)
    parser.add_argument("--pilot-container",required=True)
    parser.add_argument("--shadow-container",required=True)
    parser.add_argument("--python",default=sys.executable)
    args=parser.parse_args()
    home=Path.home()
    root=home/"Library/Application Support/HomeAssistantThermostat/telemetry"
    root.mkdir(mode=0o700,parents=True,exist_ok=True)
    root.chmod(0o700)
    source=Path(__file__).resolve().parents[1]/"home-assistant/telemetry"
    config_path=root/"config.json"
    old=json.loads(config_path.read_bytes()) if config_path.exists() else {}
    docker, gh=shutil.which("docker"),shutil.which("gh")
    if not docker or not gh:
        raise RuntimeError("Docker CLI and authenticated GitHub CLI are required")
    config={"repo":args.repo,"production":str(args.production.resolve()),"docker":docker,"gh":gh,
            "pilot_container":args.pilot_container,"shadow_container":args.shadow_container,
            "anonymization_key":old.get("anonymization_key") or secrets.token_hex(32),
            "local_bytes":512*1024*1024}
    for name in ("worker.py","evidence.py"):
        install_bytes(root/name,(source/name).read_bytes(),0o600)
    install_bytes(config_path,(json.dumps(config,indent=2)+"\n").encode(),0o600)
    destination=home/f"Library/LaunchAgents/{LABEL}.plist"
    destination.parent.mkdir(exist_ok=True)
    target=f"gui/{os.getuid()}"
    control="/bin/launchctl"
    if subprocess.run([control,"print",f"{target}/{LABEL}"],capture_output=True).returncode==0:
        subprocess.run([control,"bootout",f"{target}/{LABEL}"],check=True)
    props={"Label":LABEL,"ProgramArguments":[args.python,"-B",str(root/"worker.py"),str(config_path)],
           "StartCalendarInterval":{"Minute":5},"RunAtLoad":True,"LowPriorityIO":True,
           "ProcessType":"Background","ThrottleInterval":60,
           "EnvironmentVariables":{"TZ":"UTC","GH_PROMPT_DISABLED":"1","GH_PAGER":"cat","GIT_TERMINAL_PROMPT":"0",
                                   "PATH":"/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"},
           "StandardOutPath":str(root/"worker.log"),"StandardErrorPath":str(root/"worker.err.log")}
    install_bytes(destination,plistlib.dumps(props),0o600)
    subprocess.run(["/usr/bin/plutil","-lint",str(destination)],check=True)
    subprocess.run([control,"enable",f"{target}/{LABEL}"],check=True)
    subprocess.run([control,"bootstrap",target,str(destination)],check=True)
    print(f"LaunchAgent {target}/{LABEL}")
    print(f"State and logs {root}")


if __name__=="__main__":
    main()
