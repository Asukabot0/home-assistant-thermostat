"""Read-only hourly evidence publisher. The durable queue advances after remote verification."""
import argparse
import base64
from collections import Counter
from datetime import datetime
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import signal
import subprocess
from tempfile import NamedTemporaryFile
from time import sleep, time
from urllib.request import Request, urlopen
from zipfile import ZipFile

from evidence import BEIJING, audit_rows, bundle, collect, iso, json_bytes, load_json, observed_versions, stamp, verify_zip, write_zip


HOUR = 3600
MINUTE = 60
MAX_BYTES = 64 * 1024 * 1024
REVISE_HOURS = 3
MAX_BACKFILL = 4
RUNTIME_PROGRAM = '''
from pathlib import Path
import hashlib,json
p=Path("/opt/shadow")
code=sorted(p.glob("*.py"))
settings=[p/"rooms.json",p/"house_policy.json",p/"topology.json"]
print(json.dumps({"code_sha256":hashlib.sha256(b"\\0".join(x.read_bytes() for x in code)).hexdigest(),
    "config_sha256":hashlib.sha256(b"\\0".join(x.read_bytes() for x in settings)).hexdigest(),
    "files":{x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in code+[settings[1]]}}))
'''


def atomic(path, value):
    with NamedTemporaryFile(dir=path.parent, delete=False) as f:
        f.write(json_bytes(value))
        f.flush()
        os.fsync(f.fileno())
    os.replace(f.name, path)


class RuntimeFailure(Exception):
    pass


class RemoteFailure(Exception):
    pass


def command(args, *, payload=None, optional_404=False, timeout=28):
    try:
        p = subprocess.run(args, input=payload, capture_output=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RemoteFailure(type(error).__name__) from error
    if p.returncode:
        stderr = p.stderr.decode(errors="replace")
        if optional_404 and ("HTTP 404" in stderr or "release not found" in stderr.lower()):
            return None
        code = re.search(r"HTTP (\d{3})",stderr)
        raise RemoteFailure("HTTP_"+code[1] if code else f"command_exit_{p.returncode}")
    return p.stdout


def runtime(config, root, now):
    containers = (config["pilot_container"], config["shadow_container"])
    docker = config["docker"]
    result = {}
    for label, name in zip(("pilot","shadow"), containers):
        actual = json.loads(command([docker,"exec",name,"python","-B","-c",RUNTIME_PROGRAM],timeout=12))
        raw = command([docker,"inspect","-f","{{.State.StartedAt}} {{.RestartCount}} {{.State.Running}}",name],timeout=10)
        started, restarts, running = raw.decode().split()
        result[label] = {**actual,"container_started_at":started,"restart_count":int(restarts),"running":running=="true"}
    rows = audit_rows(root/"pilot-data",now-300,now+1)
    startup = next((r for r in reversed(rows) if r.get("event")=="startup"), None)
    if (not startup or not re.fullmatch(r"[0-9a-f]{40}", startup.get("commit") or "") or
            any(not entry["running"] or any(entry[key]!=startup[key] for key in ("code_sha256","config_sha256")) for entry in result.values())):
        raise RuntimeFailure("startup_hash_or_service_mismatch")
    return {"startup":{k:startup.get(k) for k in ("commit","code_sha256","config_sha256","observation_version","observation_fingerprint","at")},
            "deployed_utc":iso(startup["at"]),"files_match_startup":True,**result}


def runtime_events(config, start, end):
    args=[config["docker"],"logs","--timestamps","--since",iso(start),"--until",iso(end),"--tail","1800",config["shadow_container"]]
    # Docker routes INFO, warning and error streams separately. Only export
    # timestamped protocol categories and exception types, never raw stdout.
    p=subprocess.run(args,capture_output=True,timeout=15)
    if p.returncode:
        return {"error":"shadow_log_unavailable"},[]
    patterns={"Offline shadow service running":"shadow_startup","Shadow input or storage unavailable":"shadow_input_failure",
              "Shadow status output unavailable":"shadow_report_failure","Learning skipped":"learning_failure",
              "AttributeError:":"learning_exception"}
    rows=[]
    for line in (p.stdout+p.stderr).decode(errors="replace").splitlines():
        t=stamp(line.split(" ",1)[0])
        if t is None or not start<=t<end:
            continue
        for marker,category in patterns.items():
            if marker in line:
                rows.append({"at":t,"event":category,"id":sha256(f"{category}:{t}".encode()).hexdigest()})
                break
    rows.sort(key=lambda row:row["at"])
    return {"acquired_at":iso(time()),"rows":len(rows),"source":"shadow_container","tail_limit":1800},rows


class GitHub:
    def __init__(self, config):
        self.gh=config["gh"]
        self.repo=config["repo"]
        if not re.fullmatch(r"[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+",self.repo):
            raise ValueError("Invalid repository")
        self.prefix="repos/"+self.repo

    def api(self, suffix, *, method="GET", body=None, optional=False):
        args=[self.gh,"api","--method",method,self.prefix+suffix]
        if body is not None:
            args.extend(["--input","-"])
        raw=command(args,payload=json_bytes(body) if body is not None else None,optional_404=optional)
        return json.loads(raw) if raw else None

    def release(self, tag):
        return self.api("/releases/tags/"+tag,optional=True)

    def upload(self, path, record, commit):
        date=datetime.fromtimestamp(record["start"],BEIJING).strftime("%Y%m%d")
        daily="telemetry-"+date
        tag=daily
        release=self.release(tag)
        if release and release.get("immutable") and not any(a["name"]==path.name for a in release["assets"]):
            tag=daily+"-"+path.stem.removeprefix("evidence-")
            release=self.release(tag)
        if release is None:
            args=[self.gh,"release","create",tag,str(path),"--repo",self.repo,"--target",commit,"--title",
                  "Thermostat evidence · "+date+" (Beijing)","--notes",
                  "De-identified thermostat evidence. Find windows and checksums in the telemetry branch index.","--prerelease","--latest=false"]
            command(args,timeout=55)
            release=self.release(tag)
        elif not any(a["name"]==path.name for a in release["assets"]):
            command([self.gh,"release","upload",tag,str(path),"--repo",self.repo],timeout=55)
            release=self.release(tag)
        if not release:
            raise RemoteFailure("release_not_visible")
        asset=next((a for a in release["assets"] if a["name"]==path.name and a["state"]=="uploaded"),None)
        if asset is None or asset.get("size")!=path.stat().st_size or asset.get("digest") not in (None,"sha256:"+record["sha256"]):
            raise RemoteFailure("remote_asset_mismatch")
        url=asset["browser_download_url"]
        prefix=f"https://github.com/{self.repo}/releases/download/{tag}/"
        if url != prefix+path.name:
            raise RemoteFailure("unexpected_download_url")
        with NamedTemporaryFile(dir=path.parent) as scratch:
            request=Request(url,headers={"User-Agent":"thermostat-evidence-verifier/1"})
            with urlopen(request,timeout=20) as response:
                scratch.write(response.read(MAX_BYTES+1))
            scratch.flush()
            check=verify_zip(Path(scratch.name),record["sha256"])
        if (check["from_utc"],check["to_utc"])!=(iso(record["start"]),iso(record["end"])):
            raise RemoteFailure("remote_window_mismatch")
        return {"url":url,"release":release["html_url"],"verified_at":iso(time()),"upload_status":"verified"}

    def index(self, records, commit):
        ref=self.api("/git/ref/heads/telemetry",optional=True)
        if ref is None:
            self.api("/git/refs",method="POST",body={"ref":"refs/heads/telemetry","sha":commit})
        filename="telemetry-index.json"
        old=self.api("/contents/"+filename+"?ref=telemetry",optional=True)
        ordered=sorted(records,key=lambda v:(v["end"],v["revision"],v["kind"]))
        public=[{"from_utc":iso(v["start"]),"to_utc":iso(v["end"]),"from_beijing":iso(v["start"],BEIJING),
                 "to_beijing":iso(v["end"],BEIJING),**{k:v.get(k) for k in ("kind","revision","url","sha256","upload_status","supersedes")},
                 "capture_commit":v["commit"], "observed_versions":v.get("observed_versions",[])}
                for v in ordered[-192:]]
        data=json_bytes({"schema":1,"latest":public[-1] if public else None,"packages":public})
        if old and base64.b64decode(old["content"])==data:
            return
        body={"branch":"telemetry","message":"Index verified thermostat evidence", "content":base64.b64encode(data).decode()}
        if old:
            body["sha"]=old["sha"]
        self.api("/contents/"+filename,method="PUT",body=body)


def retry(operation):
    for delay in (1,3,None):
        try:
            return operation()
        except (RemoteFailure, OSError, ValueError):
            if delay is None:
                raise
            sleep(delay)


def last_full_end(now):
    hour=int(now)//HOUR*HOUR
    return hour if now-hour>=5*MINUTE else hour-HOUR


def next_time(now):
    hour=int(now)//HOUR*HOUR
    due=hour+5*MINUTE
    return due if now<due else due+HOUR


def prune(folder, state, now, budget):
    files={p.name:p for p in folder.glob("evidence-*.zip") if p.is_file()}
    total=sum(f.stat().st_size for f in files.values())
    verified=sorted((r for r in state["uploaded"] if r["file"] in files),key=lambda r:r["end"])
    for record in verified:
        path=files[record["file"]]
        if now-record["end"] >= 2*86400 or total>budget:
            total -= path.stat().st_size
            path.unlink()
    if total>budget:
        raise RuntimeFailure("pending_queue_capacity")


def perform(config, directory, state, gh, now):
    root=Path(config["production"])
    path=directory/"queue.json"
    spool=directory/"spool"
    current=runtime(config,root,now)
    release=current["startup"]
    commit=release["commit"]
    # On initial installation also schedule the complete hour preceding the
    # current clock hour. It remains tagged with its actual historical release.
    state.setdefault("cursor",min(int(release["at"])//HOUR*HOUR,int(now)//HOUR*HOUR-HOUR))
    for record in state["uploaded"]:
        path_to_zip=spool/record["file"]
        if "observed_versions" not in record and path_to_zip.is_file():
            verify_zip(path_to_zip,record["sha256"])
            with ZipFile(path_to_zip) as archive:
                record["observed_versions"]=observed_versions(json.loads(archive.read("summary.json")))
    # Resend staged operations first. Network errors retain the archive and
    # neither discard that window nor move its durable cursor.
    if state.get("pending"):
        send_pending(directory,state,gh)
    release_key=[commit,release["config_sha256"]]
    if state.get("initial_verified")!=release_key:
        end=max(int(release["at"])+1,int(now))
        if end > now:
            end=now
        package(config,directory,state,gh,current,(release["at"],max(release["at"]+0.001,end)),"state")
    cursor=state["cursor"]
    limit=last_full_end(now)
    for _ in range(MAX_BACKFILL):
        if cursor+HOUR>limit:
            break
        package(config,directory,state,gh,current,(cursor,cursor+HOUR),"hourly")
        cursor=state["cursor"]
    for end in (limit-offset*HOUR for offset in range(1,REVISE_HOURS+1)):
        if any(r["kind"]=="hourly" and r["end"]==end for r in state["uploaded"]):
            package(config,directory,state,gh,current,(end-HOUR,end),"hourly",revision_only=True)
    prune(spool,state,now,config["local_bytes"])
    state["last_error"]=None
    state["last_check"]=iso(now)
    atomic(path,state)


def package(config,directory,state,gh,current,window,kind,revision_only=False):
    path=directory/"queue.json"
    spool=directory/"spool"
    start,end=window
    prior=next((x for x in reversed(state["uploaded"]) if x["end"]<=start),None) or next(iter(reversed(state["uploaded"])),None)
    last=next((x for x in reversed(state["uploaded"]) if (x["start"],x["end"],x["kind"])==(start,end,kind)),None)
    snapshot=collect(Path(config["production"]),start,end,current,config["anonymization_key"])
    try:
        mark, events=runtime_events(config,start,end)
    except (OSError, subprocess.TimeoutExpired):
        mark,events={"error":"shadow_log_read_timeout"},[]
    snapshot["raw"]["shadow-operations"]=events
    snapshot["watermarks"]["shadow_operations"]=mark
    manifest,files=bundle(window,kind,snapshot,prior["url"] if prior else None,last["url"] if last else None)
    if last and manifest["logical_sha256"]==last["logical_sha256"]:
        return
    revision=last["revision"]+1 if last else 1
    when=datetime.fromtimestamp(end,BEIJING).strftime("%Y%m%dT%H%M%S")
    filename=f"evidence-{when}-{kind}-r{revision}-{manifest['logical_sha256'][:12]}.zip"
    temp=spool/(filename+".tmp")
    target=spool/filename
    prune(spool,state,time(),config["local_bytes"]-MAX_BYTES)
    write_zip(temp,files)
    if temp.stat().st_size>MAX_BYTES:
        temp.unlink()
        raise RuntimeFailure("archive_size_limit")
    digest=sha256(temp.read_bytes()).hexdigest()
    verify_zip(temp,digest)
    os.replace(temp,target)
    state["pending"]={"file":filename,"sha256":digest,"logical_sha256":manifest["logical_sha256"],
                      "start":start,"end":end,"kind":kind,"revision":revision,"supersedes":last["url"] if last else None,
                      "advance":kind=="hourly" and not revision_only,"commit":current["startup"]["commit"],
                      "config_sha256":current["startup"]["config_sha256"],"observed_versions":manifest["observed_versions"],
                      "upload_status":"queued"}
    atomic(path,state)
    send_pending(directory,state,gh)


def send_pending(directory,state,gh):
    record=state["pending"]
    pkg=directory/"spool"/record["file"]
    verify_zip(pkg,record["sha256"])
    uploaded=retry(lambda: gh.upload(pkg,record,record["commit"]))
    record={**record,**uploaded}
    all_records=[*state["uploaded"],record]
    retry(lambda: gh.index(all_records,record["commit"]))
    state["uploaded"]=all_records
    state["last_success"]=record["verified_at"]
    if record["advance"]:
        state["cursor"]=max(state["cursor"],record["end"])
    if record["kind"]=="state":
        state["initial_verified"]=[record["commit"],record.get("config_sha256")]
    state["pending"]=None
    atomic(directory/"queue.json",state)
    print(json.dumps({"event":"uploaded","window_end":iso(record["end"]),"sha256":record["sha256"],"url":record["url"]}),flush=True)


def status(directory,state,now):
    latest=(state["uploaded"][-1] if state["uploaded"] else None)
    stop=last_full_end(now)
    cursor=state.get("cursor",stop)
    atomic(directory/"status.json",{"last_success":state.get("last_success"),"last_error":state.get("last_error"),
            "last_check":state.get("last_check"),"pending":state.get("pending"),"backfill_from_utc":iso(cursor),
            "backfill_completed_hour_count":max(0,(stop-cursor)//HOUR),"next_due_utc":iso(next_time(now)),
            "next_due_beijing":iso(next_time(now),BEIJING),"latest_upload":latest})


def expire(*_):
    raise RuntimeFailure("job_deadline")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config",type=Path)
    args=parser.parse_args()
    os.umask(0o077)
    path=args.config.resolve()
    if path.stat().st_mode & 0o077:
        raise PermissionError("Telemetry configuration must be owner-only")
    config=load_json(path)
    directory=path.parent
    (directory/"spool").mkdir(exist_ok=True)
    with (directory/"upload.lock").open("a") as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        state=load_json(directory/"queue.json") if (directory/"queue.json").exists() else {"uploaded":[]}
        now=time()
        signal.signal(signal.SIGALRM,expire)
        signal.alarm(280)
        try:
            os.nice(8)
            perform(config,directory,state,GitHub(config),now)
            return 0
        except Exception as error:
            # Do not log subprocess output, credentials, paths or arbitrary input text.
            location=error.__traceback__
            while location.tb_next:
                location=location.tb_next
            state["last_error"]={"at":iso(time()),"type":type(error).__name__,
                                 "reason":str(error) if isinstance(error,(RemoteFailure,RuntimeFailure)) else None,
                                 "function":location.tb_frame.f_code.co_name,"line":location.tb_lineno}
            state["last_check"]=iso(now)
            atomic(directory/"queue.json",state)
            print(json.dumps({"event":"failed",**state["last_error"]}),flush=True)
            return 1
        finally:
            signal.alarm(0)
            status(directory,state,time())


if __name__=="__main__":
    raise SystemExit(main())
