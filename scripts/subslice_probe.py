#!/usr/bin/env python3
"""Step 0 Live Pre-Flight & Sub-Slice Verification Probe on YOUR_TPU_V5P_32_NODEPOOL (2x4x4).

Runs across all 8 hosts (32 v5p chips) of the LeaderWorkerSet:
1. Installs stack + TPU v5p compatibility patch + stages Hybrid /dev/shm/DeepSeek-V4.1-Flash
   (model-00047 & model-00048 in /dev/shm + 00001..00046 symlinked to /models) in parallel.
2. Captures full host & GCE TPU metadata (RAM, /dev/shm, tpu-env, worker-network-endpoints).
3. Executes live multi-host torch_tpu ICI all_reduce + all_to_all_single checks for:
   - Full slice: 32 chips (2x4x4, 8 hosts) using real GCE metadata + vbar.sock
   - Sub-slice 16+16: two concurrent independent 16-chip (2x2x4 + 2x2x4) instances
   - Sub-slice 8+24: concurrent 8-chip (2x4x1) + 24-chip (2x4x3) instances (with 75s timeout)
   - Sub-slice 16+8: concurrent 16-chip (2x2x4) + 8-chip (2x2x2) instances
4. Persists subslice_verdict.json to /models/dsv41-v5p/step0_probe/.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

ICI_WORKER_CODE = r"""
import datetime as _dt, json, os, sys, time, traceback
try:
    import torch
    import torch.distributed as dist
    import torch_tpu

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    b_port = int(os.environ.get("TORCH_TPU_PRE_INIT_BARRIER_PORT", "0") or "0")
    t0 = time.time()

    if world > 4 and b_port > 0:
        b_store = dist.TCPStore(
            host_name=os.environ["MASTER_ADDR"],
            port=b_port,
            world_size=world,
            is_master=(rank == 0),
            timeout=_dt.timedelta(seconds=90),
        )
        b_store.set(f"r_{rank}", "1")
        b_store.wait([f"r_{i}" for i in range(world)], _dt.timedelta(seconds=90))

    dev = torch.device("tpu")
    c10d_port = int(os.environ["MASTER_PORT"]) + 100
    dist.init_process_group(
        backend="tpu_dist",
        init_method=f"tcp://{os.environ['MASTER_ADDR']}:{c10d_port}",
        world_size=world,
        rank=rank,
        timeout=_dt.timedelta(seconds=90),
    )

    x = torch.ones((1024, 1024), dtype=torch.bfloat16, device=dev) * (rank + 1)
    dist.all_reduce(x)
    torch.tpu.synchronize()
    ar_val = float(x[0, 0].cpu().item())
    ar_exp = float(sum(range(1, world + 1)))
    assert ar_val == ar_exp, f"rank {rank}: all_reduce got {ar_val}, expected {ar_exp}"

    send_t = torch.arange(world, dtype=torch.float32, device=dev) + (rank * 100.0)
    recv_t = torch.zeros(world, dtype=torch.float32, device=dev)
    dist.all_to_all_single(recv_t, send_t)
    torch.tpu.synchronize()
    recv_cpu = recv_t.cpu().tolist()
    exp_cpu = [src * 100.0 + rank for src in range(world)]
    assert recv_cpu == exp_cpu, f"rank {rank}: all_to_all got {recv_cpu}, expected {exp_cpu}"

    dt = round(time.time() - t0, 3)
    if local_rank == 0:
        print("ICI_PASS " + json.dumps({
            "rank": rank,
            "world_size": world,
            "all_reduce_sum": ar_val,
            "all_to_all_ok": True,
            "elapsed_s": dt,
        }), flush=True)
    dist.destroy_process_group()
except Exception:
    print("ICI_EXCEPTION:\n" + traceback.format_exc(), flush=True)
    sys.exit(1)
"""


def log(tag: str, msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] [{tag}] {msg}", flush=True)


def get_meta(rel_path: str) -> str:
    req = urllib.request.Request(
        f"http://169.254.169.254/computeMetadata/v1/{rel_path}",
        headers={"Metadata-Flavor": "Google"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.read().decode("utf-8", errors="replace").strip()
    except Exception as e:
        return f"<ERR:{e}>"


def get_primary_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("<private-ip>", 1))
        return s.getsockname()[0]
    finally:
        s.close()


def discover_slice_hosts() -> list[dict]:
    full_hn = get_meta("instance/hostname")
    domain = full_hn.split(".", 1)[1]
    eps = [x.strip() for x in get_meta("instance/attributes/worker-network-endpoints").split(",") if x.strip()]
    out = []
    for wid, shn in enumerate(eps):
        ip = socket.gethostbyname(f"{shn}.{domain}")
        out.append({"worker_id": wid, "short_hostname": shn, "ip": ip})
    return out


def start_http_coord(port: int = 29699):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading

    store: dict[str, dict[str, dict]] = {}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def do_POST(self):
            parts = self.path.strip("/").split("/")
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}
            step, wid = parts[1], parts[2]
            with lock:
                store.setdefault(step, {})[wid] = body
                cnt = len(store[step])
            out = json.dumps({"ok": True, "ready": cnt}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def do_GET(self):
            parts = self.path.strip("/").split("/")
            if parts[0] == "healthz":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")
                return
            step = parts[1]
            with lock:
                items = dict(store.get(step, {}))
            out = json.dumps({"ready": len(items), "items": items}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    return srv


def tcp_exchange(w0_ip: str, port: int, step: str, worker_id: int, payload: dict, num_hosts: int = 8, timeout_s: int = 300) -> dict[str, dict]:
    t0 = time.time()
    data_bytes = json.dumps(payload).encode("utf-8")
    posted = False
    while not posted:
        try:
            req = urllib.request.Request(
                f"http://{w0_ip}:{port}/report/{step}/{worker_id}",
                data=data_bytes,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=5).read()
            posted = True
        except Exception:
            if time.time() - t0 > timeout_s:
                raise TimeoutError(f"POST to {w0_ip}:{port} for {step} timed out")
            time.sleep(0.5)

    while True:
        try:
            with urllib.request.urlopen(f"http://{w0_ip}:{port}/step/{step}", timeout=5) as r:
                resp = json.loads(r.read().decode("utf-8"))
            if resp["ready"] >= num_hosts:
                return resp["items"]
        except Exception:
            pass
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"Barrier {step} timed out waiting for {num_hosts} hosts")
        time.sleep(0.5)


def run_ici_group(
    group_name: str,
    member_hosts: list[dict],
    my_worker_id: int,
    full_slice: bool,
    topology: str,
    host_bounds: str,
    wrap: str,
    accel_type: str,
    master_port: int,
    mds_port: int = 8088,
    timeout_s: int = 120,
) -> dict:
    member_ids = [h["worker_id"] for h in member_hosts]
    if my_worker_id not in member_ids:
        return {"group": group_name, "participated": False, "ok": True}

    sub_idx = member_ids.index(my_worker_id)
    nnodes = len(member_hosts)
    master_ip = member_hosts[0]["ip"]
    endpoints_csv = ",".join(h["short_hostname"] for h in member_hosts)
    sb_addrs = ",".join(
        f"{h['ip']}:{8070 + slot}"
        for h in member_hosts
        for slot in range(4)
    )

    env = os.environ.copy()
    for k in list(env.keys()):
        if k.startswith("TPU_") or k in {
            "VBAR_CONTROL_SERVICE_URL",
            "TORCH_TPU_TOPOLOGY",
            "TORCH_TPU_SLICEBUILDER_ADDRESSES",
            "HOST_BOUNDS",
            "CHIPS_PER_HOST_BOUNDS",
            "LD_PRELOAD",
            "GCE_METADATA_HOST",
            "GCE_METADATA_IP",
            "CLOUD_TPU_TASK_ID",
        }:
            env.pop(k, None)

    mds_proc = None
    procs = []
    out_files = []
    try:
        if full_slice:
            if Path("/var/run/tpu-plugin/vbar.sock").is_socket():
                env["VBAR_CONTROL_SERVICE_URL"] = "unix:///var/run/tpu-plugin/vbar.sock"
        else:
            mds_cmd = [
                sys.executable,
                "/workspace/v5p_subslice_metadata.py",
                "--port",
                str(mds_port),
                "--worker-id",
                str(sub_idx),
                "--worker-endpoints",
                endpoints_csv,
                "--topology",
                topology,
                "--host-bounds",
                host_bounds,
                "--chips-per-host-bounds",
                "2,2,1",
                "--wrap",
                wrap,
                "--accelerator-type",
                accel_type,
            ]
            mds_env = {k: v for k, v in os.environ.items() if k != "LD_PRELOAD"}
            mds_proc = subprocess.Popen(mds_cmd, env=mds_env)
            for _ in range(30):
                try:
                    req = urllib.request.Request(
                        f"http://127.0.0.1:{mds_port}/computeMetadata/v1/instance/attributes/healthz",
                        headers={"Metadata-Flavor": "Google"},
                    )
                    if urllib.request.urlopen(req, timeout=1).read() == b"ok":
                        break
                except Exception:
                    time.sleep(0.2)
            env["MOCK_METADATA_PORT"] = str(mds_port)
            env["LD_PRELOAD"] = "/workspace/metadata_hook_8088.so"

        dims = topology.replace("x", ",")
        env["TORCH_TPU_TOPOLOGY"] = dims
        env["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = sb_addrs
        env["TPU_NUM_HOSTS"] = str(nnodes)
        env["NNODES"] = str(nnodes)
        env["NODE_RANK"] = str(sub_idx)
        env["TPU_RUNTIME_METRICS_PORTS"] = "8431,8432,8433,8434"
        env["TORCH_TPU_PRE_INIT_BARRIER_PORT"] = str(master_port + 50)
        env["MASTER_ADDR"] = master_ip
        env["MASTER_PORT"] = str(master_port)
        env["WORLD_SIZE"] = str(nnodes * 4)
        env["LOCAL_WORLD_SIZE"] = "4"

        subprocess.run(
            ["bash", "-lc", "rm -rf /dev/shm/psm_* /dev/shm/sem.* || true"],
            check=False,
        )
        worker_py = Path(f"/tmp/ici_check_{group_name}.py")
        worker_py.write_text(ICI_WORKER_CODE)
        t0 = time.time()
        for slot in range(4):
            c_env = env.copy()
            c_env["RANK"] = str(sub_idx * 4 + slot)
            c_env["LOCAL_RANK"] = str(slot)
            f_out = open(f"/tmp/ici_{group_name}_s{slot}.log", "w+")
            out_files.append(f_out)
            procs.append(
                subprocess.Popen(
                    [sys.executable, str(worker_py)],
                    env=c_env,
                    stdout=f_out,
                    stderr=subprocess.STDOUT,
                )
            )
        rcs = []
        deadline = t0 + timeout_s
        for p in procs:
            rem = max(1.0, deadline - time.time())
            rcs.append(p.wait(timeout=rem))
        dt = round(time.time() - t0, 2)
        combined = ""
        for slot, f_out in enumerate(out_files):
            f_out.flush()
            f_out.seek(0)
            combined += f"=== slot {slot} (rc={rcs[slot]}) ===\n" + f_out.read()[-1500:] + "\n"
            f_out.close()
        out_files.clear()
        ok = all(rc == 0 for rc in rcs) and "ICI_PASS" in combined
        return {
            "group": group_name,
            "participated": True,
            "sub_idx": sub_idx,
            "nnodes": nnodes,
            "world_size": nnodes * 4,
            "topology": topology,
            "host_bounds": host_bounds,
            "ok": ok,
            "returncode": rcs,
            "elapsed_s": dt,
            "stdout_tail": combined[-3500:],
            "stderr_head": combined[:2500],
            "stderr_tail": combined[-2500:],
        }
    except subprocess.TimeoutExpired:
        for p in procs:
            try:
                p.kill()
            except Exception:
                pass
        combined = ""
        for slot, f_out in enumerate(out_files):
            try:
                f_out.flush()
                f_out.seek(0)
                combined += f"=== slot {slot} (TIMEOUT) ===\n" + f_out.read()[-1200:] + "\n"
                f_out.close()
            except Exception:
                pass
        return {
            "group": group_name,
            "participated": True,
            "sub_idx": sub_idx,
            "nnodes": nnodes,
            "world_size": nnodes * 4,
            "topology": topology,
            "host_bounds": host_bounds,
            "ok": False,
            "error": f"TimeoutExpired({timeout_s}s)",
            "stderr_head": combined[:2000],
            "stderr_tail": combined[-2000:],
        }
    finally:
        if mds_proc is not None:
            mds_proc.terminate()
            try:
                mds_proc.wait(timeout=5)
            except Exception:
                mds_proc.kill()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scripts-dir", default="/workspace/scripts_live")
    ap.add_argument("--probe-dir", default="/models/dsv41-v5p/step0_probe")
    ap.add_argument("--coord-port", type=int, default=29699)
    ap.add_argument("--skip-setup", action="store_true")
    ap.add_argument("--run-tag", default="r3")
    ap.add_argument("--hold-after", action="store_true")
    args = ap.parse_args()

    raw_worker_id = get_meta("instance/attributes/agent-worker-number")
    worker_id = int(raw_worker_id)
    tag = f"w{worker_id}"
    rt = args.run_tag

    if not args.skip_setup:
        subprocess.run(
            [
                sys.executable,
                str(Path(args.scripts_dir) / "in_pod_setup_v5p.py"),
                "--scripts-dir",
                str(args.scripts_dir),
                "--summary-out",
                f"/tmp/setup_w{worker_id}.json",
            ],
            check=True,
        )

    base_hosts = discover_slice_hosts()
    w0_ip = base_hosts[0]["ip"]
    coord_srv = start_http_coord(args.coord_port) if worker_id == 0 else None

    free_out = subprocess.check_output(["free", "-h"], text=True).strip()
    df_shm = subprocess.check_output(["df", "-h", "/dev/shm"], text=True).strip()
    short_hn = get_meta("instance/hostname").split(".")[0]
    host_meta = {
        "worker_id": worker_id,
        "lws_worker_index": int(os.environ.get("LWS_WORKER_INDEX", "-1")),
        "k8s_hostname": socket.gethostname(),
        "short_hostname": short_hn,
        "ip": base_hosts[worker_id]["ip"],
        "agent_worker_number": raw_worker_id,
        "worker_network_endpoints": get_meta("instance/attributes/worker-network-endpoints"),
        "tpu_env": get_meta("instance/attributes/tpu-env"),
        "vbar_sock_exists": Path("/var/run/tpu-plugin/vbar.sock").is_socket(),
        "free_h": free_out,
        "df_shm": df_shm,
    }
    n_hosts = len(base_hosts)
    hosts_map = tcp_exchange(w0_ip, args.coord_port, f"hosts_{rt}", worker_id, host_meta, num_hosts=n_hosts)
    hosts = [hosts_map[str(i)] for i in range(n_hosts)]
    log(tag, f"TCP coordinated {n_hosts} hosts (zero GCSFuse): w0={w0_ip} my_ip={host_meta['ip']}")

    if n_hosts == 4:
        log(tag, "Running Test A (v5p-16): Full 16-chip (2x2x4) ICI check...")
        r_full16 = run_ici_group(
            "full_16",
            hosts,
            worker_id,
            full_slice=True,
            topology="2x2x4",
            host_bounds="1,1,4",
            wrap="false,false,false",
            accel_type="v5p-32",
            master_port=29610,
            timeout_s=100,
        )
        all_full16 = tcp_exchange(w0_ip, args.coord_port, f"full16_{rt}", worker_id, r_full16, num_hosts=4)

        log(tag, "Running Test B (v5p-16): Concurrent 8 + 8 (2x2x2 + 2x2x2) sub-slice ICI check...")
        g01 = [h for h in hosts if h["worker_id"] in {0, 1}]
        g23 = [h for h in hosts if h["worker_id"] in {2, 3}]
        if worker_id in {0, 1}:
            r88 = run_ici_group(
                "sub8_hz01",
                g01,
                worker_id,
                full_slice=False,
                topology="2x2x2",
                host_bounds="1,1,2",
                wrap="false,false,false",
                accel_type="v5p-16",
                master_port=29620,
                timeout_s=100,
            )
        else:
            r88 = run_ici_group(
                "sub8_hz23",
                g23,
                worker_id,
                full_slice=False,
                topology="2x2x2",
                host_bounds="1,1,2",
                wrap="false,false,false",
                accel_type="v5p-16",
                master_port=29621,
                timeout_s=100,
            )
        all_88 = tcp_exchange(w0_ip, args.coord_port, f"sub88_{rt}", worker_id, r88, num_hosts=4)
        if worker_id == 0:
            verdict = {
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "slice_hosts": 4,
                "hosts": hosts,
                "full_16_2x2x4_ok": all(all_full16[str(i)]["ok"] for i in range(4)),
                "subslice_8_8_2x2x2_ok": all(all_88[str(i)]["ok"] for i in range(4)),
                "details": {
                    "full_16_w0": all_full16["0"],
                    "sub_8_8_w0": all_88["0"],
                    "sub_8_8_w2": all_88["2"],
                },
            }
            Path("/tmp/subslice_verdict_v5p16.json").write_text(json.dumps(verdict, indent=2))
            log(tag, "WROTE /tmp/subslice_verdict_v5p16.json: " + json.dumps({k: v for k, v in verdict.items() if k not in ("hosts", "details")}))
        tcp_exchange(w0_ip, args.coord_port, f"done_{rt}", worker_id, {"done": True}, num_hosts=4)
        if coord_srv is not None:
            time.sleep(2.0)
            coord_srv.shutdown()
        return

    # Test A: Full 32-chip (2x4x4) ICI check
    log(tag, "Running Test A: Full 32-chip (2x4x4) ICI check...")
    r_full = run_ici_group(
        "full_32",
        hosts,
        worker_id,
        full_slice=True,
        topology="2x4x4",
        host_bounds="1,2,4",
        wrap="false,false,false",
        accel_type="v5p-64",
        master_port=29610,
        timeout_s=120,
    )
    all_full = tcp_exchange(w0_ip, args.coord_port, f"full32_{rt}", worker_id, r_full, num_hosts=8)

    # Test B: Concurrent 16 + 16 (2x2x4 + 2x2x4, hy=0 vs hy=1)
    log(tag, "Running Test B: Concurrent 16 + 16 (2x2x4 + 2x2x4) sub-slice ICI check...")
    group_hy0 = [h for h in hosts if h["worker_id"] in {0, 1, 2, 3}]
    group_hy1 = [h for h in hosts if h["worker_id"] in {4, 5, 6, 7}]
    if worker_id in {0, 1, 2, 3}:
        r16 = run_ici_group(
            "sub16_hy0",
            group_hy0,
            worker_id,
            full_slice=False,
            topology="2x2x4",
            host_bounds="1,1,4",
            wrap="false,false,false",
            accel_type="v5p-32",
            master_port=29620,
            timeout_s=100,
        )
    else:
        r16 = run_ici_group(
            "sub16_hy1",
            group_hy1,
            worker_id,
            full_slice=False,
            topology="2x2x4",
            host_bounds="1,1,4",
            wrap="false,false,false",
            accel_type="v5p-32",
            master_port=29621,
            timeout_s=100,
        )
    all_16_16 = tcp_exchange(w0_ip, args.coord_port, f"sub16_16_{rt}", worker_id, r16, num_hosts=8)

    # Test C: Concurrent 8 + 24 (2x4x1 hz=0 + 2x4x3 hz=1..3) with 60s timeout
    log(tag, "Running Test C: Concurrent 8 + 24 (2x4x1 + 2x4x3) sub-slice ICI check...")
    group_hz0 = [h for h in hosts if h["worker_id"] in {0, 4}]
    group_hz123 = [h for h in hosts if h["worker_id"] in {1, 2, 3, 5, 6, 7}]
    if worker_id in {0, 4}:
        r8_24 = run_ici_group(
            "sub8_2x4x1",
            group_hz0,
            worker_id,
            full_slice=False,
            topology="2x4x1",
            host_bounds="1,2,1",
            wrap="false,false,false",
            accel_type="v5p-16",
            master_port=29630,
            timeout_s=60,
        )
    else:
        r8_24 = run_ici_group(
            "sub24_2x4x3",
            group_hz123,
            worker_id,
            full_slice=False,
            topology="2x4x3",
            host_bounds="1,2,3",
            wrap="false,false,false",
            accel_type="v5p-48",
            master_port=29631,
            timeout_s=60,
        )
    all_8_24 = tcp_exchange(w0_ip, args.coord_port, f"sub8_24_{rt}", worker_id, r8_24, num_hosts=8)

    # Test D: Concurrent 16 + 8 (2x2x4 hy=0 + 2x2x2 hy=1,hz=0..1)
    log(tag, "Running Test D: Concurrent 16 + 8 (2x2x4 + 2x2x2) sub-slice ICI check...")
    group_8_2x2x2 = [h for h in hosts if h["worker_id"] in {4, 5}]
    if worker_id in {0, 1, 2, 3}:
        r16_8 = run_ici_group(
            "sub16_for_16_8",
            group_hy0,
            worker_id,
            full_slice=False,
            topology="2x2x4",
            host_bounds="1,1,4",
            wrap="false,false,false",
            accel_type="v5p-32",
            master_port=29640,
            timeout_s=100,
        )
    elif worker_id in {4, 5}:
        r16_8 = run_ici_group(
            "sub8_2x2x2",
            group_8_2x2x2,
            worker_id,
            full_slice=False,
            topology="2x2x2",
            host_bounds="1,1,2",
            wrap="false,false,false",
            accel_type="v5p-16",
            master_port=29641,
            timeout_s=100,
        )
    else:
        r16_8 = {"group": "idle_w6_w7", "participated": False, "ok": True}
    all_16_8 = tcp_exchange(w0_ip, args.coord_port, f"sub16_8_{rt}", worker_id, r16_8, num_hosts=8)

    if worker_id == 0:
        verdict = {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "hosts": hosts,
            "full_32_2x4x4_ok": all(all_full[str(i)]["ok"] for i in range(8)),
            "subslice_16_16_2x2x4_ok": all(all_16_16[str(i)]["ok"] for i in range(8)),
            "subslice_8_24_2x4x1_2x4x3_ok": all(all_8_24[str(i)]["ok"] for i in range(8)),
            "subslice_16_8_2x2x4_2x2x2_ok": all(all_16_8[str(i)]["ok"] for i in range(8)),
            "details": {
                "full_32_w0": all_full["0"],
                "sub_16_16_w0": all_16_16["0"],
                "sub_16_16_w4": all_16_16["4"],
                "sub_8_24_w0": all_8_24["0"],
                "sub_8_24_w1": all_8_24["1"],
                "sub_16_8_w0": all_16_8["0"],
                "sub_16_8_w4": all_16_8["4"],
            },
            "note_ep24_model_constraint": "DeepSeek-V4.1-Flash-1017 MTP/DSpark has dspark_n_routed_experts=128 (128 % 24 != 0), so EP=24 is unsupported by model expert count regardless of topology; valid EP degrees are 8, 16, 32.",
        }
        Path("/tmp/subslice_verdict.json").write_text(json.dumps(verdict, indent=2))
        log(tag, "WROTE /tmp/subslice_verdict.json: " + json.dumps({k: v for k, v in verdict.items() if k not in ("hosts", "details")}))
    tcp_exchange(w0_ip, args.coord_port, f"done_{rt}", worker_id, {"done": True}, num_hosts=8)
    if coord_srv is not None:
        time.sleep(2.0)
        coord_srv.shutdown()

    if args.hold_after:
        while True:
            time.sleep(60)


if __name__ == "__main__":
    main()
