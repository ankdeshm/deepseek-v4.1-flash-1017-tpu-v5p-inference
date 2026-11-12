#!/usr/bin/env python3
"""Multi-host TPU v5p sub-slice GCE metadata proxy.

IMPORTANT: Launch this daemon WITHOUT LD_PRELOAD so its pass-through calls to
169.254.169.254:80 reach the real GKE/GCE metadata server (e.g., for OAuth
service-account tokens), while child worker processes launched WITH
LD_PRELOAD=/workspace/metadata_hook_8088.so + MOCK_METADATA_PORT=8088 see
the sub-slice TPU topology, worker ID, and worker-network-endpoints.
"""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import sys
import urllib.error
import urllib.parse
import urllib.request


def make_handler(cfg: dict[str, str]):
    tpu_env_map = {
        "TPU_WORKER_ID": cfg["worker_id"],
        "ALT": "false",
        "CHIPS_PER_HOST_BOUNDS": cfg["chips_per_host_bounds"],
        "CONSUMER_PROJECT_ID": cfg["project_id"],
        "ENABLE_ICI_RESILIENCY": "false",
        "ENABLE_IMPROVED_REROUTE_ALLREDUCE_STRATEGY": "false",
        "HOST_BOUNDS": cfg["host_bounds"],
        "INJECT_SLICE_BUILDER_FAULT": "",
        "NODE_ID": cfg["worker_id"],
        "TOPOLOGY": cfg["topology"],
        "TPU_ACCELERATOR_TYPE": cfg["accelerator_type"],
        "TPU_RUNTIME_METRICS_PORTS": "8431,8432,8433,8434",
        "WRAP": cfg["wrap"],
        "ZONE": cfg["zone"],
    }
    tpu_env_yaml = "\n".join(f"{k}: '{v}'" for k, v in tpu_env_map.items()) + "\n"

    class SubsliceMetadataHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            if cfg.get("verbose") == "1":
                sys.stderr.write("[subslice-mds] " + (fmt % args) + "\n")

        def forward_to_real_mds(self) -> None:
            real_url = f"http://169.254.169.254{self.path}"
            req = urllib.request.Request(real_url)
            for k, v in self.headers.items():
                if k.lower() != "host":
                    req.add_header(k, v)
            try:
                with urllib.request.urlopen(req, timeout=5) as resp:
                    body = resp.read()
                    self.send_response(resp.status)
                    for k, v in resp.headers.items():
                        if k.lower() not in {"transfer-encoding", "content-length", "connection"}:
                            self.send_header(k, v)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
            except urllib.error.HTTPError as e:
                body = e.read()
                self.send_response(e.code)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as e:
                body = str(e).encode("utf-8")
                self.send_response(502)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        def do_GET(self) -> None:
            if self.headers.get("Metadata-Flavor") != "Google":
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Missing Metadata-Flavor: Google")
                return

            path = urllib.parse.urlparse(self.path).path.rstrip("/")
            if path == "/computeMetadata/v1/instance/attributes/healthz":
                body = b"ok"
            elif path.endswith("/computeMetadata/v1/instance/attributes/tpu-env"):
                body = tpu_env_yaml.encode("utf-8")
            elif path.endswith("/computeMetadata/v1/instance/attributes/worker-network-endpoints"):
                body = cfg["worker_endpoints"].encode("utf-8")
            elif path.endswith("/computeMetadata/v1/instance/attributes/agent-worker-number"):
                body = cfg["worker_id"].encode("utf-8")
            elif path.endswith("/computeMetadata/v1/instance/attributes/accelerator-type"):
                body = cfg["accelerator_type"].encode("utf-8")
            elif "/computeMetadata/v1/instance/attributes/" in path:
                attr = path.rsplit("/attributes/", 1)[-1]
                if attr in tpu_env_map:
                    body = tpu_env_map[attr].encode("utf-8")
                else:
                    self.forward_to_real_mds()
                    return
            else:
                self.forward_to_real_mds()
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return SubsliceMetadataHandler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8088)
    ap.add_argument("--worker-id", required=True)
    ap.add_argument("--worker-endpoints", required=True, help="Comma-separated sub-slice hostnames or IPs")
    ap.add_argument("--topology", required=True, help="e.g. 2x2x4, 2x2x2, 2x4x1, 2x4x3")
    ap.add_argument("--host-bounds", required=True, help="e.g. 1,1,4 or 1,1,2 or 1,2,1 or 1,2,3")
    ap.add_argument("--chips-per-host-bounds", default="2,2,1")
    ap.add_argument("--wrap", default="false,false,false")
    ap.add_argument("--accelerator-type", required=True, help="e.g. v5p-32, v5p-16, v5p-8")
    ap.add_argument("--zone", default="us-central1-a")
    ap.add_argument("--project-id", default="YOUR_GCP_PROJECT_ID")
    ap.add_argument("--verbose", default="0")
    args = ap.parse_args()

    cfg = {
        "worker_id": str(args.worker_id),
        "worker_endpoints": args.worker_endpoints,
        "topology": args.topology,
        "host_bounds": args.host_bounds,
        "chips_per_host_bounds": args.chips_per_host_bounds,
        "wrap": args.wrap,
        "accelerator_type": args.accelerator_type,
        "zone": args.zone,
        "project_id": args.project_id,
        "verbose": str(args.verbose),
    }
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(cfg))
    print(
        f"[subslice-mds] Serving 127.0.0.1:{args.port} worker_id={cfg['worker_id']} "
        f"topology={cfg['topology']} host_bounds={cfg['host_bounds']} "
        f"accel={cfg['accelerator_type']} endpoints={cfg['worker_endpoints']}",
        flush=True,
    )
    httpd.serve_forever()


if __name__ == "__main__":
    main()
