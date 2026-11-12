#!/usr/bin/env python3
"""Renders hostname-pinned LeaderWorkerSet manifests for TPU v5p-32 sub-slices.

Reads Step 0's subslice_verdict.json (which maps physical GCE agent-worker-number 0..7
to Kubernetes node hostnames on YOUR_TPU_V5P_32_NODEPOOL) and generates:
  - llmd-prefill-ep16.yaml  (hy=0: leader=w0, workers=[w1, w2, w3], :8000 direct)
  - llmd-decode-ep16.yaml   (hy=1: leader=w4 + llm-d-router-disagg-sidecar :8000->:9400, workers=[w5, w6, w7])
  - llmd-decode-ep8.yaml    (hy=1,hz=0..1: leader=w4 + llm-d-router-disagg-sidecar :8000->:9400, worker=[w5])
  - colocated-ep16.yaml     (hy=0: leader=w0, workers=[w1, w2, w3], :8000 direct, fallback/comparison)
  - colocated-ep8.yaml      (hy=0,hz=0..1: leader=w0, worker=[w1], :8000 direct, fallback/comparison)

Why hostname-pinned nodeAffinity WITHOUT cloud.google.com/gke-tpu-topology in nodeSelector?
  1. GKE's TPU mutating webhook enforces pod count == 8 if cloud.google.com/gke-tpu-topology: 2x4x4
     is present on nodeSelector, whereas omitting gke-tpu-topology and pinning kubernetes.io/hostname
     allows multiple independent LeaderWorkerSets (e.g. size=4 + size=4 or size=4 + size=2) to
     coexist on the 8 hosts of YOUR_TPU_V5P_32_NODEPOOL.
  2. Pinning leaderTemplate to sub-slice worker 0 (w0 or w4) guarantees LWS leader (which carries
     llm-d.ai/inference-serving="true" and the routing sidecar) is ALWAYS vLLM head rank 0!
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


IMAGE = "docker.io/vllm/vllm-tpu@sha256:f9565be370a7af2b631198aa95f891221a974b2b1727d22db9b275923d3261c6"
SIDECAR_IMAGE = "ghcr.io/llm-d/llm-d-router-disagg-sidecar:v1.0.0"


def render_lws(
    name: str,
    role: str,
    ep_size: int,
    leader_host: str,
    worker_hosts: list[str],
    run_id: str,
    max_num_seqs: int,
    mnbt: int,
    llmd_enabled: bool,
) -> str:
    size = 1 + len(worker_hosts)
    llmd_labels = ""
    if llmd_enabled:
        llmd_labels = (
            '          llm-d.ai/inference-serving: "true"\n'
            '          llm-d.ai/guide: "pd-disaggregation"\n'
            f'          llm-d.ai/role: "{role}"\n'
        )

    sidecar_block = ""
    if llmd_enabled and role == "decode":
        sidecar_block = f"""        initContainers:
          - name: routing-proxy
            image: {SIDECAR_IMAGE}
            imagePullPolicy: IfNotPresent
            args:
              - --port=8000
              - --model-server-port=9400
              - --kv-connector=nixlv2
              - --zap-log-level=1
              - --secure-serving=false
              - --metrics-port=9091
            ports:
              - name: sidecar
                containerPort: 8000
                protocol: TCP
              - name: sidecar-metrics
                containerPort: 9091
                protocol: TCP
            restartPolicy: Always
            securityContext:
              allowPrivilegeEscalation: false
              runAsNonRoot: true
"""

    worker_hosts_yaml = "\n".join(f'                      - "{h}"' for h in worker_hosts)

    return f"""apiVersion: leaderworkerset.x-k8s.io/v1
kind: LeaderWorkerSet
metadata:
  name: {name}
  namespace: default
spec:
  replicas: 1
  leaderWorkerTemplate:
    size: {size}
    restartPolicy: RecreateGroupOnPodRestart
    leaderTemplate:
      metadata:
        labels:
          app: {name}
          role: leader
{llmd_labels}        annotations:
          gke-gcsfuse/volumes: "true"
          cluster-autoscaler.kubernetes.io/safe-to-evict: "false"
      spec:
        serviceAccountName: default
        hostNetwork: true
        dnsPolicy: ClusterFirstWithHostNet
        nodeSelector:
          cloud.google.com/gke-nodepool: YOUR_TPU_V5P_32_NODEPOOL
          kubernetes.io/hostname: "{leader_host}"
        tolerations:
          - key: "google.com/tpu"
            operator: "Exists"
            effect: "NoSchedule"
          - key: "cloud.google.com/gke-spot"
            operator: "Exists"
            effect: "NoSchedule"
          - key: "cloud.google.com/gke-queued"
            operator: "Exists"
            effect: "NoSchedule"
{sidecar_block}        containers:
          - name: vllm
            image: {IMAGE}
            imagePullPolicy: IfNotPresent
            securityContext:
              privileged: true
            command: ["bash", "-lc"]
            args:
              - |
                exec python3 -u /models/dsv41-v5p/scripts/v5p_vllm_driver.py \\
                  --run-id="{run_id}" \\
                  --role={role} \\
                  --ep-size={ep_size} \\
                  --subslice-mode=subslice \\
                  --max-num-seqs={max_num_seqs} \\
                  --max-num-batched-tokens={mnbt}
            readinessProbe:
              httpGet:
                path: /v1/models
                port: {9400 if (llmd_enabled and role == "decode") else 8000}
              periodSeconds: 5
              timeoutSeconds: 3
              failureThreshold: 3
            resources:
              requests:
                google.com/tpu: "4"
              limits:
                google.com/tpu: "4"
            volumeMounts:
              - name: dshm
                mountPath: /dev/shm
              - name: tpu-plugin
                mountPath: /var/run/tpu-plugin
              - name: models-gcs
                mountPath: /models
        volumes:
          - name: dshm
            emptyDir:
              medium: Memory
              sizeLimit: 240Gi
          - name: tpu-plugin
            hostPath:
              path: /var/run/tpu-plugin
              type: DirectoryOrCreate
          - name: models-gcs
            persistentVolumeClaim:
              claimName: YOUR_MODELS_PVC
    workerTemplate:
      metadata:
        labels:
          app: {name}
          role: worker
        annotations:
          gke-gcsfuse/volumes: "true"
          cluster-autoscaler.kubernetes.io/safe-to-evict: "false"
      spec:
        serviceAccountName: default
        hostNetwork: true
        dnsPolicy: ClusterFirstWithHostNet
        nodeSelector:
          cloud.google.com/gke-nodepool: YOUR_TPU_V5P_32_NODEPOOL
        affinity:
          nodeAffinity:
            requiredDuringSchedulingIgnoredDuringExecution:
              nodeSelectorTerms:
                - matchExpressions:
                    - key: kubernetes.io/hostname
                      operator: In
                      values:
{worker_hosts_yaml}
          podAntiAffinity:
            requiredDuringSchedulingIgnoredDuringExecution:
              - labelSelector:
                  matchLabels:
                    app: {name}
                topologyKey: kubernetes.io/hostname
        tolerations:
          - key: "google.com/tpu"
            operator: "Exists"
            effect: "NoSchedule"
          - key: "cloud.google.com/gke-spot"
            operator: "Exists"
            effect: "NoSchedule"
          - key: "cloud.google.com/gke-queued"
            operator: "Exists"
            effect: "NoSchedule"
        containers:
          - name: vllm
            image: {IMAGE}
            imagePullPolicy: IfNotPresent
            securityContext:
              privileged: true
            command: ["bash", "-lc"]
            args:
              - |
                exec python3 -u /models/dsv41-v5p/scripts/v5p_vllm_driver.py \\
                  --run-id="{run_id}" \\
                  --role={role} \\
                  --ep-size={ep_size} \\
                  --subslice-mode=subslice \\
                  --max-num-seqs={max_num_seqs} \\
                  --max-num-batched-tokens={mnbt}
            resources:
              requests:
                google.com/tpu: "4"
              limits:
                google.com/tpu: "4"
            volumeMounts:
              - name: dshm
                mountPath: /dev/shm
              - name: tpu-plugin
                mountPath: /var/run/tpu-plugin
              - name: models-gcs
                mountPath: /models
        volumes:
          - name: dshm
            emptyDir:
              medium: Memory
              sizeLimit: 240Gi
          - name: tpu-plugin
            hostPath:
              path: /var/run/tpu-plugin
              type: DirectoryOrCreate
          - name: models-gcs
            persistentVolumeClaim:
              claimName: YOUR_MODELS_PVC
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdict-json", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--decode-max-num-seqs", type=int, default=48)
    ap.add_argument("--decode-mnbt", type=int, default=288)
    args = ap.parse_args()

    verdict = json.loads(Path(args.verdict_json).read_text())
    hosts = sorted(verdict["hosts"], key=lambda h: h["worker_id"])
    k8s_names = [h["k8s_hostname"] for h in hosts]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        (
            "llmd-prefill-ep16.yaml",
            render_lws(
                "llmd-prefill-ep16",
                "prefill",
                16,
                k8s_names[0],
                k8s_names[1:4],
                "llmd_p16",
                64,
                8192,
                llmd_enabled=True,
            ),
        ),
        (
            "llmd-decode-ep16.yaml",
            render_lws(
                "llmd-decode-ep16",
                "decode",
                16,
                k8s_names[4],
                k8s_names[5:8],
                "llmd_d16",
                args.decode_max_num_seqs,
                args.decode_mnbt,
                llmd_enabled=True,
            ),
        ),
        (
            "llmd-decode-ep8.yaml",
            render_lws(
                "llmd-decode-ep8",
                "decode",
                8,
                k8s_names[4],
                [k8s_names[5]],
                "llmd_d8",
                args.decode_max_num_seqs,
                args.decode_mnbt,
                llmd_enabled=True,
            ),
        ),
        (
            "colocated-ep16.yaml",
            render_lws(
                "v5p32-colocated-ep16",
                "colocated",
                16,
                k8s_names[0],
                k8s_names[1:4],
                "coloc_ep16",
                args.decode_max_num_seqs,
                max(2048, args.decode_mnbt),
                llmd_enabled=False,
            ),
        ),
        (
            "colocated-ep8.yaml",
            render_lws(
                "v5p32-colocated-ep8",
                "colocated",
                8,
                k8s_names[0],
                [k8s_names[1]],
                "coloc_ep8",
                args.decode_max_num_seqs,
                max(2048, args.decode_mnbt),
                llmd_enabled=False,
            ),
        ),
    ]

    for fname, content in specs:
        (out_dir / fname).write_text(content)
        print(f"Wrote {out_dir / fname}")


if __name__ == "__main__":
    main()
