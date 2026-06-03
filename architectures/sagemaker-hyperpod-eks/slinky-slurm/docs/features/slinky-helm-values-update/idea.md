---
id: slinky-helm-values-update
name: Slinky Slurm Helm Values Syntax Update
type: Enhancement
priority: P1
effort: Medium
impact: High
created: 2026-03-06
---

# Slinky Slurm Helm Values Syntax Update

## Problem Statement

The `g5/g5-values.yaml` and `p5/p5-values.yaml` Helm values files use an older
Slinky Slurm chart API structure. The upstream chart has evolved and the current
files need to be updated to reflect the new syntax. A reference template using
the new syntax exists at
`ai-on-eks-slurm/ai-on-eks/blueprints/training/slinky-slurm/slurm-values.yaml.template`.

## Proposed Solution

Update both `g5/g5-values.yaml` and `p5/p5-values.yaml` to align with the new
Slinky chart structure. Key structural changes identified:

### 1. Controller section simplified

The new syntax removes most boilerplate from the `controller:` section, keeping
only persistence and pod-level overrides:

```yaml
# New (simplified)
controller:
  persistence:
    storageClassName: gp3
  podSpec:
    affinity: *commonAffinity
  metrics:
    enabled: true
    serviceMonitor:
      enabled: true
```

### 2. Login node: `login:` replaced by `loginsets:`

The single `login:` top-level key is replaced by a map-based `loginsets:` key
with named login sets, each containing nested `login:`, `podSpec:`, and
`service:` blocks:

```yaml
# Old
login:
  enabled: true
  replicas: 1
  service:
    type: LoadBalancer
  extraVolumeMounts: [...]
  extraVolumes: [...]
  affinity: *commonAffinity

# New
loginsets:
  slinky:
    enabled: true
    replicas: 1
    login:
      volumeMounts: [...]
    rootSshAuthorizedKeys: |
      <key>
    extraSshdConfig: |
      ...
    podSpec:
      affinity: *commonAffinity
      volumes: [...]
    service:
      spec:
        type: LoadBalancer
      port: 22
```

### 3. Compute: `compute:` replaced by `nodesets:`

The `compute:` top-level key with nested `nodesets:` list is replaced by a
top-level `nodesets:` map with named node sets:

```yaml
# Old
compute:
  image:
    repository: ghcr.io/slinkyproject/slurmd
    tag: 25.05-ubuntu24.04
  nodesets:
    - name: hp-node
      replicas: 4
      image: ...
      nodeSelector: ...
      resources: ...
      extraVolumeMounts: [...]
      extraVolumes: [...]
      partition: ...

# New
nodesets:
  slinky:
    enabled: true
    replicas: 4
    slurmd:
      image:
        repository: ...
        tag: ...
      resources: ...
      volumeMounts: [...]
    podSpec:
      nodeSelector: ...
      tolerations: [...]
      volumes: [...]
    extraConfMap:
      Gres: ["gpu:..."]
```

### 4. Accounting simplified

The `accounting:` section uses a nested `podSpec:` key instead of flat
`affinity:`/`nodeSelector:` keys:

```yaml
# New
accounting:
  enabled: true
  podSpec:
    affinity: *commonAffinity
```

### 5. REST API simplified

Same pattern -- flat keys replaced by `podSpec:`:

```yaml
# New
restapi:
  podSpec:
    affinity: *commonAffinity
```

### 6. New top-level sections

- `vendor.nvidia.dcgm.enabled: true` -- DCGM GPU-to-job mapping
- Top-level `configFiles:` for `gres.conf` (moved out of `slurm.configFiles:`)

### 7. Removed/consolidated sections

The following verbose sections in the old syntax are removed or folded into the
simplified structure above:

- `debug:`, `nameOverride`, `fullnameOverride`, `namespaceOverride`
- `imagePullSecrets`, `imagePullPolicy` (global)
- `priorityClassName` (global)
- `jwt:`, `slurm.auth:` (default generation is sufficient)
- `slurm.extraSlurmdbdConf`, `slurm.extraSlurmConf` (now per-section)
- `slurm.prologScripts`, `slurm.epilogScripts` (now per-nodeset)
- `authcred:` full section (defaults are sufficient)
- `slurm-exporter:` full section (kept only if metrics needed)
- `mariadb:` full configuration block (simplified or uses defaults)
- Per-nodeset `updateStrategy`, `persistentVolumeClaimRetentionPolicy`,
  `volumeClaimTemplates` (moved to defaults or removed)

### Key References

- **New syntax template:**
  `ai-on-eks-slurm/ai-on-eks/blueprints/training/slinky-slurm/slurm-values.yaml.template`
- **Existing files to update:**
  `g5/g5-values.yaml`, `p5/p5-values.yaml`

### Intentional g5/p5 differences to preserve

- `commonAffinity` instance type (`ml.m5.2xlarge` for both currently)
- `nodeSelector` instance types (`ml.g5.8xlarge` vs `ml.p5.48xlarge`)
- GPU/EFA resource counts (g5: 1 GPU, 0 EFA; p5: 4 GPU, 16 EFA)
- GRES configuration (`gpu:a10g:1` vs appropriate p5 config)

## Success Criteria

- [ ] TBD

## Notes

Created via feature-capture
