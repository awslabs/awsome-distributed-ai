# Supermarket Shelves Dataset

## Overview

This test case uses the **Supermarket Shelves** object detection dataset from
[Humans in the Loop](https://humansintheloop.org/). The dataset contains images
of supermarket shelves annotated with two object classes:

- **Price** - Price tags and labels
- **Product** - Products on shelves

### Statistics

| Property | Value |
|----------|-------|
| Images | 45 |
| Classes | 2 (Price, Product) |
| Total annotations | ~11,743 bounding boxes |
| Annotation format | Supervisely JSON |
| License | CC0 1.0 (Public Domain) |
| Source | Humans in the Loop + Techfugees Lebanon |

## Download Instructions

1. Visit the dataset page:
   **https://humansintheloop.org/resources/datasets/supermarket-shelves-dataset/**

2. Fill in the required fields (name, email, company) and submit to download
   the dataset archive.

3. Extract the archive. You should get the following structure:

```
Supermarket shelves/
├── images/
│   ├── 001.jpg
│   ├── 002.jpg
│   └── ... (45 images)
└── annotations/
    ├── 001.jpg.json
    ├── 002.jpg.json
    └── ... (45 JSON files)
```

## Prepare the Dataset

The training script expects the following directory layout on the shared
filesystem (FSx for Lustre):

```
/fsx/data/
├── meta.json
└── Supermarket shelves/
    ├── images/
    │   ├── 001.jpg
    │   ├── 002.jpg
    │   └── ...
    └── annotations/
        ├── 001.jpg.json
        ├── 002.jpg.json
        └── ...
```

### Create `meta.json`

The `meta.json` file defines the class mapping. Create it with the following
content:

```json
{
    "classes": [
        {
            "title": "Price",
            "shape": "rectangle",
            "color": "#FF0000",
            "id": 10213293
        },
        {
            "title": "Product",
            "shape": "rectangle",
            "color": "#00FF00",
            "id": 10213294
        }
    ]
}
```

### Upload to FSx

If using Kubernetes, you can copy data into the FSx volume:

```bash
# Create a temporary pod with FSx access
kubectl run data-upload --image=ubuntu:22.04 --restart=Never \
    --overrides='{"spec":{"volumes":[{"name":"fsx","persistentVolumeClaim":{"claimName":"fsx-pvc"}}],"containers":[{"name":"data-upload","image":"ubuntu:22.04","command":["sleep","3600"],"volumeMounts":[{"name":"fsx","mountPath":"/fsx"}]}]}}'

# Wait for pod to be ready
kubectl wait --for=condition=Ready pod/data-upload --timeout=120s

# Copy dataset
kubectl cp /path/to/meta.json default/data-upload:/fsx/data/meta.json
kubectl cp "/path/to/Supermarket shelves" "default/data-upload:/fsx/data/Supermarket shelves"

# Verify
kubectl exec data-upload -- ls -la /fsx/data/
kubectl exec data-upload -- ls -la "/fsx/data/Supermarket shelves/images/" | head

# Clean up
kubectl delete pod data-upload
```

## Verification

After uploading, verify the data is correct:

```bash
# Check file counts
kubectl exec <training-pod> -- bash -c 'ls /fsx/data/Supermarket\ shelves/images/ | wc -l'
# Expected: 45

kubectl exec <training-pod> -- bash -c 'ls /fsx/data/Supermarket\ shelves/annotations/ | wc -l'
# Expected: 45

# Check meta.json
kubectl exec <training-pod> -- cat /fsx/data/meta.json
```

## Annotation Format

Each annotation file is a Supervisely-format JSON with the following structure:

```json
{
    "description": "",
    "tags": [],
    "size": {
        "height": 1536,
        "width": 2048
    },
    "objects": [
        {
            "classTitle": "Product",
            "points": {
                "exterior": [[x1, y1], [x2, y2]],
                "interior": []
            }
        }
    ]
}
```

The training script automatically:
- Reads all annotation files
- Maps class titles ("Price", "Product") to numeric indices
- Normalizes bounding boxes to [0, 1]
- Converts to DETR format [center_x, center_y, width, height]
- Splits into 80% train / 20% validation
