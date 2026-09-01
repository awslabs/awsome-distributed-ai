ENROOT_IMAGE   ?= deepspeed
APPS_PATH      ?= /fsx/apps
SQUASH_FILE    ?= $(APPS_PATH)/$(ENROOT_IMAGE).sqsh
PARTITION      ?= dev
NODES          ?= 8
LOGS_DIR       ?= logs
RESULTS_DIR    ?= sweep_results

.PHONY: all build clean import build-remote train parse help

all: build import

help:
	@echo "Container targets:"
	@echo "  build          - Build Docker image locally"
	@echo "  import         - Convert Docker image to Enroot squash file"
	@echo "  build-remote   - Build image on a compute node via sbatch"
	@echo "  clean          - Remove local squash file"
	@echo ""
	@echo "Training targets:"
	@echo "  train          - Submit 103B GPT pretraining (best config: TP=8, PP=8, fusions)"
	@echo ""
	@echo "Results targets:"
	@echo "  parse          - Parse training logs into benchmark JSON"

# ---- Container ----

build:
	docker build -t $(ENROOT_IMAGE) -f 0.deepspeed.dockerfile .

import:
	mkdir -p $(APPS_PATH)
	enroot import -o $(SQUASH_FILE) dockerd://$(ENROOT_IMAGE):latest

build-remote:
	sbatch 1.build-image.sbatch

clean:
	-rm -f $(SQUASH_FILE)

# ---- Training (best config: TP=8, PP=8, ZeRO=0, fusions enabled) ----

train:
	sbatch --partition=$(PARTITION) --nodes=$(NODES) \
		--export=ALL,TP=8,PP=8,ZERO_STAGE=0,ENABLE_FUSIONS=1,CONFIG_NAME=best_fused_tp8_pp8 \
		gpt/slurm/pretrain_gpt_103b.sbatch

# ---- Results ----

parse:
	python3 gpt/parse_results.py --jobs-csv $(RESULTS_DIR)/sweep_jobs.csv \
		--logs-dir $(LOGS_DIR) --output-dir $(RESULTS_DIR)
