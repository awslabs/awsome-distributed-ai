# HyperPod x AWS DevOps Agent
#
# Single-template deployment. Everything (IAM, Agent Space, EKS access, webhook
# bridge, periodic audit, email notifier, skills) is deployed by one
# CloudFormation stack per cluster, driven by deploy/deploy.sh. See README.md.
#
# Quick start:
#   cp deploy/params.example.json deploy/params.json   # edit cluster + email
#   make deploy
#
# Per-cluster params: PARAMS_FILE=deploy/params.<cluster>.json make deploy

PYTHON3 ?= python3

DEPLOY_DIR   := deploy
PARAMS_FILE  ?= $(DEPLOY_DIR)/params.json

# Per-cluster slug + stack name, derived from params.json so multiple clusters
# coexist. Override STACK_NAME to target a specific stack. Empty when params.json
# is absent (e.g. before first deploy) — the deploy/teardown scripts recompute it.
NAME_PREFIX  := $(shell $(PYTHON3) -c "import json,re,sys; \
n=json.load(open('$(PARAMS_FILE)')).get('HyperPodClusterName','') if __import__('os').path.exists('$(PARAMS_FILE)') else ''; \
s=re.sub(r'-{2,}','-',re.sub(r'[^a-z0-9]+','-',n.lower()).strip('-'))[:20].strip('-'); \
print(s or 'cluster')" 2>/dev/null)
STACK_NAME   ?= hyperpod-devops-agent-$(NAME_PREFIX)

.PHONY: help \
        deploy update teardown-stack \
        sync-skills import-upstream-skills \
        stack-status stack-outputs \
        bridge-logs audit-logs email-logs audit-test

help:
	@echo "Single-stack deployment (deploy/hyperpod_devops_agent.template.yaml):"
	@echo "  deploy        - One command: sync skills, embed Lambdas, deploy the whole stack."
	@echo "                  Reads deploy/params.json (copy from params.example.json)."
	@echo "                  Per-cluster: PARAMS_FILE=deploy/params.<cluster>.json make deploy"
	@echo "  update        - Alias for deploy (idempotent; re-uploads changed skills)."
	@echo "  teardown-stack- Delete the stack + assets bucket (custom resources self-clean first)."
	@echo
	@echo "Skills:"
	@echo "  sync-skills            - Zip skills + mental-model doc, upload to the S3 assets bucket."
	@echo "                           (Uploads only; unlike 'make deploy' it does not rebuild the"
	@echo "                           skill-uploader Lambda or apply the stack. Prefer 'make deploy'.)"
	@echo "  import-upstream-skills - Stage curated awslabs/agent-plugins hyperpod-* skills into"
	@echo "                           skills/ (then 'make deploy' uploads them)."
	@echo
	@echo "Observe:"
	@echo "  stack-status  - Show the stack status."
	@echo "  stack-outputs - Show the stack outputs (console URL, secret ARN, ...)."
	@echo "  bridge-logs / audit-logs / email-logs - Tail a Lambda's logs."
	@echo "  audit-test    - Manually invoke the periodic-audit Lambda once."

# ------------------------------------------------------------ deploy / teardown

deploy:
	PARAMS_FILE=$(PARAMS_FILE) STACK_NAME=$(STACK_NAME) bash $(DEPLOY_DIR)/deploy.sh

update: deploy

teardown-stack:
	PARAMS_FILE=$(PARAMS_FILE) STACK_NAME=$(STACK_NAME) bash $(DEPLOY_DIR)/teardown.sh

# ------------------------------------------------------------ skills

# Upload skills to S3 only. Does NOT rebuild the skill-uploader Lambda or apply
# the stack (so a lone 'make sync-skills' can leave the uploader Lambda stale);
# 'make deploy' does the full path and is preferred.
sync-skills:
	@bash -c 'set -euo pipefail; \
		REGION=$${REGION:-$$(aws configure get region)}; \
		ACCOUNT_ID=$$(aws sts get-caller-identity --query Account --output text); \
		BUCKET=hpda-assets-$(NAME_PREFIX)-$$ACCOUNT_ID-$$REGION; \
		if ! aws s3api head-bucket --bucket $$BUCKET --region $$REGION 2>/dev/null; then \
			if [ "$$REGION" = "us-east-1" ]; then \
				aws s3api create-bucket --bucket $$BUCKET --region $$REGION >/dev/null; \
			else \
				aws s3api create-bucket --bucket $$BUCKET --region $$REGION \
					--create-bucket-configuration LocationConstraint=$$REGION >/dev/null; \
			fi; \
		fi; \
		$(PYTHON3) $(DEPLOY_DIR)/prepare_deployment.py sync-skills --bucket $$BUCKET --region $$REGION'

import-upstream-skills:
	bash $(DEPLOY_DIR)/import_upstream_skills.sh

# ------------------------------------------------------------ observe

stack-status:
	@bash -c 'REGION=$${REGION:-$$(aws configure get region)}; \
		aws cloudformation describe-stacks --region $$REGION --stack-name $(STACK_NAME) \
			--query "Stacks[0].StackStatus" --output text'

stack-outputs:
	@bash -c 'REGION=$${REGION:-$$(aws configure get region)}; \
		aws cloudformation describe-stacks --region $$REGION --stack-name $(STACK_NAME) \
			--query "Stacks[0].Outputs" --output table --no-cli-pager'

# Tail a Lambda's logs by reading its function name from a stack output.
# $(1) = output key
define TAIL_STACK_LAMBDA
	@bash -c 'REGION=$${REGION:-$$(aws configure get region)}; \
		FUNC=$$(aws cloudformation describe-stacks --region $$REGION --stack-name $(STACK_NAME) \
			--query "Stacks[0].Outputs[?OutputKey=='"'"'$(1)'"'"'].OutputValue | [0]" --output text); \
		if [ -z "$$FUNC" ] || [ "$$FUNC" = "None" ]; then echo "No $(1) in stack outputs."; exit 1; fi; \
		aws logs tail /aws/lambda/$$FUNC --region $$REGION --follow --since 1h'
endef

bridge-logs:
	$(call TAIL_STACK_LAMBDA,WebhookBridgeFunctionName)

audit-logs:
	$(call TAIL_STACK_LAMBDA,PeriodicAuditFunctionName)

email-logs:
	$(call TAIL_STACK_LAMBDA,EmailNotifierFunctionName)

audit-test:
	@bash -c 'REGION=$${REGION:-$$(aws configure get region)}; \
		FUNC=$$(aws cloudformation describe-stacks --region $$REGION --stack-name $(STACK_NAME) \
			--query "Stacks[0].Outputs[?OutputKey=='"'"'PeriodicAuditFunctionName'"'"'].OutputValue | [0]" --output text); \
		if [ -z "$$FUNC" ] || [ "$$FUNC" = "None" ]; then echo "Periodic audit not enabled in this stack."; exit 1; fi; \
		echo "Invoking $$FUNC..."; \
		aws lambda invoke --region $$REGION --function-name $$FUNC --payload "{\"trigger\":\"periodic\"}" /tmp/audit-test-output.json \
			--cli-binary-format raw-in-base64-out >/dev/null && cat /tmp/audit-test-output.json && echo'
