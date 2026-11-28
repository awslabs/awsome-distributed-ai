{{/* Pod labels; llm-d.ai/guide is the InferencePool selector */}}
{{- define "dsv4flash.labels" -}}
llm-d.ai/guide: {{ .Values.guide }}
llm-d.ai/role: decode
llm-d.ai/model: {{ .Values.model | splitList "/" | last }}
llm-d.ai/engine-type: sglang
{{- end }}

{{- define "dsv4flash.nvmeHostPath" -}}
{{- required "set nvmeHostPath to the node NVMe path holding huggingface/hub" .Values.nvmeHostPath -}}
{{- end }}

{{- define "dsv4flash.scheduling" -}}
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
      - matchExpressions:
        - key: node.kubernetes.io/instance-type
          operator: In
          values: {{ toJson .Values.instanceTypes }}
tolerations: [{key: nvidia.com/gpu, operator: Exists, effect: NoSchedule}]
{{- end }}
