{{/* Labels on every resource. */}}
{{- define "ai.labels" -}}
app.kubernetes.io/part-of: shopnexus
shopnexus.id.vn/stack: {{ required "stack is required" .Values.stack }}
{{- end }}

{{/* Pod labels. Call with (dict "root" $ "name" "ai-gateway"). */}}
{{- define "ai.podLabels" -}}
app.kubernetes.io/name: {{ .name }}
{{ include "ai.labels" .root }}
{{- end }}

{{/* repository:tag of one images.* entry. */}}
{{- define "ai.image" -}}
{{ .repository }}:{{ .tag }}
{{- end }}

{{/* Argo CD sync wave annotation. */}}
{{- define "ai.wave" -}}
argocd.argoproj.io/sync-wave: {{ . | quote }}
{{- end }}

{{/* Prefer nodes labelled role=<nodeRole>. Goes under affinity. */}}
{{- define "ai.preferRole" -}}
nodeAffinity:
  preferredDuringSchedulingIgnoredDuringExecution:
    - weight: 100
      preference:
        matchExpressions:
          - { key: role, operator: In, values: [{{ required "nodeRole is required" .Values.nodeRole }}] }
{{- end }}

{{/* The env var of the key Job that holds one key's value: server-llm gives
     KEY_SERVER_LLM. */}}
{{- define "ai.keyEnv" -}}
KEY_{{ . | upper | replace "-" "_" }}
{{- end }}

{{/* The JSON body of /key/update and /key/generate for one key, without the
     key's value, which the Job adds at run time. Call with
     (dict "alias" "server-llm" "k" .Values.gateway.keys.server-llm). */}}
{{- define "ai.keyBody" -}}
{{- $k := .k -}}
{{- $units := required (printf "gateway.keys.%s.dailyUnits is required" .alias) $k.dailyUnits -}}
{{- $b := dict "key_alias" .alias "models" $k.models "budget_duration" "1d" "max_budget" (divf $units 1000000) -}}
{{- if hasKey $k "tpmLimit" }}{{ $_ := set $b "tpm_limit" (int $k.tpmLimit) }}{{ end -}}
{{- if hasKey $k "rpmLimit" }}{{ $_ := set $b "rpm_limit" (int $k.rpmLimit) }}{{ end -}}
{{- if not (or (hasKey $k "tpmLimit") (hasKey $k "rpmLimit")) }}{{ fail (printf "gateway.keys.%s needs tpmLimit or rpmLimit" .alias) }}{{ end -}}
{{- toJson $b -}}
{{- end }}

{{/* LiteLLM's config.yaml: one model_list entry per gateway.aliases, priced so
     budgets count tokens, seconds or dollars. */}}
{{- define "ai.gatewayConfig" -}}
model_list:
{{- range $name, $a := .Values.gateway.aliases }}
  - model_name: {{ $name }}
    litellm_params:
      model: {{ required (printf "gateway.aliases.%s.model is required" $name) $a.model }}
      api_base: {{ required (printf "gateway.aliases.%s.apiBase is required" $name) $a.apiBase }}
      {{- if $a.apiKeySecret }}
      api_key: os.environ/{{ $a.apiKeySecret }}
      {{- else }}
      api_key: none   # ai-models, which has no authentication
      {{- end }}
      {{- with $a.extraBody }}
      extra_body: {{ toJson . }}
      {{- end }}
    model_info:
      {{- with $a.mode }}
      mode: {{ . }}
      {{- end }}
      {{- if eq $a.price "token" }}
      input_cost_per_token: 0.000001
      output_cost_per_token: 0.000001
      {{- else if eq $a.price "second" }}
      input_cost_per_second: 0.000001
      {{- else if eq $a.price "usd" }}
      input_cost_per_token: {{ required (printf "gateway.aliases.%s.inputCostPerToken is required" $name) $a.inputCostPerToken }}
      output_cost_per_token: {{ required (printf "gateway.aliases.%s.outputCostPerToken is required" $name) $a.outputCostPerToken }}
      {{- else }}
      {{- fail (printf "gateway.aliases.%s.price must be token, second or usd" $name) }}
      {{- end }}
{{- end }}
general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  database_url: os.environ/DATABASE_URL
  # A search photo passes through here; the spend logs keep no bodies, and
  # they are kept 30 days.
  store_prompts_in_spend_logs: false
  maximum_spend_logs_retention_period: 30d
litellm_settings:
  drop_params: true
{{- end }}

{{/* An env file of env/ (KEY=value lines, # comments) as a JSON object.
     The umbrella's docker-compose.yml reads the same files with env_file. */}}
{{- define "ai.envFile" -}}
{{- $out := dict }}
{{- range splitList "\n" (.root.Files.Get .file) }}
{{- $line := trim . }}
{{- if and $line (not (hasPrefix "#" $line)) }}
{{- $kv := splitn "=" 2 $line }}
{{- if lt (len $kv) 2 }}{{ fail (printf "%s: %q is not KEY=value" $.file $line) }}{{ end }}
{{- $_ := set $out $kv._0 $kv._1 }}
{{- end }}
{{- end }}
{{- toJson $out }}
{{- end }}
