{{- define "github-pages.name" -}}
{{- .Chart.Name -}}
{{- end -}}

{{- define "github-pages.labels" -}}
app: {{ include "github-pages.name" . }}
{{- end -}}
