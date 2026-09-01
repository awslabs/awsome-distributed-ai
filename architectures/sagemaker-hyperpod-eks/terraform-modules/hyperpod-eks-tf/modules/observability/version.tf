# keep to force terraform to use non hashicorp/* provider
terraform {
  required_providers {
    grafana = {
      source = "grafana/grafana"
    }
  }
}
