"""Constants for Beaker launch configuration."""

# Infrastructure environment variables for Beaker jobs
# These configure olmo-eval's InfrastructureConfig when running in Beaker
BEAKER_INFRA_ENV_VARS = {
    "OLMO_CONTAINER_RUNTIME": "podman",
    "SWEREX_REGISTRY": "us-east1-docker.pkg.dev/ai2-allennlp/olmo-eval",
    "OLMO_PASTA_HOST_IP": "169.254.1.2",
    "OLMO_RESULT_DIR": "/results",
}
