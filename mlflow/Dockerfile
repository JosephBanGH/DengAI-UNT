# Servidor MLflow con cliente S3 para artefactos en MinIO.
FROM ghcr.io/mlflow/mlflow:v3.1.4

# El backend de artefactos usa S3 (MinIO); boto3 no viene en la imagen base.
RUN pip install --no-cache-dir "boto3>=1.35,<2"

EXPOSE 5000
