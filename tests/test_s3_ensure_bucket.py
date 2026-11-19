"""Self-check: S3 create_bucket kwargs (no network)."""

from app.services.image_service.storage_backend import create_bucket_kwargs


def test_create_bucket_kwargs_minio_bare():
    kw = create_bucket_kwargs("ai-tutor", region="ap-south-1", endpoint_url="https://files.forefoldai.com")
    assert kw == {"Bucket": "ai-tutor"}


def test_create_bucket_kwargs_aws_region():
    kw = create_bucket_kwargs("ai-tutor", region="ap-south-1", endpoint_url=None)
    assert kw["Bucket"] == "ai-tutor"
    assert kw["CreateBucketConfiguration"]["LocationConstraint"] == "ap-south-1"


def test_create_bucket_kwargs_us_east_1():
    kw = create_bucket_kwargs("ai-tutor", region="us-east-1", endpoint_url=None)
    assert kw == {"Bucket": "ai-tutor"}
