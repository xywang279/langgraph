from __future__ import annotations

import os
from typing import Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
    api_auth_token: Optional[str] = Field(default=None, env="API_AUTH_TOKEN")
    auth_secret_key: Optional[str] = Field(default=None, env="AUTH_SECRET_KEY")
    celery_broker_url: Optional[str] = Field(default=None, env="CELERY_BROKER_URL")
    s3_endpoint_url: Optional[str] = Field(default=None, env="S3_ENDPOINT_URL")

    class Config:
        case_sensitive = False

    @model_validator(mode="after")
    def _ensure_auth(self):
        token = (self.api_auth_token or "").strip()
        secret = (self.auth_secret_key or "").strip()
        if not token and not secret:
            raise ValueError("Must configure API_AUTH_TOKEN or AUTH_SECRET_KEY for production.")
        return self

    @field_validator("celery_broker_url")
    def _validate_celery(cls, v):
        if not v:
            return v
        if not (v.startswith("redis://") or v.startswith("amqp://") or v.startswith("sqs://") or v.startswith("rediss://")):
            raise ValueError("CELERY_BROKER_URL must be redis://, amqp://, sqs:// or rediss://")
        return v


def load_settings() -> AppSettings:
    return AppSettings()
