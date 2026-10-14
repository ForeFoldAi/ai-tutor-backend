"""OTP reset flow self-check (no DB / SMTP)."""

from __future__ import annotations

from app.modules.auth.schemas import (
    ForgotPasswordLookupResponse,
    ForgotPasswordRequest,
    ResetPasswordRequest,
    VerifyResetOtpRequest,
    VerifyResetOtpResponse,
)


def main() -> None:
    verify = VerifyResetOtpRequest(email="anita@example.com", user_id=42, otp="482913")
    assert verify.otp == "482913"
    assert "reset_token" in VerifyResetOtpResponse.model_fields

    req = ResetPasswordRequest(reset_token="dummy.jwt.token", new_password="NewPassw0rd")
    assert "otp" not in ResetPasswordRequest.model_fields
    assert "reset_token" in ResetPasswordRequest.model_fields

    send = ForgotPasswordRequest(email="anita@example.com", user_id=42)
    assert send.user_id == 42

    empty = ForgotPasswordLookupResponse(accounts=[])
    assert empty.accounts == []
    print("auth otp schema selfcheck ok")


if __name__ == "__main__":
    main()
