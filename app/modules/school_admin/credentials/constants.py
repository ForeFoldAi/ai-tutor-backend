from __future__ import annotations

DEFAULT_PAGE_LIMIT = 10
MAX_PAGE_LIMIT = 500
PAGE_SIZE_OPTIONS = [10, 20, 30, 40, 50, 100, 200, 500]

DELIVERY_NOT_SENT = "not_sent"
DELIVERY_IN_PROCESS = "in_process"
DELIVERY_FAILED = "failed"
DELIVERY_EMAIL_SENT = "email_sent"
DELIVERY_SMS_SENT = "sms_sent"
# Legacy (pre-failed status); still mapped for old rows
DELIVERY_PENDING = "pending"

DELIVERY_LABELS = {
    DELIVERY_NOT_SENT: "Not Sent",
    DELIVERY_IN_PROCESS: "In Process",
    DELIVERY_FAILED: "Failed",
    DELIVERY_EMAIL_SENT: "Email Sent",
    DELIVERY_SMS_SENT: "SMS Sent",
    DELIVERY_PENDING: "Failed",  # treat old pending as Failed in the UI
}

FIRST_LOGIN_COMPLETED = "Completed"
FIRST_LOGIN_PENDING = "Pending"
FIRST_LOGIN_NOT_STARTED = "Not Started"

ROLE_TEACHER = "Teacher"
ROLE_STUDENT = "Student"
