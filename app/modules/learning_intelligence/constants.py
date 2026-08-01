"""Learning Intelligence Agent (LIA) constants."""

from __future__ import annotations

# Event types (append-only learning_events.event_type)
EVENT_CHAT_USER_QUESTION = "chat_user_question"
EVENT_CHAT_ASSISTANT_RESPONSE = "chat_assistant_response"
EVENT_VOICE_USER_UTTERANCE = "voice_user_utterance"
EVENT_VOICE_ASSISTANT_RESPONSE = "voice_assistant_response"
EVENT_QUIZ_ANSWER = "quiz_answer"
EVENT_ASSIGNMENT_SUBMITTED = "assignment_submitted"
EVENT_CHAPTER_COMPLETED = "chapter_completed"
EVENT_STUDY_SESSION_END = "study_session_end"
EVENT_HINT_USED = "hint_used"
EVENT_TOPIC_SWITCH = "topic_switch"

# Bloom understanding levels 1–6
UNDERSTANDING_REMEMBER = 1
UNDERSTANDING_UNDERSTAND = 2
UNDERSTANDING_APPLY = 3
UNDERSTANDING_ANALYZE = 4
UNDERSTANDING_EVALUATE = 5
UNDERSTANDING_CREATE = 6

# Learning style keys (inferred from behaviour)
STYLE_VISUAL = "visual"
STYLE_CONVERSATION = "conversation"
STYLE_EXPERIMENT = "experiment"
STYLE_READING = "reading"
STYLE_PRACTICE = "practice"
STYLE_STORY = "story"
STYLE_EXAMPLE = "example"
STYLE_ANIMATION = "animation"

# Risk levels
RISK_NOT_STARTED = "not_started"
RISK_LOW = "low"
RISK_MEDIUM = "medium"
RISK_HIGH = "high"

# Metric history types
METRIC_CONFIDENCE = "confidence"
METRIC_ATTENTION = "attention"
METRIC_ENGAGEMENT = "engagement"
METRIC_PERFORMANCE = "performance"

# Summary periods
PERIOD_WEEKLY = "weekly"
PERIOD_MONTHLY = "monthly"

# Mastery thresholds
MASTERY_STRONG = 0.75
MASTERY_WEAK = 0.45
GAP_THRESHOLD = 0.5

# Guidance cache TTL (seconds)
GUIDANCE_CACHE_TTL = 300

# Celery queue
QUEUE_LIA = "lia"
