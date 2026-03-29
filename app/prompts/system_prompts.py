from __future__ import annotations

from typing import Iterable


ROLE_STARTER_PROMPTS = {
    "seller": [
        "Help me improve a listing",
        "How do I manage seller leads?",
        "Show me negotiation tips for sellers",
    ],
    "buyer": [
        "Help me compare marketplace listings",
        "How do I negotiate as a buyer?",
        "What should I verify before checkout?",
    ],
    "default": [
        "Help me prioritize my next step",
        "How can you help on TrustTrade?",
        "What should I review before checkout?",
    ],
}

INTENT_GUIDANCE = {
    "listing": (
        "Make the asset easy to trust at a glance: lead with the asset type, quantity, condition, "
        "business use case, and why it is available now. Add specifics buyers can verify quickly, "
        "like model, age, maintenance status, and pickup or shipping readiness."
    ),
    "negotiation": (
        "Anchor the conversation with business facts instead of a vague price defense. State the asset's "
        "operational value, define your walk-away range, and reply with one concession at a time so the "
        "other side sees a structured counter instead of open-ended flexibility."
    ),
    "pricing": (
        "Frame pricing around replacement cost, condition, urgency, and transaction friction. A strong "
        "TrustTrade price message explains the range, what is included, and what would justify a lower or "
        "higher final number."
    ),
    "dashboard": (
        "Use the dashboard as a triage board: start with live leads, then review listings with weak response, "
        "and only after that move into analytics. That order helps you act on revenue opportunities before "
        "switching to analysis mode."
    ),
    "checkout": (
        "Before checkout, confirm commercial terms in writing: asset scope, quantity, condition, payment path, "
        "delivery responsibility, and any inspection window. Clean deal structure reduces last-minute drop-off."
    ),
    "marketplace": (
        "When browsing the marketplace, compare listings using three filters first: fit for the business need, "
        "credibility of the seller details, and total transaction effort including transport or installation."
    ),
    "profile": (
        "Profile is about identity quality and trust. Keep contact details, role context, and account imagery "
        "up to date so the user looks credible in business transactions."
    ),
    "general": (
        "Answer using the TrustTrade platform data whenever it is relevant. If the platform data does not "
        "cover the request, say so clearly and guide the user toward a supported TrustTrade workflow."
    ),
}

DEFAULT_HELP_TEXT = (
    "A fast way to start is to choose one concrete objective for this session, such as improving a listing, "
    "preparing a buyer response, or reviewing next steps before payment. I can guide that workflow step by step."
)

INTENT_NEXT_STEPS = {
    "listing": "Next step: draft the headline, the top three trust signals, and the commercial terms you want visible.",
    "negotiation": "Next step: send me the other side's last offer and I can help structure your counter.",
    "pricing": "Next step: share the asset type, quantity, and condition so I can help shape a pricing narrative.",
    "dashboard": "Next step: tell me whether you want help with leads, listings, analytics, or checkout.",
    "checkout": "Next step: list the commercial terms you already agreed so I can help spot missing details.",
    "marketplace": "Next step: share the two or three listings you are comparing and I can help score them.",
    "profile": "Next step: tell me whether you want help with account details, password updates, or profile trust signals.",
    "general": "Next step: ask a specific TrustTrade question so I can answer from the platform data.",
}

ROLE_HINTS = {
    "seller": "Open the seller dashboard and work from leads to listings to analytics.",
    "buyer": "Open your buyer dashboard and review active interests, then shortlist the next listing to compare.",
    "admin": "Use the admin routes for operational oversight across users, support, orders, and businesses.",
    "default": "Open your dashboard and choose the workflow you want to improve first.",
}

INTENT_QUICK_REPLIES = {
    "listing": [
        "Write a stronger asset headline",
        "What details should a listing include?",
        "How do I make the listing more trustworthy?",
    ],
    "negotiation": [
        "Draft a counter-offer reply",
        "How do I defend my price?",
        "What questions should I ask before agreeing?",
    ],
    "pricing": [
        "Help me set a pricing range",
        "How should I justify my price?",
        "What lowers deal confidence?",
    ],
    "dashboard": [
        "Guide me through the dashboard",
        "What should I check first?",
        "How do I use analytics better?",
    ],
    "checkout": [
        "Give me a checkout checklist",
        "What terms should be confirmed?",
        "How do I reduce deal risk?",
    ],
    "marketplace": [
        "Help me compare listings",
        "What makes a strong seller profile?",
        "How do I shortlist assets faster?",
    ],
    "profile": [
        "What should I update on my profile?",
        "How does profile improve trust?",
        "How do profile image updates work?",
    ],
}

ROLE_SPECIFIC_GENERAL_QUICK_REPLIES = {
    "seller": ROLE_STARTER_PROMPTS["seller"],
    "buyer": ROLE_STARTER_PROMPTS["buyer"],
    "admin": [
        "What does the admin dashboard contain?",
        "How do admin routes work?",
        "What can admins manage on TrustTrade?",
    ],
}

TOPIC_KEYWORDS = {
    "pricing": ("price", "pricing", "cost", "quote", "margin"),
    "negotiation": ("negotiate", "counter", "offer", "deal", "discount"),
    "seller dashboard": ("seller", "dashboard", "listing", "post asset"),
    "buyer review": ("buyer", "checkout", "compare", "listing"),
    "marketplace": ("marketplace", "browse", "compare", "search", "discover"),
    "profile": ("profile", "account", "avatar", "password", "contact"),
    "support": ("issue", "problem", "support", "help", "bug"),
}

FORMAT_KEYWORDS = {
    "json": ("json",),
    "table": ("table", "tabular"),
    "steps": ("step by step", "steps", "stepwise"),
    "bullets": ("bullet", "bullets", "points", "list"),
    "paragraph": ("paragraph",),
    "short": ("short", "brief", "one line", "summarize", "summary"),
    "detailed": ("detailed", "in detail", "deep", "elaborate", "full details"),
}


def detect_response_format(message: str) -> str:
    lowered = message.lower()

    for style, keywords in FORMAT_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return style

    return "default"


def format_instruction_for(style: str) -> str:
    instructions = {
        "json": "Return the answer as valid JSON inside the reply string.",
        "table": "Return the answer as a markdown table.",
        "steps": "Return the answer as clear numbered steps.",
        "bullets": "Return the answer as concise bullet points.",
        "paragraph": "Return the answer as a paragraph.",
        "short": "Return the answer in a very concise format.",
        "detailed": "Return the answer in a detailed format with clear structure.",
        "default": "Match the format requested by the user. If no format is requested, use a clear concise paragraph.",
    }
    return instructions.get(style, instructions["default"])


def quick_replies_for(intent: str, role: str) -> list[str]:
    if intent in INTENT_QUICK_REPLIES:
        return INTENT_QUICK_REPLIES[intent][:3]

    if role in ROLE_SPECIFIC_GENERAL_QUICK_REPLIES:
        return ROLE_SPECIFIC_GENERAL_QUICK_REPLIES[role][:3]

    return ROLE_STARTER_PROMPTS["default"][:3]


def topic_guidance_for(active_topics: Iterable[str]) -> str:
    topics = [topic for topic in active_topics if topic]
    if not topics:
        return DEFAULT_HELP_TEXT

    guidance = []
    for topic in topics[:3]:
        guidance.append(f"{topic}: {INTENT_GUIDANCE.get(topic, DEFAULT_HELP_TEXT)}")

    return " | ".join(guidance)
