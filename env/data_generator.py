"""
Synthetic email data generator for Email Triage OpenEnv.
Generates realistic professional emails across 5 priority categories.
No external dependencies.
"""
import random
import uuid
from datetime import datetime, timedelta
from env.models import Email

SENDERS = [
    ("ceo@company.com",          "Sarah Chen (CEO)"),
    ("client@bigcorp.com",       "James Miller (BigCorp Client)"),
    ("newsletter@techdigest.com","Tech Digest Newsletter"),
    ("hr@company.com",           "HR Department"),
    ("alerts@github.com",        "GitHub Notifications"),
    ("campaigns@marketing.io",   "Marketing Team"),
    ("partner@enterprise.com",   "Enterprise Partner"),
    ("ops@company.com",          "Operations Team"),
    ("support@vendor.com",       "Vendor Support"),
    ("noreply@deals.spam.com",   "Special Offers"),
    ("cto@company.com",          "Alex Kumar (CTO)"),
    ("finance@company.com",      "Finance Department"),
]

TEMPLATES = {
    "urgent": [
        (
            "URGENT: Production server down - immediate action needed",
            "Our main production server has been returning 500 errors for the last 15 minutes. "
            "All customer-facing services are affected. Revenue impact is approximately $5,000 "
            "per minute. The on-call engineer is unreachable. Please authorize emergency "
            "escalation to our cloud provider immediately. I need your authorization NOW.",
        ),
        (
            "CRITICAL: Data breach detected - legal response required within 30 minutes",
            "Our security monitoring flagged unusual data access at 03:14 UTC. Approximately "
            "2,400 customer records may have been exposed. Legal requires your sign-off on the "
            "incident response plan before we can notify affected customers per GDPR. "
            "The 72-hour regulatory clock started 2 hours ago. Please respond immediately.",
        ),
        (
            "URGENT: Board meeting moved to 2pm - need exec summary in 45 minutes",
            "The board has moved today's quarterly review from 4pm to 2pm. They are specifically "
            "asking for Q3 revenue breakdown and updated product roadmap. Finance has the numbers "
            "but needs your approval on the commentary before sharing with the board. "
            "We have 45 minutes. Please confirm you received this.",
        ),
        (
            "CRITICAL: Payment gateway completely down - $12k/hour revenue loss",
            "Stripe integration started failing at 09:42. All checkout flows are broken. "
            "We have lost 340 failed transactions worth approximately $12,000 in the last hour. "
            "Engineering needs the Stripe dashboard admin credentials to diagnose the issue - "
            "only you have access to those credentials. Please send immediately.",
        ),
    ],
    "high": [
        (
            "Q3 budget approval required by end of day today",
            "Hi, the Q3 departmental budgets are due to finance by 5pm today. I have attached "
            "the spreadsheet with all team allocations. The engineering request is $180k over "
            "the original estimate due to new security infrastructure we discussed last month. "
            "Please review and send back with your approval or any changes so I can submit on time.",
        ),
        (
            "Top candidate offer expiring Friday - need approval to increase offer",
            "We have an offer out to Priya Sharma for the Senior ML Engineer role. She has a "
            "competing offer from Google that expires this Friday. Our recruiter strongly "
            "recommends increasing our base offer by $15k to close her. This is a rare candidate "
            "with exactly the skills we need. Please confirm so we don't lose her.",
        ),
        (
            "Client escalation: Acme Corp threatening to cancel $240k contract",
            "David from Acme Corp called this morning very unhappy about last week's SLA breach. "
            "Their $240k annual contract is up for renewal in 6 weeks. He is asking for a "
            "30-minute call with leadership this week and a written root cause analysis. "
            "Can you find time Thursday or Friday? I can set it up immediately.",
        ),
        (
            "Investor due diligence - documents needed by Friday EOD",
            "Sequoia's team sent over their standard due diligence checklist. They need the "
            "cap table, last 3 years of financials, and three customer references by Friday EOD. "
            "Legal has everything except the references. We need your approval to share "
            "contact details for our three design partners. Please respond today.",
        ),
    ],
    "normal": [
        (
            "Weekly engineering standup notes - Jan 15",
            "Hi team, here are this week's standup notes. All sprints are on track. "
            "The auth service refactor is 80% complete, targeting merge by next Wednesday. "
            "The mobile team shipped the new onboarding flow to staging yesterday. "
            "No blockers reported. Next standup Monday at 10am. Let me know if I missed anything.",
        ),
        (
            "Project Phoenix Phase 2 kickoff scheduled for Jan 22",
            "Happy to share that Phase 1 sign-off has been received from the client. "
            "Phase 2 kickoff is scheduled for January 22nd at 2pm in the main conference room. "
            "Please send your team's availability for the afternoon workshop to Sarah by Thursday. "
            "Full project brief will be circulated 48 hours before the meeting.",
        ),
        (
            "Monthly metrics report - December 2023",
            "Attached is the monthly metrics dashboard for December. MAU grew 12% month-over-month "
            "to 48,200 active users. Churn remained flat at 2.1%. NPS improved from 42 to 47. "
            "Revenue was $1.24M, slightly above forecast. Full segment breakdown in the attachment. "
            "Happy to walk through the numbers at your convenience.",
        ),
        (
            "Office lease renewal options - response needed before March",
            "Our current lease expires March 31st. The landlord has offered three options: "
            "1-year renewal at current rate, 3-year with 5% increase, or 5-year with 8% increase. "
            "Property management also asked whether we need additional floor space. "
            "No immediate rush but would appreciate your preference before February.",
        ),
    ],
    "low": [
        (
            "Friendly reminder: please clean up the office kitchen",
            "Hi everyone, just a friendly reminder to please wash your dishes and clean up "
            "any spills in the kitchen. The cleaning crew comes on Wednesdays and Fridays only. "
            "Also, anything left in the fridge unlabeled will be cleared out every Friday at 5pm. "
            "Thanks for keeping our shared space pleasant for everyone!",
        ),
        (
            "Optional: company trivia night this Thursday 6pm",
            "We are hosting a fun company trivia night this Thursday at 6pm in the main conf room. "
            "Drinks and snacks will be provided! Teams of 4, sign up on the wiki page. "
            "Completely optional and work talk is not allowed. Hope to see you there! "
            "Last time we had a great turnout and it was a blast.",
        ),
        (
            "FYI: updated 2024 holiday calendar now on HR portal",
            "The 2024 company holiday calendar has been finalized and posted on the HR portal. "
            "No major changes from what was announced in November. The only addition is a "
            "company half-day on July 5th following the Fourth of July holiday. "
            "Please update your personal calendars accordingly.",
        ),
        (
            "Parking lot B closed Monday-Wednesday for resurfacing",
            "Facilities has informed us that Lot B will be closed Monday through Wednesday "
            "next week for asphalt resurfacing work. Please use Lot A or the metered street "
            "parking on Oak Street during that time. Apologies for the inconvenience. "
            "Lot B will reopen Thursday morning.",
        ),
    ],
    "spam": [
        (
            "You have been selected! Claim your $500 Amazon gift card NOW",
            "Congratulations! Our automated system has randomly selected your email address "
            "as today's lucky winner. You are entitled to a $500 Amazon gift card completely "
            "free of charge. Simply click the secure link below and verify your details to "
            "claim your prize. This exclusive offer expires in 24 hours so act now!",
        ),
        (
            "LIMITED TIME: 90% off all software licenses - today only",
            "Do not miss this incredible opportunity! For the next 24 hours only we are "
            "offering 90% off our entire software catalog. Antivirus, productivity suites, "
            "design tools - all at prices you will not believe. Order now and receive a "
            "free USB drive worth $49. Use code SAVE90 at checkout. Act fast!",
        ),
        (
            "SECURITY ALERT: Your account has been compromised - verify immediately",
            "We have detected suspicious login activity on your account from an unknown IP "
            "address in Eastern Europe. To protect your account and prevent unauthorized access, "
            "please click the secure verification link below immediately and reset your password. "
            "Failure to verify within 2 hours will result in account suspension.",
        ),
        (
            "Earn $5,000 per week from home - no experience needed - 100% guaranteed",
            "Are you looking for financial freedom? Our proven passive income system has helped "
            "thousands of ordinary people earn $5,000 or more per week working just 2-3 hours "
            "a day from home. No experience required, no upfront investment, zero risk. "
            "Join our exclusive community today and receive our complete starter kit FREE.",
        ),
    ],
}


def generate_email(priority_label: str) -> Email:
    """Generate a single realistic email for the given priority category."""
    subject, body = random.choice(TEMPLATES[priority_label])
    sender_email, sender_name = random.choice(SENDERS)
    hours_ago = random.randint(0, 72)
    timestamp = (datetime.now() - timedelta(hours=hours_ago)).strftime("%Y-%m-%dT%H:%M:%S")
    email_id = str(uuid.uuid4())[:8]
    return Email(
        id=email_id,
        sender=f"{sender_name} <{sender_email}>",
        subject=subject,
        body=body,
        timestamp=timestamp,
    )


def generate_inbox(size: int = 10) -> list:
    """
    Generate a realistic inbox of `size` emails.
    Returns list of (Email, ground_truth_priority_str) tuples.
    Distribution: ~15% urgent, ~25% high, ~30% normal, ~15% low, ~15% spam
    """
    raw_counts = {
        "urgent": max(1, round(size * 0.15)),
        "high":   max(1, round(size * 0.25)),
        "normal": max(2, round(size * 0.30)),
        "low":    max(1, round(size * 0.15)),
        "spam":   max(1, round(size * 0.15)),
    }
    # Adjust total to exact size
    total = sum(raw_counts.values())
    diff = size - total
    if diff > 0:
        raw_counts["normal"] += diff
    elif diff < 0:
        raw_counts["low"] = max(1, raw_counts["low"] + diff)

    emails = []
    for priority, count in raw_counts.items():
        for _ in range(count):
            emails.append((generate_email(priority), priority))

    random.shuffle(emails)
    return emails[:size]
