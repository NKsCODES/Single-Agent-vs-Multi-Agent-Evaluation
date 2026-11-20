"""A synthetic enterprise knowledge base for Contoso, a fictional cloud software company.

12 products, each with an overview, a security page, a support policy and release notes, plus
a product catalog and a few distractor pages. The generator is seeded, and the facts it writes
into the text are kept in ``FACTS`` as ground truth for grading.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

PRODUCTS = [
    "Contoso Mail", "Contoso Drive", "Contoso Chat", "Contoso Forms", "Contoso Analytics",
    "Contoso CRM", "Contoso Payroll", "Contoso Helpdesk", "Contoso Sign", "Contoso Meet",
    "Contoso Planner", "Contoso Vault",
]
SLAS = ["99.9%", "99.95%", "99.99%"]
REGIONS = ["India Central", "India South", "West Europe", "North Europe", "East US",
           "Southeast Asia"]
PRICES = ["$4", "$6", "$8", "$10", "$12", "$15"]
MAX_USERS = ["5,000", "10,000", "25,000", "50,000", "100,000"]
ENCRYPTION = ["AES-256 with service-managed keys", "AES-256 with customer-managed keys",
              "AES-256 with double encryption"]
P1_RESPONSE = ["15 minutes", "30 minutes", "1 hour", "4 hours"]
FEATURE_WORDS = ["Smart", "Focus", "Auto", "Quick", "Secure", "Live", "Bulk", "Insight",
                 "Flex", "Guided", "Shared", "Instant"]
FEATURE_NOUNS = ["Compose", "Inbox", "Sync", "Templates", "Approvals", "Dashboards",
                 "Routing", "Export", "Reminders", "Archive", "Search", "Links"]

ATTRIBUTES = {
    "sla": "uptime SLA",
    "regions": "data residency regions",
    "price": "price per user per month",
    "max_users": "maximum users per tenant",
    "encryption": "encryption at rest",
    "p1_response": "P1 support response time",
}


@dataclass
class ProductFacts:
    name: str
    sla: str
    regions: list[str]
    price: str
    max_users: str
    encryption: str
    p1_response: str
    releases: list[tuple[str, str, str]] = field(default_factory=list)  # version, date, feature

    def value(self, attr: str) -> str:
        v = getattr(self, attr)
        return ", ".join(v) if isinstance(v, list) else str(v)


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str
    product: str | None = None


def slug(name: str) -> str:
    return name.lower().replace("contoso ", "").replace(" ", "-")


def build_facts(seed: int = 7) -> dict[str, ProductFacts]:
    rng = random.Random(seed)
    words, nouns = FEATURE_WORDS[:], FEATURE_NOUNS[:]
    features = [f"{w} {n}" for w in words for n in nouns]
    rng.shuffle(features)
    facts = {}
    for i, name in enumerate(PRODUCTS):
        regions = sorted(rng.sample(REGIONS, rng.randint(2, 4)), key=REGIONS.index)
        releases = []
        for j, (version, date) in enumerate([("3.1", "2025-11-04"), ("3.2", "2026-02-17"),
                                             ("4.0", "2026-06-30")]):
            releases.append((version, date, features[i * 3 + j]))
        facts[name] = ProductFacts(
            name=name, sla=rng.choice(SLAS), regions=regions, price=rng.choice(PRICES),
            max_users=rng.choice(MAX_USERS), encryption=rng.choice(ENCRYPTION),
            p1_response=rng.choice(P1_RESPONSE), releases=releases,
        )
    return facts


def build_documents(facts: dict[str, ProductFacts]) -> list[Document]:
    docs: list[Document] = []
    for name, f in facts.items():
        s = slug(name)
        docs.append(Document(f"{s}-overview", f"{name} overview", (
            f"{name} overview. {name} is part of the Contoso cloud suite for growing "
            "businesses. It is backed by a {sla} uptime SLA. Customer data can be stored in "
            "these regions: {regions}. Pricing starts at {price} per user per month on the "
            "standard plan. A single tenant supports up to {users} users. Administrators "
            "manage settings from the Contoso admin center, and usage reports refresh daily."
        ).format(sla=f.sla, regions=", ".join(f.regions), price=f.price, users=f.max_users),
            name))
        docs.append(Document(f"{s}-security", f"{name} security", (
            f"{name} security and compliance. All {name} data is encrypted at rest with "
            f"{f.encryption}. Data in transit uses TLS 1.2 or later. The service is audited "
            "every year, and access to customer content by support staff requires approval "
            "from the customer administrator."), name))
        docs.append(Document(f"{s}-support", f"{name} support policy", (
            f"{name} support policy. P1 incidents receive a first response within "
            f"{f.p1_response}. P2 incidents receive a first response within one business "
            "day. Support is available in English and Hindi, and every ticket can be "
            "escalated to a duty manager."), name))
        notes = " ".join(f"Version {v} ({d}) introduced {feat}." for v, d, feat in f.releases)
        docs.append(Document(f"{s}-releases", f"{name} release notes", (
            f"{name} release notes. {notes} Minor fixes ship every two weeks and do not "
            "change pricing or limits."), name))
    docs.append(Document("catalog", "Contoso product catalog", (
        "Contoso product catalog. Contoso offers these cloud products: "
        + ", ".join(facts) + ". Every product is sold per user per month.")))
    docs.append(Document("sla-glossary", "Service level glossary", (
        "Service level glossary. An uptime SLA is the monthly availability Contoso commits "
        "to. Service credits apply when availability falls below the committed level. "
        "Scheduled maintenance is announced seven days in advance.")))
    docs.append(Document("regions-guide", "Data residency guide", (
        "Data residency guide. Customers choose a region when they create a tenant. Data "
        "stays in the chosen region unless the administrator enables cross-region backup. "
        "Not every product is available in every region.")))
    docs.append(Document("company-news", "Contoso company news", (
        "Contoso company news. Contoso opened a new support centre in Pune and expanded its "
        "partner programme. The annual customer conference will be held in Bengaluru.")))
    return docs


FACTS = build_facts()
DOCUMENTS = build_documents(FACTS)


def feature_owner(feature: str) -> tuple[str, str] | None:
    for name, f in FACTS.items():
        for v, _, feat in f.releases:
            if feat.lower() == feature.lower():
                return name, v
    return None
