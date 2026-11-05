# ruff: noqa: E501  (template and distractor tables read best one per line)
"""Scenario D: scenario C plus what the Gmail and Google Calendar connectors add.

    python -m corpus.scenario_d build

Reads `corpus/scenario_c.json` and writes `corpus/scenario_d.json`. Deterministic:
a seeded random.Random, no LLM, no network. It adds connector copies of every
screen-captured Gmail message, calendar events tied to the planted plans (one of
them deliberately stale), about 300 bulk and promo emails with distractors, and a
few duplicate deliveries from sync retries. None of it changes a scenario C answer.
"""

from __future__ import annotations

import argparse
import random
import string
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from corpus.scenario_c import read_json, write_json
from sme.store import repo

SCENARIO = "d"
NOW = "2026-11-26T20:00:00+05:30"
SEED = 20261126
BACKFILL_AT = "2026-11-04T18:00:00+05:30"
FIRST_DAY = date(2026, 11, 2)
LAST_DAY = date(2026, 11, 26)
# Latest promo send time on the last day, so every sync lands before "now".
LAST_DAY_CUTOFF = 19 * 60 + 30
PROMO_TOTAL = 300
INBOX_ONLY_RATE = 0.10
SIGNATURE_RATE = 0.5

CORPUS_DIR = Path(__file__).resolve().parent
SOURCE_PATH = CORPUS_DIR / "scenario_c.json"
OUT_PATH = CORPUS_DIR / "scenario_d.json"

GMAIL = "gmail_connector"
CALENDAR = "calendar"
FORBIDDEN_FLIGHT = "6E 2134"

DESCRIPTION = (
    "Fictional test corpus. Scenario C plus what minimi's Gmail and Google Calendar"
    " connectors add: connector copies of every screen-captured email, calendar events"
    " (the Nov 20 dinner invite is stale), bulk and promo mail with distractors, and"
    " duplicate deliveries from sync retries. All times are Asia/Kolkata (IST, +05:30)."
    " Conversations are listed in ingest order and messages within a conversation in"
    " time order. Labels c01 to c48 are planted, n0001 and up are scenario C noise,"
    " d0001 and up are connector items."
)

SENDER_EMAILS = {
    "You": "you@example.com",
    "Samuel Rao": "samuel.rao@example.com",
    "Priya Nair": "priya.nair@lightboxventures.example.com",
    "Dev Mehta": "dev.mehta@example.com",
    "The Weekly Byte": "noreply@weeklybyte.example.com",
}
SIGNATURES = {
    "You": ("Sent from my iPhone",),
    "Samuel Rao": ("Sam", "Sent from my iPhone"),
    "Priya Nair": ("Priya Nair\nPartner, Lightbox Ventures",),
    "Dev Mehta": ("Dev", "Dev Mehta"),
}
NEWSLETTER_FOOTER = (
    "You are receiving The Weekly Byte because you subscribed. Unsubscribe | Manage preferences"
)

# Screen Gmail messages whose connector copy is delivered twice.
DUPLICATE_COPIES = (
    ("friday-dinner", "2026-11-18T10:15:00+05:30"),
    ("standup", "2026-11-12T09:30:00+05:30"),
)

WEEKDAY_ABBR = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
MONTH_ABBR = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
EVENT_ID_ALPHABET = "0123456789abcdefghijklmnopqrstuv"


@dataclass(frozen=True)
class Event:
    key: str
    title: str
    created: str
    start: str
    end: str
    location: str | None
    attendees: tuple[str, ...]
    organizer: str = "You"


# Anchors: c14 Nov 2 19:07, c17 Nov 4 18:02, c19 Nov 6 22:42, c21 Nov 13 23:02,
# c25 Nov 24 18:22, c27 Nov 24 18:26, c48 Nov 25 19:05, c04 Nov 10 11:30.
EVENTS: tuple[Event, ...] = (
    Event("event-2026-11-06-dinner-sam", "Dinner with Sam", "2026-11-02T19:10:00+05:30",
          "2026-11-06T20:00:00+05:30", "2026-11-06T22:00:00+05:30", "usual place", ("You", "Sam")),
    Event("event-2026-11-07-brunch-sam", "Brunch with Sam", "2026-11-04T18:05:00+05:30",
          "2026-11-07T10:00:00+05:30", "2026-11-07T12:00:00+05:30", None, ("You", "Sam")),
    Event("event-2026-11-10-investor-sync", "Investor sync", "2026-11-10T11:45:00+05:30",
          "2026-11-17T16:00:00+05:30", "2026-11-17T16:45:00+05:30", "Google Meet",
          ("Priya Nair", "You"), "Priya Nair"),
    Event("event-2026-11-13-dinner-sam", "Dinner with Sam", "2026-11-06T22:46:00+05:30",
          "2026-11-13T20:00:00+05:30", "2026-11-13T22:00:00+05:30", "usual place", ("You", "Sam")),
    # Stale on purpose: Sam's Nov 18 email (c01, c02) moved this dinner to 9:30 PM
    # and the invite was never updated.
    Event("event-2026-11-20-dinner-sam", "Dinner with Sam", "2026-11-13T23:05:00+05:30",
          "2026-11-20T20:00:00+05:30", "2026-11-20T22:00:00+05:30", "usual place", ("You", "Sam")),
    Event("event-2026-11-27-dinner-sam", "Dinner with Sam", "2026-11-24T18:23:00+05:30",
          "2026-11-27T21:30:00+05:30", "2026-11-27T23:30:00+05:30", "usual place", ("You", "Sam")),
    Event("event-2026-11-28-brunch-sam-arjun", "Brunch with Sam and Arjun",
          "2026-11-24T18:28:00+05:30", "2026-11-28T09:30:00+05:30", "2026-11-28T11:30:00+05:30",
          None, ("You", "Sam", "Arjun")),
    Event("event-2026-11-29-lunch-mom", "Lunch at home with Mom", "2026-11-25T19:08:00+05:30",
          "2026-11-29T13:00:00+05:30", "2026-11-29T15:00:00+05:30", "Home", ("You", "Mom")),
)  # fmt: skip


@dataclass(frozen=True)
class Brand:
    key: str
    name: str
    email: str
    category: str
    weight: int
    items: tuple[str, ...]
    templates: tuple[tuple[str, str], ...]
    amounts: tuple[str, ...] = ("Rs 99", "Rs 149", "Rs 199", "Rs 249", "Rs 299", "Rs 499", "Rs 999")


PROMOS = "CATEGORY_PROMOTIONS"
UPDATES = "CATEGORY_UPDATES"
SOCIAL = "CATEGORY_SOCIAL"

CITIES = ("Goa", "Jaipur", "Kochi", "Mumbai", "Hyderabad", "Chennai", "Kolkata", "Pune",
          "Udaipur", "Srinagar", "Amritsar", "Varanasi")  # fmt: skip
RESTAURANTS = ("Olive Bar & Kitchen", "Toit", "The Fatty Bao", "Farzi Cafe", "Bombay Brasserie",
               "Smoke House Deli", "Social", "Indigo Deli")  # fmt: skip
PERCENTS = (10, 15, 20, 25, 30, 40, 50, 60)
FARES = ("Rs 1,999", "Rs 2,499", "Rs 2,899", "Rs 3,199", "Rs 3,799", "Rs 4,499")
COUNTS = (3, 5, 7, 12, 18, 24, 31)

BRANDS: tuple[Brand, ...] = (
    Brand("promo-swiggy", "Swiggy", "noreply@swiggy.in", PROMOS, 22,
          ("biryani", "pizza", "dosa", "momos", "burgers", "rolls"), (
        ("{pct}% off on your next {item} order",
         "Craving {item}? Use code {code} and get {pct}% off on orders above {amt}. Valid till midnight."),
        ("Free delivery all week with Swiggy One",
         "Members get free delivery on orders above {amt} and extra discounts at top restaurants. Join today."),
        ("Meals under {amt}, delivered hot",
         "Top-rated restaurants near you, every meal under {amt}. Order in the next 2 hours."),
        ("Late night cravings? We deliver till 3 AM",
         "{item} and more, delivered late. Flat {pct}% off with code {code}."),
        ("Instamart: groceries in 10 minutes",
         "Fresh fruit, vegetables and daily essentials in minutes. Get {amt} off your first Instamart order."),
    )),
    Brand("promo-zomato", "Zomato", "noreply@zomato.com", PROMOS, 20,
          ("biryani", "pizza", "shawarma", "thali", "noodles", "desserts"), (
        ("{item} at {pct}% off near you",
         "The most ordered {item} in your area, now {pct}% off. Use code {code}."),
        ("Zomato Gold: unlimited free deliveries",
         "Upgrade to Gold for {amt} and get free deliveries from {count}+ restaurants near you."),
        ("Trending in your city: {item}",
         "{count} people near you ordered {item} today. See what the fuss is about."),
        ("Get {amt} cashback on your order today",
         "Pay with any UPI app and get up to {amt} cashback. Offer ends tonight."),
    )),
    Brand("promo-eazydiner", "EazyDiner", "noreply@eazydiner.com", PROMOS, 12,
          RESTAURANTS, (
        ("Up to {pct}% off at {item}",
         "Book a table at {item} through EazyDiner and pay {pct}% less on the total bill."),
        ("Prime members save {pct}% more this week",
         "EazyDiner Prime gets you an extra {pct}% off at {count} restaurants. Renew for {amt}."),
        ("New on EazyDiner: {item}",
         "{item} is now on EazyDiner. Reserve in seconds and earn {count} EazyPoints."),
    )),
    Brand("promo-eatfit", "EatFit", "noreply@eatfit.in", PROMOS, 8,
          ("protein bowls", "millet khichdi", "paneer wraps", "salads", "smoothie bowls"), (
        ("Healthy {item} from {amt}",
         "Calorie-counted {item}, delivered fresh. Starting at {amt}."),
        ("{pct}% off your first EatFit subscription",
         "Plan your meals for the week and save {pct}%. Use code {code}."),
    )),
    Brand("promo-indigo", "IndiGo", "noreply@goindigo.in", PROMOS, 12, CITIES, (
        ("Sale: fares from {amt} to {item}",
         "Book by Sunday for travel till March. One way fares to {item} from {amt}. T&C apply."),
        ("Add 6E Prime to your next booking",
         "Seat, meal and priority check-in in one bundle. Save up to {pct}% when you add it at booking."),
        ("Fly to {item} this winter",
         "Winter getaways to {item} with fares from {amt}. Free date change on select flights."),
    ), FARES),
    Brand("promo-airindia", "Air India", "noreply@airindia.com", PROMOS, 8, CITIES, (
        ("Flying Returns: earn double points to {item}",
         "Earn double Flying Returns points on all flights to {item} booked this week."),
        ("Upgrade to Premium Economy from {amt}",
         "Extra legroom and priority boarding on domestic routes from {amt}. Limited seats."),
    ), FARES),
    Brand("promo-makemytrip", "MakeMyTrip", "noreply@makemytrip.com", PROMOS, 12, CITIES, (
        ("Hotels in {item} up to {pct}% off",
         "Handpicked stays in {item} at up to {pct}% off. Use code {code}."),
        ("Your winter holiday to {item} from {amt}",
         "Flights, hotel and transfers to {item} in one package from {amt} per person."),
        ("Long weekend ideas: {count} places under 5 hours away",
         "Short trips that fit a long weekend. Bus, train and flight deals inside."),
    ), FARES),
    Brand("promo-myntra", "Myntra", "noreply@myntra.com", PROMOS, 16,
          ("sneakers", "winter jackets", "kurtas", "denim", "watches", "sunglasses"), (
        ("End of Reason Sale: {pct}% off {item}",
         "Top brands, {pct}% off on {item}. Sale ends in 48 hours."),
        ("Just dropped: new {item}",
         "Fresh arrivals in {item}. Free delivery on orders above {amt}."),
        ("Extra {pct}% off with code {code}",
         "Stack an extra {pct}% on sale prices across {item} and more."),
    )),
    Brand("promo-ajio", "AJIO", "noreply@ajio.com", PROMOS, 10,
          ("shirts", "trainers", "handbags", "ethnic wear", "hoodies"), (
        ("Big Bold Sale: {item} from {amt}",
         "{item} from {amt}. {count} brands, one sale."),
        ("{pct}% off for the next 6 hours",
         "Flash sale on {item}. Ends at midnight."),
    )),
    Brand("promo-nykaa", "Nykaa", "noreply@nykaa.com", PROMOS, 8,
          ("skincare", "fragrances", "grooming kits", "sunscreen"), (
        ("Pink Friday: up to {pct}% off {item}",
         "Our biggest sale on {item} is live. Free gift on orders above {amt}."),
        ("Bestsellers in {item} you'll love",
         "{count} top picks in {item}, curated for you."),
    )),
    Brand("promo-hdfc", "HDFC Bank", "noreply@hdfcbank.net", UPDATES, 10,
          ("electronics", "travel", "dining", "fuel"), (
        ("Your monthly account statement is ready",
         "Your e-statement is available in NetBanking. This is an automated email, please do not reply."),
        ("{pct}% cashback on {item} with your credit card",
         "Use your HDFC Bank credit card on {item} and get {pct}% cashback up to {amt}."),
        ("Pre-approved personal loan up to Rs 5 lakh",
         "You are eligible for an instant personal loan. Check your offer in the app."),
    )),
    Brand("promo-icici", "ICICI Bank", "noreply@icicibank.com", PROMOS, 6,
          ("shopping", "travel", "movies"), (
        ("Lifetime free credit card for {item}",
         "No joining fee, no annual fee, {pct}% value back on {item}. Apply in 2 minutes."),
        ("Festive offers on {item}",
         "Save up to {amt} on {item} with ICICI Bank cards."),
    )),
    Brand("promo-paytm", "Paytm", "noreply@paytm.com", UPDATES, 10,
          ("mobile recharge", "DTH recharge", "electricity bill", "FASTag"), (
        ("Cashback up to {amt} on {item}",
         "Pay for {item} on Paytm and get up to {amt} cashback. Offer valid this week."),
        ("Scratch card unlocked",
         "You have a new scratch card worth up to {amt}. Open the app to reveal it."),
    )),
    Brand("promo-amazon", "Amazon.in", "store-news@amazon.in", PROMOS, 18,
          ("headphones", "air fryers", "running shoes", "smartwatches", "backpacks", "kindles"), (
        ("Deals on {item} you might like",
         "Today's deals on {item}: up to {pct}% off. Prices as marked."),
        ("Black Friday sale: {pct}% off {item}",
         "Limited time deals on {item} and more. Prime members get early access."),
        ("Recommended for you: {item}",
         "Based on your browsing, here are {count} picks in {item}."),
    )),
    Brand("promo-flipkart", "Flipkart", "noreply@flipkart.com", PROMOS, 14,
          ("smartphones", "laptops", "TVs", "earbuds", "mixer grinders"), (
        ("Big Billion savings on {item}",
         "{item} at up to {pct}% off. No cost EMI and exchange offers available."),
        ("Price drop alert: {item}",
         "{item} on your wishlist just got cheaper by {amt}."),
        ("SuperCoins: {count} coins expiring soon",
         "Use your SuperCoins on {item} before they expire."),
    )),
    Brand("promo-linkedin", "LinkedIn", "messages-noreply@linkedin.com", SOCIAL, 14,
          ("product", "engineering", "design", "growth"), (
        ("You appeared in {count} searches this week",
         "See who's looking. Upgrade to Premium to see everyone who viewed your profile."),
        ("Trending in {item} today",
         "{count} new posts from people in {item} you might know."),
        ("New jobs in {item} for you",
         "{count} new jobs match your preferences. Apply with one click."),
    )),
    Brand("promo-instagram", "Instagram", "no-reply@mail.instagram.com", SOCIAL, 10,
          ("reels", "posts", "stories"), (
        ("You have {count} new notifications",
         "See {item} and comments you might have missed."),
        ("See what you missed on Instagram",
         "{count} new {item} from accounts you follow."),
    )),
    Brand("promo-notion", "Notion", "team@makenotion.com", UPDATES, 6,
          ("databases", "AI", "calendar", "sites"), (
        ("What's new in Notion: {item}",
         "This month we shipped updates to {item}. Read the release notes."),
        ("Your weekly workspace digest",
         "{count} pages were edited in your workspace this week."),
    )),
    Brand("promo-dropbox", "Dropbox", "no-reply@dropbox.com", UPDATES, 6,
          ("photos", "documents", "backups"), (
        ("Your Dropbox is {pct}% full",
         "Upgrade to Plus for 2 TB of space for {item} and more."),
        ("Try Dropbox Plus free for 30 days",
         "More space, more devices, better sharing for your {item}."),
    )),
    Brand("promo-finshots", "Finshots", "morning@finshots.in", UPDATES, 14,
          ("the rupee", "gold prices", "IPOs", "credit cards", "quick commerce"), (
        ("Finshots Daily: the story of {item}",
         "In today's issue we explain {item} in under 3 minutes."),
        ("Why everyone is talking about {item}",
         "A short read on {item}, plus {count} links worth your time."),
    )),
    Brand("promo-medium", "Medium Daily Digest", "noreply@medium.com", UPDATES, 12,
          ("Python", "LLMs", "startups", "productivity", "system design"), (
        ("Today's highlights: {item}",
         "Stories on {item} picked for you. {count} min read."),
        ("{count} stories on {item} you'll like",
         "Based on your reading history, here are new stories on {item}."),
    )),
    Brand("promo-bookmyshow", "BookMyShow", "noreply@bookmyshow.com", PROMOS, 10,
          ("comedy shows", "concerts", "new releases", "plays"), (
        ("{item} near you this week",
         "{count} {item} happening near you. Book now, seats are filling fast."),
        ("{pct}% off on movie tickets",
         "Use code {code} on your next booking. Valid on weekdays."),
    )),
)  # fmt: skip
BRAND_BY_KEY = {b.key: b for b in BRANDS}


@dataclass(frozen=True)
class Distractor:
    brand: str
    sent_at: str
    subject: str
    text: str
    kind: str
    duplicate: bool = False


DISTRACTORS: tuple[Distractor, ...] = (
    Distractor("promo-eazydiner", "2026-11-25T12:10:00+05:30",
               "Friday dinner for two at 8 PM: 20% off at Olive",
               "Reserve Friday dinner for two at 8 PM at Olive Bar & Kitchen and get 20% off the bill.",
               "dinner_time", duplicate=True),
    Distractor("promo-swiggy", "2026-11-26T11:40:00+05:30",
               "Dinner tomorrow at 8? Flat 25% off on Dineout tables",
               "Planning dinner tomorrow at 8 PM? Book a Dineout table and get flat 25% off.",
               "dinner_time"),
    Distractor("promo-zomato", "2026-11-19T17:15:00+05:30",
               "Friday night sorted: dinner reservations at 8 PM, 15% off",
               "Lock in a Friday table for 8 PM at restaurants near you and save 15%.",
               "dinner_time"),
    Distractor("promo-zomato", "2026-11-25T09:20:00+05:30",
               "Your weekend plans: brunch deals this Saturday",
               "Make this Saturday count. Unlimited brunch buffets from Rs 899 at 40 places near you.",
               "weekend"),
    Distractor("promo-bookmyshow", "2026-11-26T10:05:00+05:30",
               "Plan your weekend: Saturday 10 AM brunch and movie combos",
               "Brunch at 10 AM followed by a matinee. Combos this Saturday and Sunday from Rs 649.",
               "weekend"),
    Distractor("promo-eazydiner", "2026-11-24T19:30:00+05:30",
               "Sunday lunch at 1 PM: family thali deals",
               "Gather the family for Sunday lunch at 1 PM. Thalis for four at 25% off this weekend.",
               "weekend"),
    Distractor("promo-indigo", "2026-11-20T08:45:00+05:30",
               "IndiGo: fares to Delhi from Rs 3,499, flight 6E 2131",
               "Fly to Delhi on 6E 2131 from Rs 3,499 one way. Book by Sunday.",
               "flight"),
    Distractor("promo-indigo", "2026-11-23T14:30:00+05:30",
               "Delhi in December: 6E 2314 from Rs 4,199",
               "Beat the rush. 6E 2314 to Delhi departs 06:15 daily in December. Fares from Rs 4,199.",
               "flight"),
    Distractor("promo-makemytrip", "2026-11-24T16:00:00+05:30",
               "Fly to Delhi on Dec 2 from Rs 4,050: AI 504, 6E 2131, UK 819",
               "Cheapest flights to Delhi on December 2: AI 504, 6E 2131 and UK 819 from Rs 4,050.",
               "flight"),
    Distractor("promo-hdfc", "2026-11-21T10:30:00+05:30",
               "Reminder: you still need to complete your KYC",
               "Your KYC update is pending. Complete it in NetBanking to avoid restrictions on your account.",
               "todo"),
    Distractor("promo-amazon", "2026-11-22T20:15:00+05:30",
               "Your cart is waiting",
               "You left 2 items in your cart. Complete your order before prices change.",
               "todo"),
    Distractor("promo-notion", "2026-11-18T09:10:00+05:30",
               "You still need to finish setting up your workspace",
               "You are 2 steps away from a finished workspace. Pick up where you left off.",
               "todo"),
    Distractor("promo-linkedin", "2026-11-25T08:05:00+05:30",
               "You have 3 pending invitations to respond to",
               "3 people are waiting for your response. Accept or ignore their invitations.",
               "todo"),
    Distractor("promo-eatfit", "2026-11-23T11:00:00+05:30",
               "Go vegetarian this week with 30% off veg bowls",
               "Meat-free week? Every veg bowl is 30% off till Sunday. Pure vegetarian kitchens only.",
               "food"),
    Distractor("promo-swiggy", "2026-11-17T12:20:00+05:30",
               "Vegetarian? 100% pure veg restaurants near you",
               "Explore pure veg restaurants near you with free delivery on your first 3 orders.",
               "food"),
    Distractor("promo-zomato", "2026-11-20T16:40:00+05:30",
               "Fish Friday: 20% off seafood platters",
               "Fish curry, fried prawns and seafood platters at 20% off, this Friday only.",
               "food"),
)  # fmt: skip

# Random promo emails (not distractors) delivered twice, picked from these brands.
DUPLICATE_PROMO_BRANDS = ("promo-swiggy",)


def days() -> list[date]:
    return [FIRST_DAY + timedelta(days=i) for i in range((LAST_DAY - FIRST_DAY).days + 1)]


def iso(dt: datetime) -> str:
    return repo.to_iso(dt)


def clock(dt: datetime) -> str:
    hour = dt.hour % 12 or 12
    return f"{hour}:{dt.minute:02d} {'AM' if dt.hour < 12 else 'PM'}"


def event_body(event: Event) -> str:
    start, end = repo.parse_time(event.start), repo.parse_time(event.end)
    day = f"{WEEKDAY_ABBR[start.weekday()]} {MONTH_ABBR[start.month - 1]} {start.day} {start.year}"
    parts = [f"Calendar event: {event.title}.", f"{day}, {clock(start)} to {clock(end)}."]
    if event.location:
        parts.append(f"Location: {event.location}.")
    parts.append(f"Attendees: {', '.join(event.attendees)}.")
    if event.organizer != "You":
        parts.append(f"Organizer: {event.organizer}.")
    return " ".join(parts)


def promo_body(brand: Brand, subject: str, text: str) -> str:
    footer = (
        f"You are receiving this email because you subscribed to {brand.name}."
        " Unsubscribe | Manage preferences"
    )
    return f"Subject: {subject}\n\n{text}\n\n{footer}"


class Generator:
    def __init__(self, source: dict[str, Any]) -> None:
        self.source = source
        self.rng = random.Random(SEED)
        self.now = repo.parse_time(NOW)
        self.backfill = repo.parse_time(BACKFILL_AT)
        self.used_ids: set[str] = set()
        # (app, key) -> list of (sent, seen, index, message)
        self.items: dict[tuple[str, str], list[tuple[datetime, datetime, int, dict[str, Any]]]] = {}
        self.conversations: list[dict[str, str]] = []
        self.count = 0

    def sync_time(self, sent: datetime) -> datetime:
        if sent < self.backfill:
            return self.backfill
        return sent + timedelta(minutes=self.rng.randint(2, 15))

    def gmail_id(self, sent: datetime) -> str:
        while True:
            value = f"{int(sent.timestamp() * 1000):011x}{self.rng.getrandbits(20):05x}"
            if value not in self.used_ids:
                self.used_ids.add(value)
                return value

    def event_id(self) -> str:
        while True:
            value = "".join(self.rng.choice(EVENT_ID_ALPHABET) for _ in range(26))
            if value not in self.used_ids:
                self.used_ids.add(value)
                return value

    def add_conversation(self, app: str, key: str, title: str) -> None:
        self.conversations.append({"app": app, "key": key, "title": title})
        self.items[(app, key)] = []

    def add(self, msg: dict[str, Any]) -> dict[str, Any]:
        sent, seen = repo.parse_time(msg["sent_at"]), repo.parse_time(msg["seen_at"])
        self.items[(msg["app"], msg["thread"])].append((sent, seen, self.count, msg))
        self.count += 1
        return msg

    def duplicate(self, msg: dict[str, Any]) -> None:
        seen = repo.parse_time(msg["seen_at"]) + timedelta(minutes=self.rng.randint(20, 90))
        self.add({**msg, "seen_at": iso(seen)})

    def mail(
        self, thread: str, sender: str, sent: datetime, body: str, labels: list[str], email: str
    ) -> dict[str, Any]:
        return {
            "label": None,
            "app": GMAIL,
            "thread": thread,
            "sender": sender,
            "sent_at": iso(sent),
            "date_source": "absolute",
            "body": body,
            "source": "connector",
            "seen_at": iso(self.sync_time(sent)),
            "external_id": self.gmail_id(sent),
            "labels": labels,
            "sender_email": email,
        }

    def gmail_copies(self) -> None:
        for conv in self.source["conversations"]:
            if conv["app"] == "gmail":
                self.add_conversation(GMAIL, conv["key"], conv["title"])
        for msg in self.source["messages"]:
            if msg["app"] != "gmail":
                continue
            sender = msg["sender"]
            body = msg["body"]
            if sender == "The Weekly Byte":
                body = f"{body}\n\n{NEWSLETTER_FOOTER}"
            elif sender in SIGNATURES and self.rng.random() < SIGNATURE_RATE:
                body = f"{body}\n\n{self.rng.choice(SIGNATURES[sender])}"
            copy = self.add(
                self.mail(
                    msg["thread"],
                    sender,
                    repo.parse_time(msg["sent_at"]),
                    body,
                    ["INBOX"],
                    SENDER_EMAILS[sender],
                )
            )
            if (msg["thread"], msg["sent_at"]) in DUPLICATE_COPIES:
                self.duplicate(copy)

    def promo_labels(self, brand: Brand) -> list[str]:
        return ["INBOX"] if self.rng.random() < INBOX_ONLY_RATE else [brand.category]

    def random_time(self) -> datetime:
        day = self.rng.choice(days())
        latest = LAST_DAY_CUTOFF if day == LAST_DAY else 21 * 60 + 30
        minute = self.rng.randint(7 * 60, latest)
        return datetime(day.year, day.month, day.day, minute // 60, minute % 60, tzinfo=repo.IST)

    def fill(self, brand: Brand, subject: str, text: str) -> tuple[str, str]:
        """Fill both templates from one draw, so subject and body agree."""
        code = "".join(self.rng.choice(string.ascii_uppercase) for _ in range(4))
        slots = {
            "item": self.rng.choice(brand.items),
            "pct": self.rng.choice(PERCENTS),
            "amt": self.rng.choice(brand.amounts),
            "count": self.rng.choice(COUNTS),
            "code": f"{code}{self.rng.choice(PERCENTS)}",
        }
        return subject.format(**slots), text.format(**slots)

    def promos(self) -> None:
        for brand in BRANDS:
            self.add_conversation(GMAIL, brand.key, brand.name)
        for d in DISTRACTORS:
            brand = BRAND_BY_KEY[d.brand]
            msg = self.add(
                self.mail(
                    brand.key,
                    brand.name,
                    repo.parse_time(d.sent_at),
                    promo_body(brand, d.subject, d.text),
                    self.promo_labels(brand),
                    brand.email,
                )
            )
            if d.duplicate:
                self.duplicate(msg)
        weights = [b.weight for b in BRANDS]
        for _ in range(PROMO_TOTAL - len(DISTRACTORS)):
            brand = self.rng.choices(BRANDS, weights=weights)[0]
            subject, text = self.fill(brand, *self.rng.choice(brand.templates))
            self.add(
                self.mail(
                    brand.key,
                    brand.name,
                    self.random_time(),
                    promo_body(brand, subject, text),
                    self.promo_labels(brand),
                    brand.email,
                )
            )
        subjects = {d.subject for d in DISTRACTORS}
        for key in DUPLICATE_PROMO_BRANDS:
            pool = [
                m
                for sent, _, _, m in self.items[(GMAIL, key)]
                if sent.date() < LAST_DAY and m["body"].split("\n", 1)[0][9:] not in subjects
            ]
            self.duplicate(self.rng.choice(pool))

    def calendar(self) -> None:
        for event in sorted(EVENTS, key=lambda e: repo.parse_time(e.created)):
            self.add_conversation(CALENDAR, event.key, event.title)
            created = repo.parse_time(event.created)
            self.add(
                {
                    "label": None,
                    "app": CALENDAR,
                    "thread": event.key,
                    "sender": event.organizer,
                    "sent_at": iso(created),
                    "date_source": "absolute",
                    "body": event_body(event),
                    "source": "connector",
                    "seen_at": iso(self.sync_time(created)),
                    "external_id": self.event_id(),
                    "event": {
                        "title": event.title,
                        "start": event.start,
                        "end": event.end,
                        "location": event.location,
                        "attendees": list(event.attendees),
                        "status": "confirmed",
                        "organizer": event.organizer,
                    },
                }
            )

    def build(self) -> dict[str, Any]:
        self.gmail_copies()
        self.promos()
        self.calendar()

        messages = [
            {**m, "source": "screen", "seen_at": m["sent_at"]} for m in self.source["messages"]
        ]
        n = 0
        for conv in self.conversations:
            for _, _, _, msg in sorted(self.items[(conv["app"], conv["key"])], key=lambda t: t[:3]):
                n += 1
                msg["label"] = f"d{n:04d}"
                messages.append(msg)

        for msg in messages[len(self.source["messages"]) :]:
            for field in ("sent_at", "seen_at"):
                if repo.parse_time(msg[field]) >= self.now:
                    raise ValueError(f"{msg['label']}: {field} {msg[field]} is not before now")
            if FORBIDDEN_FLIGHT in msg["body"]:
                raise ValueError(f"{msg['label']}: body mentions {FORBIDDEN_FLIGHT}")

        return {
            "scenario": SCENARIO,
            "description": DESCRIPTION,
            "now": self.source["now"],
            "conversations": self.source["conversations"] + self.conversations,
            "messages": messages,
        }


def build(source: dict[str, Any]) -> dict[str, Any]:
    if source["now"] != NOW:
        raise ValueError(f"source scenario has now={source['now']!r}, expected {NOW!r}")
    return Generator(source).build()


def cmd_build(args: argparse.Namespace) -> int:
    scenario = build(read_json(args.source))
    write_json(args.out, scenario)
    new = sum(1 for m in scenario["messages"] if m["label"].startswith("d"))
    print(f"wrote {len(scenario['messages'])} messages ({new} connector items) to {args.out}")
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Scenario D corpus tools")
    sub = parser.add_subparsers(dest="command", required=True)
    build_cmd = sub.add_parser("build", help="build scenario D from scenario C")
    build_cmd.add_argument("--source", default=str(SOURCE_PATH))
    build_cmd.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args(argv)
    sys.exit(cmd_build(args))


if __name__ == "__main__":
    main()
