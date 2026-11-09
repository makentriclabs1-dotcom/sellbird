"""Tests for the free PROFILE collector and the observation model."""

import asyncio
import os
from contextlib import closing
import tempfile

import pytest
import pytest_asyncio

from sellbird.collectors.profile import ProfileCollector, title_looks_like_role, _strip_code
from sellbird.models.company import Company
from sellbird.models.prospect import Prospect
from sellbird.signals import SIGNAL_CATALOG, seed_signal_catalog
from sellbird.state import StateManager


@pytest_asyncio.fixture
async def state():
    with tempfile.TemporaryDirectory() as tmp:
        sm = StateManager(os.path.join(tmp, "t.db"))
        await sm.init_db()
        yield sm


def collector(confirmed=None):
    c = ProfileCollector.__new__(ProfileCollector)
    c.confirmed = confirmed or set()
    c.site_names = {}
    return c


# ── agency detection (the highest-value signal) ──

def test_agency_detected_from_strong_credit():
    html = """<footer><p>© 2026 Acme Roofing.
      Powered by <a href="https://rcmdigital.com">RCM Digital</a></p></footer>"""
    got = collector().detect_agency(html, "acmeroofing.com")
    assert got is not None
    name, conf, url = got
    assert "RCM" in name and conf == 0.9 and "rcmdigital" in url


def test_agency_descriptive_anchor_scores_lower_but_records():
    html = """<footer><a href="https://xyz.com">Roofing Website Design and Marketing</a></footer>"""
    got = collector().detect_agency(html, "acme.com")
    assert got is not None and got[1] == 0.75


def test_bare_link_is_not_recorded():
    """A bare footer link is usually the business's own second brand."""
    html = """<footer><a href="https://someplace.com">Someplace</a></footer>"""
    assert collector().detect_agency(html, "acme.com") is None


def test_benign_footer_links_ignored():
    html = """<footer>
      <a href="https://facebook.com/acme">Facebook</a>
      <a href="https://g.page/acme">Google</a>
      <a href="https://gaf.com">GAF Certified</a>
      <a href="https://accessibe.com">Accessibility</a>
    </footer>"""
    assert collector().detect_agency(html, "acme.com") is None


def test_self_links_ignored():
    html = """<footer>Powered by <a href="https://acme.com/about">Acme</a></footer>"""
    assert collector().detect_agency(html, "acme.com") is None


# ── people: real names vs page titles ──

def test_real_people_extracted_with_titles():
    html = """<div><h3>Michael Torres</h3><p>Owner &amp; President</p>
               <h3>Sarah Chen</h3><p>Operations Manager</p>
               <h3>Dave Kowalski</h3><p>Lead Estimator</p></div>"""
    people = collector().detect_people(html)
    assert len(people) == 3, "trailing context must not consume later matches"
    assert people[0]["title"] == "Owner & President"   # entities decoded, no bleed
    assert people[1]["title"] == "Operations Manager"
    assert people[0]["is_decision_maker"] is True
    assert people[2]["is_decision_maker"] is False


def test_page_titles_are_not_people():
    """The classic failure: nav headings that look like Firstname Lastname."""
    html = """<ul><li><h3>Request Service</h3><p>Get a free estimate</p></li>
                  <li><h3>Baker History</h3><p>Founded in 1915</p></li>
                  <li><h3>Commercial Services</h3><p>Roof repair</p></li></ul>"""
    assert collector().detect_people(html) == []


def test_feature_lists_are_not_people():
    """Bold lead-ins on a services page: "<strong>HOA Boards</strong> — ..."."""
    html = """<ul><li><strong>HOA Boards</strong> — Desiring consistent oversight and
                  a trusted management partner dedicated to accountability.</li>
              <li><strong>Scope Alignment</strong> — Service levels are defined
                  clearly before engagement begins.</li>
              <li><strong>Process-Driven Consistency</strong> — Structured workflows
                  for every community.</li></ul>"""
    assert collector().detect_people(html) == []


def test_lowercase_nav_heading_is_not_a_name():
    html = """<strong class="nav">Customer care</strong>
              <div>Responsive Customer Service. 24/7/365 Availability.</div>"""
    assert collector().detect_people(html) == []


def test_title_stops_at_the_block_and_never_carries_html():
    """The 220-char window used to end mid-tag: 'CEO <div class="descriptio'."""
    bio = "Peter Lindqvist joined Acme in 2005 and serves as CEO, bringing experience. " * 4
    html = (f'<div><h3>Peter Lindqvist</h3>\n<p>CEO</p>\n<p class="description">{bio}</p>'
            '<h3>Rosa Delgado</h3><p>President</p></div>')
    people = collector().detect_people(html)
    assert [(p["first_name"], p["title"]) for p in people] == [
        ("Peter", "CEO"), ("Rosa", "President")]
    assert all(p["is_decision_maker"] for p in people)


def test_name_in_link_with_title_after_br():
    html = ('<p><a href="/nora-bexley/"><strong>Nora Bexley</strong></a>'
            '<br />Chief Executive Officer</p>')
    [person] = collector().detect_people(html)
    assert person["title"] == "Chief Executive Officer"
    assert person["is_decision_maker"] is True


def test_title_role_gate():
    assert title_looks_like_role("Owner and founder of the company")
    assert title_looks_like_role("Operations Manager")
    assert not title_looks_like_role("Get a free roof estimate today")


def test_strip_code_removes_script_before_text_extraction():
    html = '<script>var x = "CEO John Smith";</script><h3>Real Person</h3>'
    assert "CEO John Smith" not in _strip_code(html)


# ── tech / ads / gaps ──

def test_tech_and_ads_detection():
    c = collector()
    html = """<script src="https://www.googletagmanager.com/gtag/js?id=AW-123"></script>
              <link href="/wp-content/themes/x.css">
              <script src="https://connect.facebook.net/en_US/fbevents.js"></script>"""
    assert "WordPress" in c.detect_tech(html)
    assert c._any(html, __import__("sellbird.collectors.profile", fromlist=["GOOGLE_ADS"]).GOOGLE_ADS)
    assert c._any(html, __import__("sellbird.collectors.profile", fromlist=["META_PIXEL"]).META_PIXEL)


def test_clean_page_has_no_false_tech():
    assert collector().detect_tech("<html><body><h1>Acme</h1></body></html>") == []


# ── confirmation gate ──

@pytest.mark.asyncio
async def test_only_confirmed_signals_are_collected(state, monkeypatch):
    await seed_signal_catalog(state)
    await state.set_signal_status("TECH_STACK", "confirmed")
    confirmed = await state.confirmed_signal_codes()
    c = collector(confirmed)

    assert c.wants("TECH_STACK")
    assert not c.wants("INCUMBENT_AGENCY")   # proposed, never confirmed

    async def fake_get(url):
        return '<html><link href="/wp-content/x.css"><footer>Powered by <a href="https://ag.com">Agency</a></footer></html>'
    c._get = fake_get
    c.state = state

    cid = await state.add_company(Company(name="Acme", domain="acme.com"))
    obs = await c.profile_company(cid, "acme.com")
    codes = {o["signal_code"] for o in obs}
    assert codes == {"TECH_STACK"}, "unconfirmed signals must not be collected"


@pytest.mark.asyncio
async def test_seeded_catalog_starts_unconfirmed(state):
    n = await seed_signal_catalog(state)
    assert n == len(SIGNAL_CATALOG)
    assert await state.confirmed_signal_codes() == set()
    proposed = await state.get_signal_codes(status="proposed")
    assert len(proposed) == n
    # every proposed signal explains itself to the user
    for sig in proposed:
        assert sig["description"] and sig["cost_note"]


@pytest.mark.asyncio
async def test_user_decision_survives_reseeding(state):
    await seed_signal_catalog(state)
    await state.set_signal_status("SERP_RANK", "rejected")
    await seed_signal_catalog(state)          # e.g. after an upgrade
    codes = {c["code"]: c["status"] for c in await state.get_signal_codes()}
    assert codes["SERP_RANK"] == "rejected", "re-seeding must not undo a user's choice"


@pytest.mark.asyncio
async def test_profile_upgrades_a_name_the_domain_does_not_back(state, monkeypatch):
    cid = await state.add_company(Company(name="Denver, CO | HOA Management Firm",
                                          domain="managementharbor.com"))
    keep = await state.add_company(Company(name="Birch Management", domain="birchmgt.com"))
    derived = await state.add_company(Company(name="Grow With Us", domain="growwithus.us"))
    pages = {
        "managementharbor.com": '<meta property="og:site_name" content="Management Harbor">',
        "birchmgt.com": "<title>Denver HOA | Birch Management Group</title>",
        "growwithus.us": '<meta property="og:site_name" content="Grow Community Management">',
    }

    async def fake_resolve(self, domain):
        return f"https://{domain}", pages[domain]

    async def fake_first(self, site, paths):
        return "", ""

    monkeypatch.setattr(ProfileCollector, "_resolve_site", fake_resolve)
    monkeypatch.setattr(ProfileCollector, "_first_present", fake_first)
    async with ProfileCollector(state, {"TECH_STACK"}) as c:
        await c.run([{"id": cid, "name": "Denver, CO | HOA Management Firm",
                      "domain": "managementharbor.com"},
                     {"id": keep, "name": "Birch Management", "domain": "birchmgt.com"},
                     {"id": derived, "name": "Grow With Us", "domain": "growwithus.us"}])
    import sqlite3
    with closing(sqlite3.connect(state.db_path)) as db:
        names = dict(db.execute("SELECT domain, name FROM companies"))
    assert names["managementharbor.com"] == "Management Harbor"
    assert names["birchmgt.com"] == "Birch Management"   # already right: untouched
    assert names["growwithus.us"] == "Grow Community Management"   # was domain-derived


@pytest.mark.asyncio
@pytest.mark.parametrize("current,domain,homepage,expected", [
    ("Roofing Services in Denver", "kettlemanroofing.com", "Kettleman Roofing", "Kettleman Roofing"),
    ("Plumbing Services in Austin", "acmeplumbing.com", "Acme Plumbing", "Acme Plumbing"),
    ("Contact Us", "contactcraft.com", "ContactCraft", "ContactCraft"),
    ("Birch Management", "birchmgt.com", "Birch Management Group", "Birch Management"),
    ("NVCM", "nvcmnet.com", "NVCM Group", "NVCM"),
    ("Northvale Association of Community Managers", "nacm.org",
     "Northvale Association of Community Managers Group", "Northvale Association of Community Managers"),
    ("Grow With Us", "growwithus.us", "Grow Community Management", "Grow Community Management"),
    ("Roofing Services in Denver", "kettlemanroofing.com", "", "Roofing Services in Denver"),
])
async def test_profile_repairs_weak_names_and_retains_brands(
        state, monkeypatch, current, domain, homepage, expected):
    cid = await state.add_company(Company(name=current, domain=domain))
    pid = await state.add_prospect(Prospect(company_id=cid, company=current,
                                            first_name="Test", last_name="Person"))

    async def fake_resolve(self, domain):
        return f"https://{domain}", f"<title>{homepage}</title>"

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(ProfileCollector, "_resolve_site", fake_resolve)
    monkeypatch.setattr("sellbird.collectors.profile.asyncio.sleep", no_sleep)
    async with ProfileCollector(state, set()) as c:
        await c.run([{"id": cid, "name": current, "domain": domain}])
    import sqlite3
    with closing(sqlite3.connect(state.db_path)) as db:
        assert db.execute("SELECT name FROM companies WHERE id = ?", (cid,)).fetchone() == (expected,)
        assert db.execute("SELECT company FROM prospects WHERE id = ?", (pid,)).fetchone() == (expected,)


@pytest.mark.asyncio
@pytest.mark.parametrize("industry", ["Landscaping", "Electrical"])
@pytest.mark.parametrize("use_fallback", [False, True])
async def test_profile_uses_homepage_brand_over_service_location_title(
        state, monkeypatch, industry, use_fallback):
    from sellbird.collectors.discover import name_from_domain
    domain = f"kettleman{industry.lower()}.com"
    brand = f"Kettleman {industry}"
    label = f"{industry} Services in Denver"
    current = name_from_domain(domain) if use_fallback else label
    cid = await state.add_company(Company(name=current, domain=domain))
    pid = await state.add_prospect(Prospect(company_id=cid, company=current))

    async def fake_resolve(self, domain):
        return f"https://{domain}", (
            f'<meta property="og:site_name" content="{brand}"><title>{label}</title>')

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(ProfileCollector, "_resolve_site", fake_resolve)
    monkeypatch.setattr("sellbird.collectors.profile.asyncio.sleep", no_sleep)
    async with ProfileCollector(state, set()) as c:
        await c.run([{"id": cid, "name": current, "domain": domain}])
    assert (await state.get_company_by_domain(domain)).name == brand
    assert (await state.get_prospect(pid)).company == brand


@pytest.mark.asyncio
async def test_run_persists_partial_observations_when_company_times_out(state):
    """A timeout must not discard signals collected before cancellation."""
    # ProfileCollector.run() is tested directly, so seed the governed signal
    # vocabulary that the normal profile_companies() entry point seeds.
    await seed_signal_catalog(state)
    cid = await state.add_company(Company(name="Slow Plumbing", domain="slow.example"))
    c = ProfileCollector(state, {"TECH_STACK"}, timeout=0.03)

    async def collect_then_stall(company_id, domain, observations=None):
        observations.append({
            "signal_code": "TECH_STACK",
            "company_id": company_id,
            "collector": "profile",
            "value_num": None,
            "value_text": "WordPress",
            "confidence": 1.0,
            "evidence_url": "https://slow.example/",
        })
        await asyncio.sleep(1)

    c.profile_company = collect_then_stall
    async with c:
        saved = await c.run([{"id": cid, "name": "Slow Plumbing", "domain": "slow.example"}])

    rows = await state.get_observations(company_id=cid, signal_code="TECH_STACK")
    assert saved == 1
    assert len(rows) == 1
    assert rows[0]["value_text"] == "WordPress"


@pytest.mark.asyncio
async def test_get_uses_remaining_company_deadline_as_request_timeout():
    class SlowClient:
        async def get(self, url):
            await asyncio.sleep(1)
            raise AssertionError("request should have been cancelled first")

    c = ProfileCollector(state=None, confirmed=set(), timeout=1)
    c._client = SlowClient()
    c._deadline = asyncio.get_running_loop().time() + 0.03

    started = asyncio.get_running_loop().time()
    result = await c._get("https://example.com/slow")
    elapsed = asyncio.get_running_loop().time() - started

    assert result == ""
    assert elapsed < 0.25


@pytest.mark.asyncio
async def test_candidate_search_shares_company_deadline_across_paths():
    c = ProfileCollector(state=None, confirmed=set(), timeout=1)
    c._deadline = asyncio.get_running_loop().time() + 0.04
    calls = []

    async def slow_get(url):
        calls.append(url)
        await asyncio.sleep(1)
        return ""

    c._get = slow_get
    started = asyncio.get_running_loop().time()
    result = await c._first_present("https://example.com", ("/one", "/two", "/three"))
    elapsed = asyncio.get_running_loop().time() - started

    assert result == ("", "")
    assert len(calls) == 1
    assert elapsed < 0.25


@pytest.mark.asyncio
async def test_get_skips_host_after_confirmed_challenge():
    class CountingClient:
        def __init__(self):
            self.calls = 0
        async def get(self, url):
            self.calls += 1
            raise AssertionError("a challenged host must not be requested again")

    c = ProfileCollector(state=None, confirmed=set())
    c._client = CountingClient()
    c.challenged.add("blocked.example")

    assert await c._get("https://www.blocked.example/contact") == ""
    assert c._client.calls == 0


@pytest.mark.asyncio
async def test_run_continues_to_next_company_after_timeout(state, monkeypatch):
    # Direct collector tests must seed the observation vocabulary themselves.
    await seed_signal_catalog(state)
    first = await state.add_company(Company(name="Slow One", domain="slow-one.example"))
    second = await state.add_company(Company(name="Fast Two", domain="fast-two.example"))
    c = ProfileCollector(state, {"TECH_STACK"}, timeout=0.03)
    real_sleep = asyncio.sleep

    async def collect(company_id, domain, observations=None):
        observations.append({
            "signal_code": "TECH_STACK",
            "company_id": company_id,
            "collector": "profile",
            "value_text": "WordPress",
            "confidence": 1.0,
            "evidence_url": f"https://{domain}/",
        })
        if domain == "slow-one.example":
            await real_sleep(1)

    c.profile_company = collect
    monkeypatch.setattr("sellbird.collectors.profile.random.uniform", lambda *_: 0)
    async with c:
        saved = await c.run([
            {"id": first, "name": "Slow One", "domain": "slow-one.example"},
            {"id": second, "name": "Fast Two", "domain": "fast-two.example"},
        ])

    rows = await state.get_observations(signal_code="TECH_STACK")
    assert saved == 2
    assert {row["company_id"] for row in rows} == {first, second}


@pytest.mark.asyncio
async def test_get_blocks_host_after_plain_429_response():
    class RateLimitedResponse:
        status_code = 429
        text = "Too many requests"
        url = "https://www.limited.example/contact"
        headers = {"content-type": "text/html"}

    class CountingClient:
        def __init__(self):
            self.calls = 0

        async def get(self, url):
            self.calls += 1
            return RateLimitedResponse()

    c = ProfileCollector(state=None, confirmed=set())
    c._client = CountingClient()

    assert await c._get("https://limited.example/") == ""
    assert "limited.example" in c.challenged

    assert await c._get("https://www.limited.example/contact") == ""
    assert c._client.calls == 1
