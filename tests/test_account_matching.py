"""Tests for AccountMatchingService."""

from ai_ready_rag.services.account_matching import (
    AccountMatchingService,
    _normalize,
    _soundex,
    _strip_suffixes,
)


class TestNormalize:
    def test_lowercases(self):
        assert _normalize("HELLO WORLD") == "hello world"

    def test_strips_whitespace(self):
        assert _normalize("  hello  world  ") == "hello world"


class TestStripSuffixes:
    def test_strips_hoa(self):
        result = _strip_suffixes("Maple Grove HOA")
        assert "hoa" not in result.lower()
        assert "maple grove" in result.lower()

    def test_strips_llc(self):
        result = _strip_suffixes("Sunset Properties LLC")
        assert "llc" not in result.lower()

    def test_strips_association(self):
        result = _strip_suffixes("Oak Hills Homeowners Association")
        assert "association" not in result.lower()

    def test_preserves_core_name(self):
        result = _strip_suffixes("River Bend Condominiums HOA")
        assert "river bend" in result.lower()


class TestSoundex:
    def test_basic(self):
        assert _soundex("Robert") == _soundex("Rupert")

    def test_same_name(self):
        code = _soundex("Smith")
        assert code == "S530"

    def test_empty(self):
        assert _soundex("") == "0000"


class TestAccountMatchingService:
    def setup_method(self):
        self.svc = AccountMatchingService(confidence_threshold=0.6)
        self.candidates = [
            "Maple Grove HOA",
            "Sunset Properties LLC",
            "Oak Hills Homeowners Association",
            "River Bend Condominiums",
        ]

    def test_exact_match(self):
        result = self.svc.match("Maple Grove HOA", self.candidates)
        assert result.matched
        assert result.tier == 1
        assert result.confidence == 1.0
        assert result.method == "exact"

    def test_exact_case_insensitive(self):
        result = self.svc.match("maple grove hoa", self.candidates)
        assert result.matched
        assert result.tier == 1

    def test_suffix_stripped_match(self):
        result = self.svc.match("Maple Grove", self.candidates)
        assert result.matched
        assert result.tier == 2
        assert result.method == "suffix_stripped"

    def test_no_match(self):
        result = self.svc.match("Completely Different Name XYZ", self.candidates)
        assert not result.matched
        assert result.method == "no_match"

    def test_empty_candidates(self):
        result = self.svc.match("Any Name", [])
        assert not result.matched

    def test_match_all_returns_sorted(self):
        candidates = ["Maple Grove HOA", "Maple Grove Association", "Oak Hills"]
        results = self.svc.match_all("Maple Grove HOA", candidates)
        assert len(results) >= 1
        assert results[0].confidence >= results[-1].confidence
