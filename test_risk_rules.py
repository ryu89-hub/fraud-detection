from risk_rules import label_risk, score_transaction


def _base_tx(**overrides):
    """Minimal zero-risk transaction; override individual fields to isolate signals."""
    tx = {
        "device_risk_score": 5,
        "is_international": 0,
        "amount_usd": 10.0,
        "velocity_24h": 1,
        "failed_logins_24h": 0,
        "prior_chargebacks": 0,
    }
    tx.update(overrides)
    return tx


# --- label_risk ---

def test_label_risk_thresholds():
    assert label_risk(10) == "low"
    assert label_risk(35) == "medium"
    assert label_risk(75) == "high"


def test_label_risk_boundaries():
    assert label_risk(29) == "low"
    assert label_risk(30) == "medium"
    assert label_risk(59) == "medium"
    assert label_risk(60) == "high"


# --- score_transaction: baseline ---

def test_clean_transaction_scores_zero():
    assert score_transaction(_base_tx()) == 0


# --- score_transaction: individual signals ---

def test_large_amount_adds_risk():
    assert score_transaction(_base_tx(amount_usd=1200)) == 25


def test_medium_amount_adds_10():
    assert score_transaction(_base_tx(amount_usd=600)) == 10


def test_high_device_risk_adds_25():
    assert score_transaction(_base_tx(device_risk_score=75)) == 25


def test_medium_device_risk_adds_10():
    assert score_transaction(_base_tx(device_risk_score=50)) == 10


def test_international_adds_15():
    assert score_transaction(_base_tx(is_international=1)) == 15


def test_high_velocity_adds_20():
    assert score_transaction(_base_tx(velocity_24h=8)) == 20


def test_medium_velocity_adds_5():
    assert score_transaction(_base_tx(velocity_24h=4)) == 5


def test_high_failed_logins_adds_20():
    assert score_transaction(_base_tx(failed_logins_24h=6)) == 20


def test_medium_failed_logins_adds_10():
    assert score_transaction(_base_tx(failed_logins_24h=3)) == 10


def test_two_prior_chargebacks_adds_20():
    assert score_transaction(_base_tx(prior_chargebacks=2)) == 20


def test_one_prior_chargeback_adds_5():
    assert score_transaction(_base_tx(prior_chargebacks=1)) == 5


# --- score_transaction: combined signals and clamping ---

def test_combined_signals_accumulate():
    # device_risk=80: +25, international: +15, amount=1200: +25 → 65
    tx = _base_tx(device_risk_score=80, is_international=1, amount_usd=1200)
    assert score_transaction(tx) == 65
    assert label_risk(65) == "high"


def test_score_capped_at_100():
    # 25 + 15 + 25 + 20 + 20 + 20 = 125, clamped to 100
    tx = _base_tx(
        device_risk_score=85,
        is_international=1,
        amount_usd=1500,
        velocity_24h=8,
        failed_logins_24h=6,
        prior_chargebacks=3,
    )
    assert score_transaction(tx) == 100
