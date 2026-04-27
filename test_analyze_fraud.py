from __future__ import annotations

import pandas as pd
import pytest

from analyze_fraud import load_inputs, score_transactions, summarize_results
from features import build_model_frame


# --- Shared fixtures ---

def _scored():
    """Five transactions spread across all three risk buckets."""
    return pd.DataFrame({
        "transaction_id": ["T1", "T2", "T3", "T4", "T5"],
        "amount_usd": [100.0, 200.0, 300.0, 400.0, 500.0],
        "risk_label": ["high", "high", "medium", "medium", "low"],
    })


def _chargebacks(*ids):
    return pd.DataFrame({"transaction_id": list(ids)})


# --- build_model_frame ---

def test_is_large_amount_flag():
    txns = pd.DataFrame({
        "transaction_id": [1, 2, 3],
        "account_id": [1, 1, 1],
        "amount_usd": [999.99, 1000.0, 1500.0],
    })
    accounts = pd.DataFrame({"account_id": [1]})
    result = build_model_frame(txns, accounts)
    assert list(result["is_large_amount"]) == [0, 1, 1]


def test_build_model_frame_merges_account_columns():
    txns = pd.DataFrame({"transaction_id": [1], "account_id": [42], "amount_usd": [50.0]})
    accounts = pd.DataFrame({"account_id": [42], "prior_chargebacks": [3]})
    result = build_model_frame(txns, accounts)
    assert result.iloc[0]["prior_chargebacks"] == 3


# --- summarize_results: sort order ---

def test_summary_sort_order_is_high_medium_low():
    summary = summarize_results(_scored(), _chargebacks())
    assert list(summary["risk_label"].astype(str)) == ["high", "medium", "low"]


# --- summarize_results: counts and amounts ---

def test_summary_transaction_counts():
    summary = summarize_results(_scored(), _chargebacks())
    counts = dict(zip(summary["risk_label"].astype(str), summary["transactions"]))
    assert counts == {"high": 2, "medium": 2, "low": 1}


def test_summary_total_and_avg_amount():
    summary = summarize_results(_scored(), _chargebacks())
    high = summary[summary["risk_label"].astype(str) == "high"].iloc[0]
    assert high["total_amount_usd"] == pytest.approx(300.0)   # 100 + 200
    assert high["avg_amount_usd"] == pytest.approx(150.0)     # 300 / 2


# --- summarize_results: chargeback_rate ---

def test_chargeback_rate_full_bucket():
    summary = summarize_results(_scored(), _chargebacks("T1", "T2"))
    high = summary[summary["risk_label"].astype(str) == "high"].iloc[0]
    assert high["chargeback_rate"] == pytest.approx(1.0)


def test_chargeback_rate_partial_bucket():
    summary = summarize_results(_scored(), _chargebacks("T3"))
    medium = summary[summary["risk_label"].astype(str) == "medium"].iloc[0]
    assert medium["chargeback_rate"] == pytest.approx(0.5)


def test_chargeback_rate_zero_for_clean_bucket():
    summary = summarize_results(_scored(), _chargebacks())
    low = summary[summary["risk_label"].astype(str) == "low"].iloc[0]
    assert low["chargeback_rate"] == pytest.approx(0.0)


# --- Integration: full pipeline against real CSVs ---

def test_confirmed_fraud_not_labeled_low_risk():
    """No confirmed chargeback should slip into the low-risk bucket."""
    accounts, transactions, chargebacks = load_inputs()
    scored = score_transactions(transactions, accounts)
    chargeback_ids = set(chargebacks["transaction_id"])
    fraud_rows = scored[scored["transaction_id"].isin(chargeback_ids)]
    low_risk_fraud = fraud_rows[fraud_rows["risk_label"] == "low"]
    assert len(low_risk_fraud) == 0, (
        f"Confirmed fraud transactions scored low risk: {list(low_risk_fraud['transaction_id'])}"
    )


def test_chargeback_rate_decreases_with_lower_risk():
    """High-risk bucket must have a higher chargeback rate than low-risk."""
    accounts, transactions, chargebacks = load_inputs()
    scored = score_transactions(transactions, accounts)
    summary = summarize_results(scored, chargebacks)
    rates = dict(zip(summary["risk_label"].astype(str), summary["chargeback_rate"]))
    assert rates["high"] > rates["low"]


def test_all_chargebacks_accounted_for_in_summary():
    """The sum of chargebacks across all risk buckets must equal total known fraud."""
    accounts, transactions, chargebacks = load_inputs()
    scored = score_transactions(transactions, accounts)
    summary = summarize_results(scored, chargebacks)
    assert summary["chargebacks"].sum() == len(chargebacks)
