#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for the ticker-universe resolution added to
``scripts/factor_verdict_screen.py`` on 2026-09-08 (generalization of the
former ``russell1000_factor_verdict_screen.py``).

Covers:
  - Each of the four ticker sources in isolation (CLI list, CSV, JSON, file).
  - The documented precedence order when multiple sources are supplied.
  - The zero-ticker error path.
  - Backward compatibility: no new flags resolves through the unchanged
    default Russell 1000 file path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from scripts.factor_verdict_screen import (
    DEFAULT_UNIVERSE_FILE,
    load_tickers_csv,
    load_tickers_json,
    load_universe,
    parse_tickers_arg,
    resolve_ticker_universe,
)


def _args(**overrides) -> argparse.Namespace:
    """Build a Namespace with every resolve_ticker_universe field defaulted."""
    base = dict(
        tickers=None,
        tickers_csv=None,
        tickers_json=None,
        universe=str(DEFAULT_UNIVERSE_FILE),
        universe_name=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


# ----------------------------------------------------------------------
# parse_tickers_arg
# ----------------------------------------------------------------------

class TestParseTickersArg:
    def test_basic_comma_list(self):
        assert parse_tickers_arg("aapl,msft,nvda") == ["AAPL", "MSFT", "NVDA"]

    def test_strips_whitespace_and_dedups_preserving_order(self):
        assert parse_tickers_arg(" AAPL , msft, AAPL ,nvda") == ["AAPL", "MSFT", "NVDA"]

    def test_drops_blank_entries(self):
        assert parse_tickers_arg("AAPL,,MSFT,") == ["AAPL", "MSFT"]

    def test_empty_string_yields_empty_list(self):
        assert parse_tickers_arg("") == []


# ----------------------------------------------------------------------
# load_tickers_csv
# ----------------------------------------------------------------------

class TestLoadTickersCsv:
    def test_symbol_column(self, tmp_path):
        p = tmp_path / "t.csv"
        p.write_text("symbol,name\nAAPL,Apple\nMSFT,Microsoft\n", encoding="utf-8")
        assert load_tickers_csv(p) == ["AAPL", "MSFT"]

    def test_ticker_column_case_insensitive(self, tmp_path):
        p = tmp_path / "t.csv"
        p.write_text("Ticker\naapl\nnvda\n", encoding="utf-8")
        assert load_tickers_csv(p) == ["AAPL", "NVDA"]

    def test_single_column_with_generic_header(self, tmp_path):
        p = tmp_path / "t.csv"
        p.write_text("watchlist\nAAPL\nMSFT\n", encoding="utf-8")
        assert load_tickers_csv(p) == ["AAPL", "MSFT"]

    def test_single_column_headerless_treats_header_cell_as_data(self, tmp_path):
        p = tmp_path / "t.csv"
        p.write_text("AAPL\nMSFT\nNVDA\n", encoding="utf-8")
        assert load_tickers_csv(p) == ["AAPL", "MSFT", "NVDA"]

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_tickers_csv(tmp_path / "does_not_exist.csv")

    def test_ambiguous_multi_column_raises(self, tmp_path):
        p = tmp_path / "t.csv"
        p.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_tickers_csv(p)


# ----------------------------------------------------------------------
# load_tickers_json
# ----------------------------------------------------------------------

class TestLoadTickersJson:
    def test_bare_array(self, tmp_path):
        p = tmp_path / "t.json"
        p.write_text(json.dumps(["aapl", "msft"]), encoding="utf-8")
        assert load_tickers_json(p) == ["AAPL", "MSFT"]

    def test_object_with_tickers_key(self, tmp_path):
        p = tmp_path / "t.json"
        p.write_text(json.dumps({"tickers": ["aapl", "nvda"]}), encoding="utf-8")
        assert load_tickers_json(p) == ["AAPL", "NVDA"]

    def test_object_with_symbols_key(self, tmp_path):
        p = tmp_path / "t.json"
        p.write_text(json.dumps({"symbols": ["aapl", "nvda"]}), encoding="utf-8")
        assert load_tickers_json(p) == ["AAPL", "NVDA"]

    def test_object_without_recognized_key_raises(self, tmp_path):
        p = tmp_path / "t.json"
        p.write_text(json.dumps({"names": ["aapl"]}), encoding="utf-8")
        with pytest.raises(ValueError):
            load_tickers_json(p)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_tickers_json(tmp_path / "does_not_exist.json")

    def test_non_list_non_dict_raises(self, tmp_path):
        p = tmp_path / "t.json"
        p.write_text(json.dumps("not-a-list"), encoding="utf-8")
        with pytest.raises(ValueError):
            load_tickers_json(p)


# ----------------------------------------------------------------------
# resolve_ticker_universe -- precedence, labeling, and error paths
# ----------------------------------------------------------------------

class TestResolveTickerUniverse:
    def test_default_no_flags_uses_russell1000_file(self):
        symbols, label, display = resolve_ticker_universe(_args())
        assert label == "russell1000"
        assert display == "Russell 1000"
        assert symbols == load_universe(DEFAULT_UNIVERSE_FILE)
        assert len(symbols) > 0

    def test_tickers_flag_highest_precedence(self, tmp_path):
        csv_p = tmp_path / "ignored.csv"
        csv_p.write_text("symbol\nIGNORED\n", encoding="utf-8")
        args = _args(tickers="AAPL,MSFT", tickers_csv=str(csv_p))
        symbols, label, display = resolve_ticker_universe(args)
        assert symbols == ["AAPL", "MSFT"]
        assert label == "custom"
        assert "2 tickers" in display

    def test_tickers_csv_beats_tickers_json(self, tmp_path):
        csv_p = tmp_path / "watch.csv"
        csv_p.write_text("symbol\nAAPL\n", encoding="utf-8")
        json_p = tmp_path / "ignored.json"
        json_p.write_text(json.dumps(["IGNORED"]), encoding="utf-8")
        args = _args(tickers_csv=str(csv_p), tickers_json=str(json_p))
        symbols, label, display = resolve_ticker_universe(args)
        assert symbols == ["AAPL"]
        assert label == "watch"

    def test_tickers_json_used_when_only_source(self, tmp_path):
        json_p = tmp_path / "basket.json"
        json_p.write_text(json.dumps({"tickers": ["nvda"]}), encoding="utf-8")
        args = _args(tickers_json=str(json_p))
        symbols, label, display = resolve_ticker_universe(args)
        assert symbols == ["NVDA"]
        assert label == "basket"

    def test_universe_name_override_applies_to_label_and_display(self, tmp_path):
        args = _args(tickers="AAPL,MSFT", universe_name="Core Longs")
        symbols, label, display = resolve_ticker_universe(args)
        assert display == "Core Longs"
        assert label == "core_longs"

    def test_custom_universe_file_derives_label_from_stem(self, tmp_path):
        p = tmp_path / "nasdaq100.txt"
        p.write_text("AAPL\t2015-01-01\t2026-09-06\nMSFT\t2015-01-01\t2026-09-06\n", encoding="utf-8")
        args = _args(universe=str(p))
        symbols, label, display = resolve_ticker_universe(args)
        assert symbols == ["AAPL", "MSFT"]
        assert label == "nasdaq100"
        assert display == "Nasdaq100"

    def test_empty_tickers_flag_raises_value_error(self):
        args = _args(tickers="  ,  ,")
        with pytest.raises(ValueError):
            resolve_ticker_universe(args)

    def test_empty_csv_raises_value_error(self, tmp_path):
        p = tmp_path / "empty.csv"
        p.write_text("symbol\n", encoding="utf-8")
        args = _args(tickers_csv=str(p))
        with pytest.raises(ValueError):
            resolve_ticker_universe(args)

    def test_empty_json_array_raises_value_error(self, tmp_path):
        p = tmp_path / "empty.json"
        p.write_text(json.dumps([]), encoding="utf-8")
        args = _args(tickers_json=str(p))
        with pytest.raises(ValueError):
            resolve_ticker_universe(args)
