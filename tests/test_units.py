"""Pure functions: version decoding, identifier quoting, thresholds, severity ladders."""

from __future__ import annotations

import json

import pytest

from pgtk.db import CATALOG_ADDITIONS, Capabilities, ServerVersion, qualified, quote_identifier
from pgtk.findings import Finding, Severity, human_bytes, sort_findings
from pgtk.locks import Session, build_blocking_forest
from pgtk.render import emit_json
from pgtk.vacuum import VacuumSettings, _age_severity, dead_tuple_threshold, parse_reloptions

SETTINGS = VacuumSettings(
    autovacuum_on=True,
    freeze_max_age=200_000_000,
    multixact_freeze_max_age=400_000_000,
    vacuum_threshold=50,
    vacuum_scale_factor=0.2,
    failsafe_age=1_600_000_000,
)


@pytest.mark.parametrize(
    ("num", "expected"),
    [(170_004, (17, 4)), (160_010, (16, 10)), (130_000, (13, 0)), (90_624, (9, 6))],
)
def test_server_version_num_decodes_across_the_ten_boundary(
    num: int, expected: tuple[int, int]
) -> None:
    version = ServerVersion.from_num(num)
    assert (version.major, version.minor) == expected


def test_versions_order_by_major_then_minor() -> None:
    assert ServerVersion(16, 9) < ServerVersion(17, 0) < ServerVersion(17, 4)


def test_asking_about_an_unregistered_column_is_an_error_not_a_false_negative() -> None:
    """A typo in a gate name must not silently disable a check."""
    caps = Capabilities(ServerVersion(17, 0), frozenset())
    with pytest.raises(KeyError, match="CATALOG_ADDITIONS"):
        caps.has("pg_stat_all_indexes.last_index_scan")


@pytest.mark.parametrize("column", sorted(CATALOG_ADDITIONS))
def test_every_registered_column_is_present_on_a_recent_server_and_absent_on_an_old_one(
    column: str,
) -> None:
    introduced = CATALOG_ADDITIONS[column]
    assert Capabilities(ServerVersion(introduced, 0), frozenset()).has(column)
    assert not Capabilities(ServerVersion(introduced - 1, 9), frozenset()).has(column)


def test_ordinary_identifiers_are_left_alone_and_awkward_ones_are_quoted() -> None:
    assert quote_identifier("orders") == "orders"
    assert quote_identifier("Orders") == '"Orders"'
    assert quote_identifier("order details") == '"order details"'
    assert quote_identifier('we"ird') == '"we""ird"'
    assert qualified("public", "orders") == "public.orders"


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0, "0 bytes"), (999, "999 bytes"), (1536, "1.5 kB"), (10 * 1024**2, "10.0 MB")],
)
def test_byte_formatting_matches_the_shape_psql_prints(value: int, expected: str) -> None:
    assert human_bytes(value) == expected


def test_reloptions_parse_into_the_knobs_autovacuum_actually_reads() -> None:
    expected = {"autovacuum_enabled": "false", "fillfactor": "70"}
    # The catalog hands them over space separated; psql prints them with commas.
    assert parse_reloptions("autovacuum_enabled=false fillfactor=70") == expected
    assert parse_reloptions("{autovacuum_enabled=false,fillfactor=70}") == expected
    assert parse_reloptions("") == {}


def test_a_per_table_scale_factor_overrides_the_cluster_default() -> None:
    default = dead_tuple_threshold(1_000_000, {}, SETTINGS)
    tuned = dead_tuple_threshold(1_000_000, {"autovacuum_vacuum_scale_factor": "0.01"}, SETTINGS)
    assert default == pytest.approx(200_050)
    assert tuned == pytest.approx(10_050)


@pytest.mark.parametrize(
    ("age", "expected"),
    [
        (10_000_000, Severity.NOTICE),
        (100_000_000, Severity.WARNING),
        (181_000_000, Severity.CRITICAL),
    ],
)
def test_the_wraparound_ladder_turns_a_share_of_freeze_max_age_into_a_severity(
    age: int, expected: Severity
) -> None:
    assert _age_severity(age, SETTINGS.freeze_max_age) is expected


def session(pid: int, blocked_by: tuple[int, ...] = (), state: str = "active") -> Session:
    return Session(
        pid=pid,
        user="app",
        database="shop",
        application=f"svc-{pid}",
        client="10.0.0.1",
        state=state,
        wait_event_type="Lock" if blocked_by else "",
        wait_event="transactionid" if blocked_by else "",
        xact_start=None,
        state_change=None,
        query_id=None,
        query=f"UPDATE t SET x = {pid}",
        blocked_by=blocked_by,
    )


def test_a_chain_becomes_one_tree_rooted_at_the_session_that_waits_for_nothing() -> None:
    forest = build_blocking_forest(
        [session(1), session(2, blocked_by=(1,)), session(3, blocked_by=(2,))]
    )
    assert len(forest) == 1
    root = forest[0]
    assert (root.session.pid, root.depth(), root.size()) == (1, 3, 3)


def test_two_independent_chains_stay_two_trees() -> None:
    forest = build_blocking_forest(
        [session(1), session(2, blocked_by=(1,)), session(10), session(11, blocked_by=(10,))]
    )
    assert [node.session.pid for node in forest] == [1, 10]


def test_sessions_that_block_nobody_are_absent_from_the_forest() -> None:
    assert build_blocking_forest([session(1), session(2)]) == []


def test_a_reported_cycle_terminates_instead_of_recursing_forever() -> None:
    """pg_blocking_pids can show a cycle before the deadlock detector fires."""
    forest = build_blocking_forest([session(1, blocked_by=(2,)), session(2, blocked_by=(1,))])
    assert forest == []


def test_a_blocker_that_is_itself_blocked_is_not_a_root() -> None:
    forest = build_blocking_forest(
        [session(1), session(2, blocked_by=(1,)), session(3, blocked_by=(1, 2))]
    )
    assert [node.session.pid for node in forest] == [1]


def test_a_session_blocked_by_two_others_is_counted_once() -> None:
    """pg_blocking_pids returns every blocker, so pid 3 hangs under 1 and under 2."""
    forest = build_blocking_forest(
        [session(1), session(2, blocked_by=(1,)), session(3, blocked_by=(1, 2))]
    )
    root = forest[0]
    assert root.pids() == {1, 2, 3}
    assert root.size() == 3
    assert root.depth() == 3


def test_findings_sort_worst_first_and_stay_stable_within_a_severity() -> None:
    findings = [
        Finding("b.check", Severity.NOTICE, "z", "s"),
        Finding("a.check", Severity.CRITICAL, "y", "s"),
        Finding("a.check", Severity.NOTICE, "a", "s"),
    ]
    assert [(f.check, f.subject) for f in sort_findings(findings)] == [
        ("a.check", "y"),
        ("a.check", "a"),
        ("b.check", "z"),
    ]


def test_json_output_carries_severity_as_a_word_and_keeps_extra_context() -> None:
    payload = json.loads(
        emit_json(
            [Finding("bloat.table", Severity.WARNING, "public.orders", "40% bloat")],
            server_version="17.4",
        )
    )
    assert payload["server_version"] == "17.4"
    assert payload["findings"][0]["severity"] == "warning"


def test_severity_compares_so_fail_on_thresholds_work() -> None:
    assert Severity.CRITICAL > Severity.WARNING > Severity.NOTICE > Severity.INFO
