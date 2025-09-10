"""The redundancy and duplicate rules, on synthetic index metadata.

Deciding that one index covers another is the only place in this repository
where dropping an object is suggested, so the cases where it must *not* fire get
as much attention as the ones where it must.
"""

from __future__ import annotations

from pgtk.indexes import IndexInfo, covers, find_duplicates, find_redundant


def index(
    name: str,
    *,
    keys: tuple[str, ...],
    table: str = "orders",
    am: str = "btree",
    include: tuple[str, ...] = (),
    predicate: str = "",
    expressions: str = "",
    unique: bool = False,
    primary: bool = False,
    replident: bool = False,
    constraint: bool = False,
    size: int = 4 * 1024 * 1024,
    directions: tuple[str, ...] | None = None,
) -> IndexInfo:
    options = directions or tuple("0" for _ in keys)
    return IndexInfo(
        schema="public",
        table=table,
        name=name,
        amname=am,
        key_columns=tuple((key, "1978", options[i]) for i, key in enumerate(keys)),
        all_columns=frozenset(keys) | frozenset(include),
        predicate=predicate,
        expressions=expressions,
        is_unique=unique,
        is_primary=primary,
        is_replident=replident,
        is_valid=True,
        is_ready=True,
        backs_constraint=constraint,
        size_bytes=size,
        definition=f"CREATE INDEX {name} ON public.{table} ({', '.join(keys)})",
        idx_scan=0,
        last_idx_scan=None,
    )


def test_a_prefix_is_covered_by_the_wider_index() -> None:
    narrow = index("orders_a", keys=("1",))
    wide = index("orders_ab", keys=("1", "2"))
    assert covers(wide, narrow)
    assert not covers(narrow, wide)


def test_identical_key_lists_do_not_cover_each_other() -> None:
    """Two identical indexes are duplicates, and that is a different finding."""
    first = index("orders_a", keys=("1",))
    second = index("orders_a_copy", keys=("1",))
    assert not covers(first, second)
    assert not covers(second, first)


def test_a_suffix_is_not_a_prefix() -> None:
    assert not covers(index("orders_ab", keys=("1", "2")), index("orders_b", keys=("2",)))


def test_a_descending_leading_column_is_a_different_access_path() -> None:
    ascending = index("orders_a", keys=("1",), directions=("0",))
    descending = index("orders_a_desc_b", keys=("1", "2"), directions=("3", "0"))
    assert not covers(descending, ascending)


def test_a_partial_index_never_covers_an_unconditional_one() -> None:
    partial = index("orders_ab_partial", keys=("1", "2"), predicate="(status = 'open')")
    full = index("orders_a", keys=("1",))
    assert not covers(partial, full)


def test_a_gin_index_does_not_cover_a_btree_prefix() -> None:
    assert not covers(index("orders_ab", keys=("1", "2"), am="gin"), index("orders_a", keys=("1",)))


def test_an_index_on_another_table_never_covers() -> None:
    assert not covers(
        index("other_ab", keys=("1", "2"), table="events"), index("orders_a", keys=("1",))
    )


def test_losing_an_include_column_blocks_the_recommendation() -> None:
    """Dropping the narrower index would turn an index-only scan into a heap access."""
    narrow = index("orders_a_inc_total", keys=("1",), include=("5",))
    wide = index("orders_ab", keys=("1", "2"))
    assert not covers(wide, narrow)
    assert covers(index("orders_ab_inc_total", keys=("1", "2"), include=("5",)), narrow)


def test_constraint_backed_indexes_are_never_reported_as_redundant() -> None:
    protected = [
        index("orders_pkey", keys=("1",), unique=True, primary=True, constraint=True),
        index("orders_a_uniq", keys=("1",), unique=True, constraint=True),
        index("orders_a_replident", keys=("1",), replident=True),
    ]
    wide = index("orders_ab", keys=("1", "2"))
    for candidate in protected:
        assert find_redundant([candidate, wide]) == []


def test_redundancy_reports_the_narrow_index_and_names_its_cover() -> None:
    narrow = index("orders_a", keys=("1",))
    wide = index("orders_ab", keys=("1", "2"))
    assert find_redundant([narrow, wide]) == [(narrow, wide)]


def test_duplicates_are_grouped_and_the_constraint_backed_copy_is_kept() -> None:
    plain = index("orders_a_idx", keys=("1",), size=8 * 1024 * 1024)
    enforcing = index("orders_a_key", keys=("1",), unique=True, constraint=True, size=1024)
    groups = find_duplicates([plain, enforcing])
    assert len(groups) == 1
    assert groups[0][0] is enforcing


def test_without_a_constraint_the_larger_copy_is_kept() -> None:
    big = index("orders_a_big", keys=("1",), size=9_000_000)
    small = index("orders_a_small", keys=("1",), size=1_000_000)
    assert find_duplicates([small, big])[0][0] is big


def test_indexes_that_differ_only_by_predicate_are_not_duplicates() -> None:
    everything = index("orders_a", keys=("1",))
    open_only = index("orders_a_open", keys=("1",), predicate="(status = 'open')")
    assert find_duplicates([everything, open_only]) == []


def test_expression_indexes_are_duplicates_only_when_the_expression_matches() -> None:
    lower_one = index("orders_lower_a", keys=("0",), expressions="lower(region)")
    lower_two = index("orders_lower_a_copy", keys=("0",), expressions="lower(region)")
    upper = index("orders_upper_a", keys=("0",), expressions="upper(region)")
    groups = find_duplicates([lower_one, lower_two, upper])
    assert len(groups) == 1
    assert {i.name for i in groups[0]} == {"orders_lower_a", "orders_lower_a_copy"}
