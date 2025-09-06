#!/usr/bin/env bash
#
# A point-in-time recovery that is actually performed, start to finish.
#
# The script starts a throwaway PostgreSQL with WAL archiving on, takes a base
# backup, writes one row, records the moment, writes a second row, and then
# restores the base backup into a second data directory with a recovery target
# between the two writes. It ends by proving the obvious thing: the first row is
# there and the second one is not.
#
# The second cluster runs inside the same container on another port, so the
# original never has to be stopped and the two can be compared side by side.
#
# Usage: backup/pitr.sh [--keep]
set -euo pipefail

# Git Bash rewrites arguments that look like Unix paths before handing them to
# docker.exe, which turns /var/lib/postgresql into a drive letter. The container
# paths have to survive verbatim. Both variables are ignored everywhere else.
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*'

IMAGE="${PGTK_PITR_IMAGE:-postgres:17-alpine}"
CONTAINER="${PGTK_PITR_CONTAINER:-pgtk-pitr}"
DB="pitr_demo"
ARCHIVE="/var/lib/postgresql/archive"
BASE="/var/lib/postgresql/basebackup"
RESTORED="/var/lib/postgresql/restored"
RESTORE_PORT=5433
KEEP=0
[ "${1:-}" = "--keep" ] && KEEP=1

step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
pg() { docker exec -u postgres "$CONTAINER" psql -qAtX -v ON_ERROR_STOP=1 -d "$DB" -c "$1"; }
pg_restored() {
    docker exec -u postgres "$CONTAINER" \
        psql -qAtX -v ON_ERROR_STOP=1 -p "$RESTORE_PORT" -d "$DB" -c "$1"
}

cleanup() {
    if [ "$KEEP" -eq 0 ]; then
        docker rm -f -v "$CONTAINER" >/dev/null 2>&1 || true
    else
        echo "container $CONTAINER left running (--keep)"
    fi
}
trap cleanup EXIT

docker rm -f -v "$CONTAINER" >/dev/null 2>&1 || true

step "start a cluster with WAL archiving on"
# archive_command runs through a shell with the data directory as its working
# directory, so %p is usable directly. The mkdir keeps the archive location
# self-contained instead of requiring a volume prepared in advance.
docker run -d --name "$CONTAINER" \
    -e POSTGRES_PASSWORD=pitr_demo_password \
    -e POSTGRES_DB="$DB" \
    -e TZ=UTC \
    "$IMAGE" \
    postgres \
    -c wal_level=replica \
    -c archive_mode=on \
    -c archive_command="mkdir -p $ARCHIVE && test ! -f $ARCHIVE/%f && cp %p $ARCHIVE/%f" \
    -c max_wal_senders=4 \
    -c wal_keep_size=64MB >/dev/null

# The official image runs a throwaway server to initialise the cluster, then stops
# it and starts the real one. `pg_isready` answers "ready" against that throwaway
# server, so a bare readiness loop wins a race it does not know it is running and
# the next command lands in the shutdown window. Wait for the entrypoint to say
# initialisation finished before readiness is worth asking about at all.
for _ in $(seq 1 120); do
    docker logs "$CONTAINER" 2>&1 | grep -q 'init process complete' && break
    sleep 1
done

for _ in $(seq 1 120); do
    docker exec "$CONTAINER" pg_isready -q -U postgres -d "$DB" && break
    sleep 1
done

# One real query, so the check proves the server answers rather than merely listens.
until docker exec -u postgres "$CONTAINER" psql -qtAX -d "$DB" -c 'SELECT 1' >/dev/null 2>&1; do
    sleep 1
done
docker exec -u postgres "$CONTAINER" postgres --version

step "create the table this recovery is about"
pg "CREATE TABLE receipts (
        id         bigserial PRIMARY KEY,
        note       text NOT NULL,
        written_at timestamptz NOT NULL DEFAULT now()
    );"
pg "INSERT INTO receipts (note) VALUES ('seeded before the backup');"

step "base backup"
docker exec -u postgres "$CONTAINER" \
    pg_basebackup -D "$BASE" -Fp -Xstream -c fast -P -U postgres
docker exec -u postgres "$CONTAINER" sh -c "du -sh $BASE"

step "write one row, take the timestamp, write another"
pg "INSERT INTO receipts (note) VALUES ('written before the target');"
pg "SELECT pg_sleep(2);" >/dev/null
TARGET="$(pg 'SELECT now();')"
echo "recovery target: $TARGET"
pg "SELECT pg_sleep(2);" >/dev/null
pg "INSERT INTO receipts (note) VALUES ('written after the target');"

step "force the WAL holding both writes into the archive"
pg "SELECT pg_switch_wal();" >/dev/null
pg "CHECKPOINT;"
docker exec -u postgres "$CONTAINER" sh -c "ls -1 $ARCHIVE | head -20"

step "state of the live cluster"
docker exec -u postgres "$CONTAINER" \
    psql -X -d "$DB" -c "SELECT id, note, written_at FROM receipts ORDER BY id;"

step "restore the base backup into a second data directory"
docker exec -u postgres "$CONTAINER" sh -c "rm -rf $RESTORED && cp -a $BASE $RESTORED"
docker exec -u postgres "$CONTAINER" sh -c "cat >> $RESTORED/postgresql.auto.conf <<CONF
restore_command = 'cp $ARCHIVE/%f %p'
recovery_target_time = '$TARGET'
recovery_target_action = 'promote'
archive_mode = off
CONF"
docker exec -u postgres "$CONTAINER" touch "$RESTORED/recovery.signal"
docker exec -u postgres "$CONTAINER" \
    pg_ctl -D "$RESTORED" -o "-p $RESTORE_PORT" -l "$RESTORED/recovery.log" -w start

step "what the recovery said"
docker exec -u postgres "$CONTAINER" \
    grep -E 'starting point-in-time|consistent recovery state|recovery stopping|last completed transaction|database system is ready' \
    "$RESTORED/recovery.log"

step "state of the restored cluster"
docker exec -u postgres "$CONTAINER" \
    psql -X -p "$RESTORE_PORT" -d "$DB" -c "SELECT id, note, written_at FROM receipts ORDER BY id;"

step "verify"
BEFORE="$(pg_restored "SELECT count(*) FROM receipts WHERE note = 'written before the target';")"
AFTER="$(pg_restored "SELECT count(*) FROM receipts WHERE note = 'written after the target';")"
echo "rows written before the target, in the restored cluster: $BEFORE (want 1)"
echo "rows written after the target,  in the restored cluster: $AFTER (want 0)"

if [ "$BEFORE" != "1" ] || [ "$AFTER" != "0" ]; then
    echo "FAILED: the restored cluster does not match the recovery target" >&2
    exit 1
fi
printf '\n\033[1;32mPITR round trip verified\033[0m\n'
