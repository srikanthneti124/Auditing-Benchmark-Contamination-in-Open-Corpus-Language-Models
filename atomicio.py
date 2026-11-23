"""Atomic file replacement that tolerates transient locks (OneDrive sync, antivirus, an editor).

replace_with_retry(tmp, dst): os.replace(tmp, dst), retried up to RETRIES times with a short
sleep on PermissionError. If it still fails, the .tmp file is KEPT (it holds the newest data),
a warning is printed to stderr and appended to WARN_LOG, and False is returned. Nothing is
deleted; the caller's in-memory data is never dropped silently.

recover_tmp(path): JSON written to <path>.tmp by a save whose replace failed; returned (and
logged) so the caller can merge it on the next start. None if absent or unreadable.
"""
import json
import os
import sys
import time

RETRIES = 10
DELAY = 0.2  # seconds between attempts
WARN_LOG = "atomic_save_warnings.log"


def warn(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{msg}"
    print(f"WARNING: {msg}", file=sys.stderr, flush=True)
    try:
        with open(WARN_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError as e:  # the log itself is locked: stderr already has the message
        print(f"WARNING: could not append to {WARN_LOG}: {e}", file=sys.stderr, flush=True)


def replace_with_retry(tmp, dst, retries=RETRIES, delay=DELAY):
    err = None
    for attempt in range(retries):
        try:
            os.replace(tmp, dst)
            return True
        except PermissionError as e:
            err = e
            time.sleep(delay)
    warn(f"could not replace {dst} after {retries} attempts ({err}); newest data kept in {tmp} "
         f"(it is merged automatically on the next run; do not delete it)")
    return False


def write_json_atomic(path, obj, **dump_kw):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, **dump_kw)
    return replace_with_retry(tmp, path)


def recover_tmp(path):
    tmp = path + ".tmp"
    if not os.path.exists(tmp):
        return None
    try:
        with open(tmp, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        warn(f"ignoring unreadable {tmp} ({type(e).__name__}: {e}); it was not deleted")
        return None
    warn(f"recovered {len(data)} entries from {tmp} left by an earlier save whose replace failed")
    return data
