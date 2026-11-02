"""Atomic output with bounded retries for transient Windows reader/scanner locks."""
import json
import os
from pathlib import Path
import time
import uuid
from contextlib import contextmanager


def atomic_text(path, content):
    path = Path(path)
    temporary = path.with_name(f'{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    try:
        with temporary.open('w', encoding='utf-8') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(50):
            try:
                temporary.replace(path)
                return
            except PermissionError:
                if attempt == 49:
                    raise
                time.sleep(min(.05*(attempt+1), .5))
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path, data):
    path = Path(path)
    if path.name == 'checkpoint.json' and path.exists():
        previous = path.read_text(encoding='utf-8')
        # Preserve the preceding valid checkpoint before committing the next one.
        json.loads(previous)
        atomic_text(path.with_name('checkpoint.previous.json'), previous)
    atomic_text(path, json.dumps(data, indent=2, allow_nan=False))


@contextmanager
def exclusive_lock(path):
    """Keep a single coordinator; Windows releases this lock after a crash/reboot."""
    import msvcrt
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError('Another coordinator holds the pilot lock.') from exc
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
