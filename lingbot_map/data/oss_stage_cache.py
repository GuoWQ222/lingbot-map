"""On-demand CPFS staging for OSS-backed trajectory directories."""

from __future__ import annotations

import errno
import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple


_MOUNT_ERRNOS = {
    errno.ENOTCONN,
    errno.EIO,
    errno.ESTALE,
    errno.ETIMEDOUT,
}


def _safe_relative_path(value: str) -> Path:
    parts = []
    for part in Path(str(value).replace("\\", "/")).parts:
        if part in ("", ".", "..", os.sep):
            continue
        parts.append(part)
    if not parts:
        raise ValueError(f"empty cache key from {value!r}")
    return Path(*parts)


def _exc_contains_mount_error(exc: BaseException) -> bool:
    current: Optional[BaseException] = exc
    while current is not None:
        if isinstance(current, OSError) and current.errno in _MOUNT_ERRNOS:
            return True
        text = str(current)
        if "Transport endpoint is not connected" in text:
            return True
        if "short pread" in text:
            return True
        current = current.__cause__ or current.__context__
    return False


def _exc_mentions_path_under(exc: BaseException, root: Path) -> bool:
    root_text = str(root)
    current: Optional[BaseException] = exc
    while current is not None:
        if root_text and root_text in str(current):
            return True
        current = current.__cause__ or current.__context__
    return False


DEFAULT_RCLONE_BIN = "/cpfs/user/guowenqi/rclone/rclone"
DEFAULT_RCLONE_CONFIG = "/cpfs/user/guowenqi/rclone/rclone.conf"


def _rclone_command(rclone_bin: str, rclone_config: str, *args: str) -> list[str]:
    command = [str(rclone_bin)]
    if rclone_config:
        command.extend(["--config", str(rclone_config)])
    command.extend(args)
    return command


def _oss_uri_to_rclone_path(oss_uri: str, remote: str) -> str:
    uri = str(oss_uri).strip().rstrip("/")
    if not uri.startswith("oss://"):
        raise ValueError(f"expected oss:// URI, got {oss_uri!r}")
    remote_name = str(remote or "aliyunoss").rstrip(":")
    return f"{remote_name}:{uri[len('oss://'):]}"


class _DirLock:
    def __init__(self, path: Path, *, stale_seconds: int = 6 * 60 * 60) -> None:
        self.path = path
        self.stale_seconds = stale_seconds
        self.acquired = False

    def __enter__(self) -> "_DirLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                self.path.mkdir()
                self.acquired = True
                (self.path / "pid").write_text(str(os.getpid()))
                return self
            except FileExistsError:
                try:
                    age = time.time() - self.path.stat().st_mtime
                    if age > self.stale_seconds:
                        shutil.rmtree(self.path, ignore_errors=True)
                        continue
                except FileNotFoundError:
                    continue
                time.sleep(0.2)

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.acquired:
            shutil.rmtree(self.path, ignore_errors=True)


class OssStageCache:
    """Stage OSS directories to CPFS after the OSS FUSE mount becomes unhealthy."""

    def __init__(
        self,
        *,
        stage_root: os.PathLike[str] | str,
        mount_root: os.PathLike[str] | str = "/oss-guowenqi",
        ossutil_bin: os.PathLike[str] | str = "ossutil",
        ossutil_config: str = "",
        max_entries: int = 1000,
        enabled: bool = False,
        ossutil_jobs: int = 64,
        ossutil_checkers: int = 128,
        ossutil_parallel: int = 4,
        delete_workers: int = 8,
        delete_batch: int = 512,
        worker_max_entries: int = 25,
        rclone_bin: os.PathLike[str] | str = DEFAULT_RCLONE_BIN,
        rclone_config: os.PathLike[str] | str = DEFAULT_RCLONE_CONFIG,
        rclone_remote: str = "aliyunoss",
        rclone_transfers: Optional[int] = None,
        rclone_checkers: Optional[int] = None,
    ) -> None:
        self.stage_root = Path(stage_root).expanduser()
        self.mount_root = Path(mount_root).expanduser()
        self.ossutil_bin = str(ossutil_bin)
        self.ossutil_config = str(ossutil_config or "")
        self.max_entries = max(1, int(max_entries))
        self.enabled = bool(enabled)
        self.ossutil_jobs = max(1, int(ossutil_jobs))
        self.ossutil_checkers = max(1, int(ossutil_checkers))
        self.ossutil_parallel = max(1, int(ossutil_parallel))
        self.delete_workers = max(1, int(delete_workers))
        self.delete_batch = max(1, int(delete_batch))
        self.worker_max_entries = max(1, int(worker_max_entries))
        # rclone_* parameters are accepted for backward CLI compatibility but ossutil performs staging.
        del rclone_bin, rclone_config, rclone_remote, rclone_transfers, rclone_checkers
        self._fallback_marker = self.stage_root / ".fallback_enabled"

    def _worker_id(self) -> Optional[int]:
        try:
            from torch.utils.data import get_worker_info
        except Exception:
            return None
        info = get_worker_info()
        return None if info is None else int(info.id)

    def _rank_stage_root(self) -> Path:
        rank = os.environ.get("RANK")
        if rank is None:
            return self.stage_root
        rank_name = f"rank{rank}"
        if self.stage_root.name == rank_name:
            return self.stage_root
        return self.stage_root / rank_name

    def _active_stage_root(self) -> Path:
        root = self._rank_stage_root()
        worker_id = self._worker_id()
        if worker_id is None:
            return root
        return root / f"worker{worker_id}"

    def _active_max_entries(self) -> int:
        return self.worker_max_entries if self._worker_id() is not None else self.max_entries

    def _rebase_to_active_stage_root(self, path: Path) -> Path:
        try:
            rel = path.relative_to(self.stage_root)
        except ValueError:
            return path
        return self._active_stage_root() / rel

    @property
    def _locks_root(self) -> Path:
        return self._active_stage_root() / ".locks"

    @property
    def _tmp_root(self) -> Path:
        return self._active_stage_root() / ".tmp"

    @property
    def fallback_enabled(self) -> bool:
        return self.enabled and self._fallback_marker.exists()

    def enable_fallback(self, reason: str) -> None:
        if not self.enabled:
            return
        self.stage_root.mkdir(parents=True, exist_ok=True)
        payload = {
            "pid": os.getpid(),
            "time": time.time(),
            "reason": str(reason),
        }
        try:
            self._fallback_marker.write_text(json.dumps(payload, sort_keys=True))
        except OSError:
            pass

    def enable_fallback_from_exception(
        self,
        exc: BaseException,
        *,
        source_path: Optional[os.PathLike[str] | str] = None,
    ) -> bool:
        if not self.enabled or not _exc_contains_mount_error(exc):
            return False
        if _exc_mentions_path_under(exc, self.stage_root):
            return False
        if not _exc_mentions_path_under(exc, self.mount_root):
            if source_path is None:
                return False
            try:
                Path(source_path).relative_to(self.mount_root)
            except ValueError:
                return False
        self.enable_fallback(str(exc))
        return True

    def mount_accessible(self) -> bool:
        if not self.enabled:
            return True
        try:
            os.stat(self.mount_root)
            return True
        except OSError as exc:
            if exc.errno in _MOUNT_ERRNOS or "Transport endpoint is not connected" in str(exc):
                self.enable_fallback(str(exc))
                return False
            raise

    def should_stage(self, source_path: os.PathLike[str] | str) -> bool:
        if not self.enabled:
            return False
        source = Path(source_path)
        try:
            source.relative_to(self.mount_root)
        except ValueError:
            return False
        if self.fallback_enabled:
            return True
        return not self.mount_accessible()

    def resolve_dir(
        self,
        *,
        source_path: os.PathLike[str] | str,
        dataset: str,
        relative_key: str,
        oss_uri: str,
        dest_root: os.PathLike[str] | str,
    ) -> Path:
        source = Path(source_path)
        if not self.should_stage(source):
            return source
        return self.stage_dir(
            dataset=dataset,
            relative_key=relative_key,
            oss_uri=oss_uri,
            dest_root=dest_root,
        )

    def stage_dir(
        self,
        *,
        dataset: str,
        relative_key: str,
        oss_uri: str,
        dest_root: Optional[os.PathLike[str] | str] = None,
        count_entry: bool = True,
    ) -> Path:
        if not self.enabled:
            raise RuntimeError("OssStageCache is disabled")
        rel = _safe_relative_path(relative_key)
        root = Path(dest_root).expanduser() if dest_root is not None else self.stage_root / "entries" / dataset
        root = self._rebase_to_active_stage_root(root)
        final_dir = root / rel
        lock_name = f"{dataset}_{'_'.join(rel.parts)}.lock"
        lock_path = self._locks_root / lock_name
        with _DirLock(lock_path):
            if self._is_complete(final_dir):
                self._touch_access(final_dir)
                return final_dir
            if count_entry:
                self._evict_until_below_limit(skip=final_dir)
            if final_dir.exists():
                self._remove_tree(final_dir)
            self._tmp_root.mkdir(parents=True, exist_ok=True)
            tmp_dir = self._tmp_root / f"{dataset}_{os.getpid()}_{uuid.uuid4().hex}"
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            tmp_dir.mkdir(parents=True)
            try:
                self._copy_from_oss(oss_uri, tmp_dir)
                self._write_meta(tmp_dir, dataset=dataset, relative_key=str(rel), oss_uri=oss_uri, count_entry=count_entry)
                final_dir.parent.mkdir(parents=True, exist_ok=True)
                os.replace(tmp_dir, final_dir)
                self._touch_access(final_dir)
            except Exception:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                raise
        return final_dir

    def _copy_from_oss(self, oss_uri: str, dest_dir: Path) -> None:
        source = str(oss_uri).strip().rstrip("/") + "/"
        if not source.startswith("oss://"):
            raise ValueError(f"expected oss:// URI, got {oss_uri!r}")
        cmd = [self.ossutil_bin, "cp"]
        if self.ossutil_config:
            cmd.extend(["-c", self.ossutil_config])
        cmd.extend([
            source,
            str(dest_dir),
            "-r",
            "-f",
            "-j",
            str(self.ossutil_jobs),
            "--checkers",
            str(self.ossutil_checkers),
            "--parallel",
            str(self.ossutil_parallel),
            "--no-progress",
        ])
        print(
            f"[oss_stage][ossutil] cp source={source} dest={dest_dir} "
            f"jobs={self.ossutil_jobs} checkers={self.ossutil_checkers} parallel={self.ossutil_parallel}",
            flush=True,
        )
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            print(
                f"[oss_stage][ossutil][error] returncode={exc.returncode} "
                f"source={source} dest={dest_dir}",
                flush=True,
            )
            raise

    @staticmethod
    def _is_complete(path: Path) -> bool:
        return path.is_dir() and (path / ".oss_stage_complete").is_file()

    @staticmethod
    def _touch_access(path: Path) -> None:
        now = time.time()
        marker = path / ".oss_stage_last_access"
        marker.touch(exist_ok=True)
        os.utime(marker, (now, now))

    @staticmethod
    def _write_meta(
        path: Path,
        *,
        dataset: str,
        relative_key: str,
        oss_uri: str,
        count_entry: bool,
    ) -> None:
        meta = {
            "dataset": dataset,
            "relative_key": relative_key,
            "oss_uri": oss_uri,
            "count_entry": bool(count_entry),
            "pid": os.getpid(),
            "time": time.time(),
        }
        (path / ".oss_stage_meta.json").write_text(json.dumps(meta, sort_keys=True))
        (path / ".oss_stage_complete").write_text("ok\n")

    def _iter_entries(self) -> Iterator[Tuple[float, Path]]:
        root = self._active_stage_root()
        if not root.exists():
            return
        for complete in root.rglob(".oss_stage_complete"):
            entry_dir = complete.parent
            meta_path = entry_dir / ".oss_stage_meta.json"
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                continue
            if not meta.get("count_entry", True):
                continue
            access = entry_dir / ".oss_stage_last_access"
            try:
                stamp = access.stat().st_mtime
            except OSError:
                stamp = complete.stat().st_mtime
            yield stamp, entry_dir

    def _evict_until_below_limit(self, *, skip: Path) -> None:
        # LRU replacement keeps recently reused staged trajectories and caps each worker cache.
        with _DirLock(self._locks_root / ".evict.lock"):
            limit = self._active_max_entries()
            entries = [(stamp, path) for stamp, path in self._iter_entries() if path != skip]
            while len(entries) >= limit:
                _, victim = min(entries, key=lambda item: item[0])
                self._remove_tree(victim, ignore_errors=True)
                entries = [(stamp, path) for stamp, path in entries if path != victim]

    def _remove_tree(self, path: Path, *, ignore_errors: bool = False) -> None:
        if not path.exists():
            return
        if self.delete_workers <= 1 or not path.is_dir():
            shutil.rmtree(path, ignore_errors=ignore_errors)
            return
        if not all(shutil.which(tool) for tool in ("find", "xargs", "rm")):
            shutil.rmtree(path, ignore_errors=ignore_errors)
            return
        try:
            self._parallel_unlink_files(path)
            shutil.rmtree(path, ignore_errors=ignore_errors)
        except Exception as exc:
            if not ignore_errors:
                try:
                    shutil.rmtree(path)
                    return
                except FileNotFoundError:
                    return
            print(f"[oss_stage][evict] parallel delete fallback path={path} error={exc}", flush=True)
            shutil.rmtree(path, ignore_errors=True)

    def _parallel_unlink_files(self, path: Path) -> None:
        find_proc = subprocess.Popen(
            ["find", str(path), "-type", "f", "-print0"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert find_proc.stdout is not None
        xargs_proc = subprocess.run(
            [
                "xargs",
                "-0",
                "-r",
                "-n",
                str(self.delete_batch),
                "-P",
                str(self.delete_workers),
                "rm",
                "-f",
                "--",
            ],
            stdin=find_proc.stdout,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        find_proc.stdout.close()
        find_stderr = find_proc.stderr.read().decode("utf-8", errors="replace") if find_proc.stderr else ""
        find_return = find_proc.wait()
        if find_return != 0 or xargs_proc.returncode != 0:
            stderr = (find_stderr + "\n" + (xargs_proc.stderr or "")).strip()
            raise RuntimeError(
                f"parallel delete failed find={find_return} xargs={xargs_proc.returncode}: {stderr}"
            )


def build_oss_stage_cache(
    *,
    enabled: bool,
    stage_root: str,
    mount_root: str,
    ossutil_bin: str,
    ossutil_config: str,
    max_entries: int,
    ossutil_jobs: int = 16,
    ossutil_checkers: int = 32,
    ossutil_parallel: int = 4,
    delete_workers: int = 8,
    delete_batch: int = 512,
    worker_max_entries: int = 25,
    rclone_bin: str = DEFAULT_RCLONE_BIN,
    rclone_config: str = DEFAULT_RCLONE_CONFIG,
    rclone_remote: str = "aliyunoss",
    rclone_transfers: Optional[int] = None,
    rclone_checkers: Optional[int] = None,
) -> Optional[OssStageCache]:
    if not enabled:
        return None
    return OssStageCache(
        stage_root=stage_root,
        mount_root=mount_root,
        ossutil_bin=ossutil_bin,
        ossutil_config=ossutil_config,
        max_entries=max_entries,
        enabled=True,
        ossutil_jobs=ossutil_jobs,
        ossutil_checkers=ossutil_checkers,
        ossutil_parallel=ossutil_parallel,
        delete_workers=delete_workers,
        delete_batch=delete_batch,
        worker_max_entries=worker_max_entries,
        rclone_bin=rclone_bin,
        rclone_config=rclone_config,
        rclone_remote=rclone_remote,
        rclone_transfers=rclone_transfers,
        rclone_checkers=rclone_checkers,
    )
