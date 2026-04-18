from __future__ import annotations

import io
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

from echo_fraud_agents.models import DatasetBundle
from echo_fraud_agents.utils import file_basename, slugify


class DatasetLoader:
    def __init__(self, *, scan_nested_archives: bool = True, max_archive_depth: int = 2) -> None:
        self.scan_nested_archives = scan_nested_archives
        self.max_archive_depth = max_archive_depth

    def discover(self, input_path: Path) -> list[DatasetBundle]:
        path = input_path.resolve()
        if path.is_dir():
            bundles = self._discover_from_directory(path, depth=0)
        elif path.suffix.lower() == ".zip":
            bundles = self._discover_from_zip_bytes(path.read_bytes(), path.name, depth=0)
        else:
            raise RuntimeError(f"Unsupported input path: {path}")
        bundles.sort(key=lambda bundle: bundle.slug)
        return bundles

    def _discover_from_directory(self, root: Path, depth: int) -> list[DatasetBundle]:
        plain_files: dict[str, bytes] = {}
        bundles: list[DatasetBundle] = []
        for entry in root.rglob("*"):
            if not entry.is_file():
                continue
            relative = entry.relative_to(root).as_posix()
            if self._ignore_path(relative):
                continue
            if entry.suffix.lower() == ".zip" and self.scan_nested_archives and depth < self.max_archive_depth:
                bundles.extend(self._discover_from_zip_bytes(entry.read_bytes(), entry.name, depth + 1))
                continue
            plain_files[relative] = entry.read_bytes()
        bundles.extend(self._split_bundles(plain_files, root.name))
        return bundles

    def _discover_from_zip_bytes(self, payload: bytes, source_label: str, depth: int) -> list[DatasetBundle]:
        bundles: list[DatasetBundle] = []
        files: dict[str, bytes] = {}
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                name = info.filename
                if self._ignore_path(name):
                    continue
                data = archive.read(info)
                if name.lower().endswith(".zip") and self.scan_nested_archives and depth < self.max_archive_depth:
                    bundles.extend(self._discover_from_zip_bytes(data, file_basename(name), depth + 1))
                    continue
                files[name] = data
        bundles.extend(self._split_bundles(files, source_label))
        return bundles

    def _split_bundles(self, files: dict[str, bytes], source_label: str) -> list[DatasetBundle]:
        if not files:
            return []
        grouped: dict[str, dict[str, bytes]] = defaultdict(dict)
        for relative_path in files:
            if relative_path.lower().endswith("transactions.csv"):
                root = PurePosixPath(relative_path).parent.as_posix()
                grouped[root or "."] = {}
        if not grouped:
            return []
        for root in list(grouped):
            root_prefix = "" if root == "." else f"{root}/"
            for relative_path, payload in files.items():
                if root == ".":
                    grouped[root][file_basename(relative_path)] = payload
                elif relative_path == root or relative_path.startswith(root_prefix):
                    grouped[root][relative_path[len(root_prefix) :]] = payload
        bundles = []
        for root, bundle_files in grouped.items():
            raw_name = file_basename(root) if root != "." else Path(source_label).stem
            name = raw_name or Path(source_label).stem or "dataset"
            bundles.append(
                DatasetBundle(
                    name=name,
                    slug=slugify(name),
                    source_label=source_label,
                    files=bundle_files,
                )
            )
        return bundles

    @staticmethod
    def _ignore_path(relative_path: str) -> bool:
        path = relative_path.replace("\\", "/")
        return (
            path.startswith("__MACOSX/")
            or "/." in path
            or path.startswith(".")
            or path.endswith(".DS_Store")
        )
