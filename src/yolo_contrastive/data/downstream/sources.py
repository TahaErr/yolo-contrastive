"""Source list loading for the downstream pool — scales to any N.

The downstream pool is built from a list of Roboflow export URLs. To add the
15th (or 50th) source you edit the list / file, not the code. A source spec is
one of:

  * dict  ``{name: url}``                — used as-is
  * list  ``[url, url, ...]``            — auto-named ds_01, ds_02, ...
  * path to ``.yaml`` / ``.yml``         — a mapping ``{name: url}`` or a list of urls
  * path to ``.txt`` (or any other ext)  — one entry per line:

        "url"                    -> auto-named
        "name <whitespace> url"  -> explicit name

    blank lines and lines starting with '#' are ignored.

Auto-generated names are zero-padded to the source count (ds_01.. for <100,
ds_001.. for >=100) so they sort lexicographically.
"""

from __future__ import annotations

from pathlib import Path

import yaml


def _auto_name(i: int, n: int) -> str:
    width = max(2, len(str(n)))
    return f"ds_{i:0{width}d}"


def _from_list(urls: list[str]) -> dict[str, str]:
    n = len(urls)
    return {_auto_name(i, n): str(u).strip() for i, u in enumerate(urls, start=1)}


def _is_url(token: str) -> bool:
    return token.lower().startswith(("http://", "https://"))


def _from_txt(path: Path) -> dict[str, str]:
    urls: list[str] = []
    named: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) == 2 and not _is_url(parts[0]):
            named[parts[0]] = parts[1].strip()
        else:
            urls.append(line)
    if named and urls:
        raise ValueError(f"{path}: mix of named and unnamed lines is ambiguous")
    return named if named else _from_list(urls)


def load_sources(spec: dict | list | tuple | str | Path) -> dict[str, str]:
    """Normalize a source spec (dict / list / file path) to an ordered ``{name: url}``."""
    if isinstance(spec, dict):
        out = {str(k): str(v).strip() for k, v in spec.items()}
    elif isinstance(spec, (list, tuple)):
        out = _from_list(list(spec))
    else:
        path = Path(spec)
        if not path.is_file():
            raise FileNotFoundError(f"sources file not found: {path}")
        if path.suffix.lower() in {".yaml", ".yml"}:
            data = yaml.safe_load(path.read_text()) or {}
            out = ({str(k): str(v).strip() for k, v in data.items()}
                   if isinstance(data, dict) else _from_list(list(data)))
        else:
            out = _from_txt(path)

    if not out:
        raise ValueError("no sources found in spec")
    if any(not u for u in out.values()):
        raise ValueError("empty url in sources")
    return out
