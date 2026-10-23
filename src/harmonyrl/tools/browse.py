"""Read-only localhost file server for browsing rendered audio and scores.

Serves a directory tree over HTTP with a dark directory page that plays WAV/MP3
inline, opens PDFs in the browser's viewer, and offers downloads for everything
else. Binds to 127.0.0.1 only: reach it through an SSH tunnel
(`ssh -L 8000:localhost:8000`) rather than exposing a port.
"""

import html
import signal
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

AUDIO_TYPES = {".wav": "audio/wav", ".mp3": "audio/mpeg"}
VIEWER_TYPES = {".pdf": "application/pdf"}
MIDI_SUFFIXES = {".mid", ".midi"}
TEXT_SUFFIXES = {".csv", ".json", ".md", ".mscx", ".musicxml", ".txt", ".xml", ".yaml", ".yml"}
ICONS = {"audio": "♪", "pdf": "▭", "archive": "▤", "midi": "♬", "text": "≡"}

_STYLE = """
body { background: #14161a; color: #d7dae0; font: 15px/1.5 system-ui, sans-serif;
       margin: 0; padding: 2rem; }
a { color: #8ab4f8; text-decoration: none; }
a:hover { text-decoration: underline; }
header { margin-bottom: 1.5rem; }
h1 { font-size: 1.1rem; font-weight: 500; color: #9aa0a6; }
.crumbs { font-size: 1rem; margin-top: .2rem; }
table { border-collapse: collapse; width: 100%; max-width: 60rem; }
td { padding: .55rem .75rem .55rem 0; border-bottom: 1px solid #23262c;
     vertical-align: middle; white-space: nowrap; }
td.name { white-space: normal; min-width: 18rem; }
.size { color: #9aa0a6; font-variant-numeric: tabular-nums; }
audio { height: 2rem; width: 22rem; max-width: 60vw; }
"""


def serve(root: Path, port: int) -> None:
	"""Serve `root` on 127.0.0.1:`port` until interrupted.

	Ctrl+C and SIGTERM are the normal ways to stop the server: the interrupt is
	caught, the listening socket is closed, and the process exits with status 0.
	"""
	root = root.resolve()
	server = ThreadingHTTPServer(("127.0.0.1", port), _make_handler(root))
	signal.signal(signal.SIGTERM, _terminate)
	try:
		server.serve_forever()
	except KeyboardInterrupt:
		pass
	finally:
		server.server_close()


def _terminate(signum: int, frame: object) -> None:
	raise KeyboardInterrupt


def _make_handler(root: Path) -> type[BaseHTTPRequestHandler]:
	class Handler(BaseHTTPRequestHandler):
		def do_GET(self) -> None:
			requested = urllib.parse.urlparse(self.path).path
			relative = urllib.parse.unquote(requested).lstrip("/")
			target = (root / relative).resolve()
			try:
				target.relative_to(root)
			except ValueError:
				self.send_error(403, "outside the served root")
				return
			if target.is_dir():
				self._send_html(_listing(root, target))
			elif target.is_file():
				self._send_file(target)
			else:
				self.send_error(404, "not found")

		def _send_html(self, page: str) -> None:
			body = page.encode("utf-8")
			self.send_response(200)
			self.send_header("Content-Type", "text/html; charset=utf-8")
			self.send_header("Content-Length", str(len(body)))
			self.end_headers()
			self._write_bytes(body)

		def _send_file(self, target: Path) -> None:
			suffix = target.suffix.lower()
			content_type = (
				AUDIO_TYPES.get(suffix) or VIEWER_TYPES.get(suffix) or "application/octet-stream"
			)
			size = target.stat().st_size
			range_header = self.headers.get("Range")
			if range_header is not None and range_header.startswith("bytes="):
				byte_range = _parse_range(range_header, size)
				if byte_range is None:
					self.send_response(416)
					self.send_header("Content-Range", f"bytes */{size}")
					self.send_header("Content-Length", "0")
					self.end_headers()
					return
				start, end = byte_range
				length = end - start + 1
				self.send_response(206)
				self.send_header("Content-Type", content_type)
				self.send_header("Accept-Ranges", "bytes")
				self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
				self.send_header("Content-Length", str(length))
				self.end_headers()
				self._stream_slice(target, start, length)
				return

			self.send_response(200)
			self.send_header("Content-Type", content_type)
			self.send_header("Accept-Ranges", "bytes")
			self.send_header("Content-Length", str(size))
			self.end_headers()
			self._stream_slice(target, 0, size)

		def _stream_slice(self, target: Path, start: int, length: int) -> None:
			remaining = length
			with target.open("rb") as f:
				if start:
					f.seek(start)
				while remaining > 0:
					chunk = f.read(min(1 << 16, remaining))
					if not chunk:
						break
					if not self._write_bytes(chunk):
						return
					remaining -= len(chunk)

		def _write_bytes(self, data: bytes) -> bool:
			try:
				self.wfile.write(data)
			except BrokenPipeError, ConnectionResetError:
				return False
			return True

	return Handler


def _parse_range(header: str, size: int) -> tuple[int, int] | None:
	"""Parse a single `bytes=start-end` header against a file of `size` bytes."""
	if size == 0:
		return None
	spec = header.removeprefix("bytes=").strip()
	if "," in spec or "-" not in spec:
		return None
	start_str, end_str = spec.split("-", 1)
	try:
		if not start_str:
			suffix_len = int(end_str)
			if suffix_len <= 0:
				return None
			return max(0, size - suffix_len), size - 1
		start = int(start_str)
		end = size - 1 if not end_str else min(int(end_str), size - 1)
	except ValueError:
		return None
	if start < 0 or start >= size or start > end:
		return None
	return start, end


def _url_path(relative: Path, *, trailing_slash: bool = False) -> str:
	if not relative.parts:
		return "/"
	encoded = "/".join(urllib.parse.quote(part) for part in relative.parts)
	return f"/{encoded}/" if trailing_slash else f"/{encoded}"


def _icon_for(suffix: str) -> str:
	if suffix in AUDIO_TYPES:
		return ICONS["audio"]
	if suffix in VIEWER_TYPES:
		return ICONS["pdf"]
	if suffix in MIDI_SUFFIXES:
		return ICONS["midi"]
	if suffix in TEXT_SUFFIXES:
		return ICONS["text"]
	return ICONS["archive"]


def _listing(root: Path, directory: Path) -> str:
	relative = directory.relative_to(root)
	crumbs = ' / <a href="/">root</a>'
	for i, part in enumerate(relative.parts):
		sub = Path(*relative.parts[: i + 1])
		crumbs += f' / <a href="{_url_path(sub, trailing_slash=True)}">{html.escape(part)}</a>'

	rows = []
	entries = sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
	if directory != root:
		parent_href = _url_path(relative.parent, trailing_slash=True)
		rows.append(
			f"<tr><td class='name'>📁 <a href='{parent_href}'>..</a></td><td></td><td></td></tr>"
		)
	for entry in entries:
		name = html.escape(entry.name)
		entry_rel = relative / entry.name
		if entry.is_dir():
			dir_href = _url_path(entry_rel, trailing_slash=True)
			rows.append(
				f"<tr><td class='name'>📁 <a href='{dir_href}'>{name}/</a></td>"
				f"<td></td><td class='size'>{_count(entry)} items</td></tr>"
			)
			continue
		stat = entry.stat()
		size = (
			f"{stat.st_size / 1e6:.1f} MB" if stat.st_size > 1e6 else f"{stat.st_size / 1e3:.0f} KB"
		)
		href = _url_path(entry_rel)
		suffix = entry.suffix.lower()
		icon = _icon_for(suffix)
		if suffix in AUDIO_TYPES:
			player = f"<audio controls preload='none' src='{href}'></audio>"
			rows.append(
				f"<tr><td class='name'>{icon} {name}</td>"
				f"<td>{player}</td><td class='size'>{size}</td></tr>"
			)
		elif suffix in VIEWER_TYPES:
			name_cell = f"<td class='name'>{icon} <a href='{href}' target='_blank'>{name}</a></td>"
			open_cell = f"<td><a href='{href}' target='_blank'>open</a></td>"
			rows.append(f"<tr>{name_cell}{open_cell}<td class='size'>{size}</td></tr>")
		else:
			name_cell = f"<td class='name'>{icon} <a href='{href}' download>{name}</a></td>"
			open_cell = f"<td><a href='{href}' download>download</a></td>"
			rows.append(f"<tr>{name_cell}{open_cell}<td class='size'>{size}</td></tr>")
	title = html.escape(str(relative) if relative.parts else "files")
	return f"""<!doctype html><html><head><meta charset='utf-8'>
<title>{title}</title><style>{_STYLE}</style></head>
<body><header><h1>harmonyrl files</h1><div class='crumbs'>{crumbs}</div></header>
<table>{"".join(rows)}</table></body></html>"""


def _count(directory: Path) -> int:
	return sum(1 for _ in directory.iterdir())
