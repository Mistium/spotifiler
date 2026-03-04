import os
import json
import time
import re
import subprocess
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import requests
from datetime import timedelta

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TaskProgressColumn, TextColumn
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.align import Align
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import yt_dlp
from dotenv import load_dotenv

load_dotenv()

# ─── Auth Setup ───────────────────────────────────────────────────────────────

auth_manager = SpotifyOAuth(
    client_id=os.getenv("CLIENT_ID"),
    client_secret=os.getenv("CLIENT_SECRET"),
    redirect_uri=os.getenv("REDIRECT_URI"),
    scope="user-library-read user-read-recently-played",
    show_dialog=True,
)
sp = None
console = Console(
    width=min(
        140, os.get_terminal_size().columns if hasattr(os, "get_terminal_size") else 140
    )
)


# ─── TUI ──────────────────────────────────────────────────────────────────────


class TUIManager:
    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if TUIManager._initialized:
            return
        TUIManager._initialized = True

        self.console = console
        self.spin_frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self.spin_index = 0
        self.live = None
        self.state = {
            "albums": {"total": 0, "completed": 0},
            "songs": {"total": 0, "completed": 0, "failed": 0, "skipped": 0},
            "threads": {},
            "start_time": None,
            "total_downloaded_bytes": 0,
            "download_speed": 0,
            "activity": "idle",
        }

        self.layout = Layout()
        self.layout.split_column(
            Layout(name="header", size=4),
            Layout(name="body", ratio=1),
            Layout(name="footer", size=3),
        )
        self.layout["body"].split_row(
            Layout(name="stats", size=40), Layout(name="threads", ratio=1)
        )

    def spinner(self):
        self.spin_index = (self.spin_index + 1) % len(self.spin_frames)
        return self.spin_frames[self.spin_index]

    def initialize(self):
        self.state["start_time"] = time.time()
        self.live = Live(self.layout, console=self.console, refresh_per_second=30)
        self.live.start()

    def shutdown(self):
        if self.live:
            self.live.stop()
            self.live = None

    def update_state(self, **kwargs):
        with self._lock:
            for key, value in kwargs.items():
                if isinstance(self.state.get(key), dict) and isinstance(value, dict):
                    self.state[key].update(value)
                else:
                    self.state[key] = value
            statuses = {t.get("status") for t in self.state["threads"].values()}
            self.state["activity"] = next(
                (
                    s
                    for s in ("downloading", "searching", "processing")
                    if s in statuses
                ),
                "idle",
            )

    def refresh(self):
        if not self.live:
            return
        s = self.state
        songs = s["songs"]
        elapsed = (
            str(timedelta(seconds=int(time.time() - s["start_time"])))
            if s["start_time"]
            else "00:00:00"
        )
        total, done = songs["total"], songs["completed"]
        pct = done / total * 100 if total else 0
        act = s["activity"]

        spin = self.spinner()
        status_map = {
            "downloading": Text(f"{spin} Downloading...", style="bold green"),
            "searching": Text(f"{spin} Searching...", style="bold yellow"),
            "processing": Text(f"{spin} Processing...", style="bold cyan"),
        }
        status_text = status_map.get(act, Text("● Idle", style="dim"))

        header_text = Text.assemble(
            ("🎵 ", "bold yellow"),
            ("Spotify Album Downloader", "bold white"),
            ("  ", ""),
            status_text,
            ("  ", ""),
            ("Albums: ", "dim"),
            (
                f"{s['albums']['completed']}/{s['albums']['total']}",
                "green" if s["albums"]["completed"] == s["albums"]["total"] else "cyan",
            ),
            (" | ", "dim"),
            ("Songs: ", "dim"),
            (f"{done}/{total}", "green" if done == total else "cyan"),
        )
        self.layout["header"].update(
            Panel(Align.center(header_text), style="bold blue")
        )

        footer_text = Text.assemble(
            ("Elapsed: ", "dim"),
            (elapsed, "cyan"),
            (" | ", "dim"),
            ("Progress: ", "dim"),
            (f"{pct:.1f}%", "yellow"),
            (" | ", "dim"),
            ("Downloaded: ", "dim"),
            (_fmt_bytes(s["total_downloaded_bytes"]), "cyan"),
            (" | ", "dim"),
            ("Speed: ", "dim"),
            (f"{_fmt_bytes(s['download_speed'])}/s", "cyan"),
        )
        self.layout["footer"].update(Panel(Align.center(footer_text)))

        # Stats panel
        filled = (
            int((done + songs["failed"] + songs["skipped"]) / total * 20)
            if total
            else 0
        )
        bar = "█" * filled + "░" * (20 - filled)
        active = sum(
            1
            for t in s["threads"].values()
            if t.get("status") in ("searching", "downloading", "processing")
        )
        st = Table.grid(padding=0)
        st.add_column(style="cyan", width=14)
        st.add_column(style="white")
        for row in [
            ("Albums:", f"{s['albums']['completed']} / {s['albums']['total']}"),
            ("", ""),
            ("Songs:", str(total)),
            ("  Completed:", f"[green]{done}[/green]"),
            ("  Failed:", f"[red]{songs['failed']}[/red]"),
            ("  Skipped:", f"[yellow]{songs['skipped']}[/yellow]"),
            ("", ""),
            ("Progress:", Text(f"{bar} {pct:.1f}%", style="cyan")),
            ("", ""),
            ("Data:", _fmt_bytes(s["total_downloaded_bytes"])),
            ("Speed:", f"{_fmt_bytes(s['download_speed'])}/s"),
            ("", ""),
            ("Active:", f"[yellow]{active}[/yellow] threads"),
        ]:
            st.add_row(*row)
        self.layout["stats"].update(
            Panel(
                Align.center(st), title="[bold]Statistics[/bold]", border_style="cyan"
            )
        )

        # Threads panel
        tt = Table(
            show_header=True, header_style="bold magenta", box=None, padding=(0, 1)
        )
        for col, kw in [
            ("Thread", {"style": "cyan", "width": 10}),
            ("Status", {"width": 14}),
            ("Current File", {"ratio": 1}),
            ("Progress", {"width": 20}),
            ("Speed", {"width": 12}),
        ]:
            tt.add_column(col, **kw)

        thread_status_map = {
            "searching": lambda sp: Text(f"🔍 {sp}", style="yellow"),
            "downloading": lambda sp: Text(f"⬇ {sp}", style="green"),
            "processing": lambda sp: Text(f"⚙ {sp}", style="blue"),
            "completed": lambda _: Text("✓ Done", style="green"),
            "failed": lambda _: Text("✗ Failed", style="red"),
            "skipped": lambda _: Text("⊘ Skip", style="yellow"),
        }

        for tid in sorted(s["threads"])[:6]:
            td = s["threads"][tid]
            status = td.get("status", "idle")
            prog = td.get("progress", 0)
            sp_txt = (
                spin if status in ("searching", "downloading", "processing") else ""
            )
            st_text = thread_status_map.get(
                status, lambda _: Text("● Idle", style="dim")
            )(sp_txt)

            if status == "downloading" and prog > 0:
                b = "█" * int(prog / 5) + "░" * (20 - int(prog / 5))
                prog_text = Text(f"{b} {prog:.0f}%", style="green")
            elif status == "completed":
                prog_text = Text("████████████████████ 100%", style="green")
            elif status == "failed":
                prog_text = Text("-------------------- Failed", style="red")
            elif status == "skipped":
                prog_text = Text("░░░░░░░░░░░░░░░░░░░░ Skipped", style="yellow")
            else:
                prog_text = Text("-" * 30, style="dim")

            song = td.get("current_song", "")
            tt.add_row(
                f"#{tid}",
                st_text,
                Text(song[:55] or "-", style="white"),
                prog_text,
                Text(td.get("speed", "") or "-", style="cyan"),
            )

        extra = len(s["threads"]) - 6
        if extra > 0:
            tt.add_row("", Text(f"+ {extra} more", style="dim"), "", "", "")

        self.layout["threads"].update(
            Panel(
                tt,
                title=f"[bold]Thread Status [{len(s['threads'])} threads][/bold]",
                border_style="magenta",
            )
        )
        self.live.update(self.layout)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _fmt_bytes(b):
    for unit in ("B", "KB", "MB", "GB"):
        if b < 1024:
            return f"{b:.2f} {unit}"
        b /= 1024
    return f"{b:.2f} TB"


def _parse_speed(s):
    try:
        s = s.strip().upper()
        multipliers = {
            "GB/S": 1 << 30,
            "GIB/S": 1 << 30,
            "MB/S": 1 << 20,
            "MIB/S": 1 << 20,
            "KB/S": 1 << 10,
            "KIB/S": 1 << 10,
            "B/S": 1,
        }
        for suffix, mult in multipliers.items():
            if suffix in s:
                return float(s.split()[0].replace(",", ".")) * mult
    except Exception:
        pass
    return 0


def _sanitize(name):
    return name


# ─── Spotify ──────────────────────────────────────────────────────────────────


def _save_token(token_info):
    with open(".spotify_token.json", "w") as f:
        json.dump(token_info, f)


def load_spotify_token():
    global sp
    if not os.path.exists(".spotify_token.json"):
        return False
    try:
        with open(".spotify_token.json") as f:
            token_info = json.load(f)
        if auth_manager.is_token_expired(token_info):
            token_info = auth_manager.refresh_access_token(token_info["refresh_token"])
            _save_token(token_info)
        sp = spotipy.Spotify(auth=token_info["access_token"])
        user = sp.current_user()
        console.print(
            f"[green]✓[/green] Authenticated as: [cyan]{user.get('display_name', 'Unknown')}[/cyan]"
        )
        return True
    except Exception as e:
        console.print(f"[red]Error loading token: {e}[/red]")
        return False


def authenticate_cli():
    global sp
    console.print(
        Panel(Text("Spotify Authentication", style="bold yellow"), style="bold")
    )
    auth_url = auth_manager.get_authorize_url()
    console.print(f"\n[cyan]Authorization URL:[/cyan]\n  {auth_url}\n")
    console.print(
        f"After authorizing, you'll be redirected to: {os.getenv('REDIRECT_URI')}?code=XXXXX...\n"
    )

    raw = console.input(
        "[yellow]Paste the full redirect URL (or just the 'code'): [/yellow]"
    ).strip()
    if "code=" in raw:
        import urllib.parse

        raw = urllib.parse.parse_qs(urllib.parse.urlparse(raw).query).get("code", [""])[
            0
        ]

    if not raw:
        console.print("[red]No authorization code provided.[/red]")
        return False

    try:
        token_info = auth_manager.get_access_token(raw)
        sp = spotipy.Spotify(auth=token_info["access_token"])
        user = sp.current_user()
        console.print(
            f"[green]✓ Authenticated as: {user.get('display_name', 'Unknown')}[/green]"
        )
        _save_token(token_info)
        return True
    except Exception as e:
        console.print(f"[red]Authentication failed: {e}[/red]")
        return False


def get_liked_albums():
    if sp is None:
        raise Exception("Not authenticated.")
    albums, results = [], sp.current_user_saved_albums(limit=50)
    while results:
        for item in results.get("items", []):
            a = item["album"]
            albums.append(
                {
                    "name": a["name"],
                    "artists": [x["name"] for x in a["artists"]],
                    "tracks": [t["name"] for t in a["tracks"]["items"]],
                    "metadata": {
                        "id": a["id"],
                        "uri": a["uri"],
                        "external_urls": a.get("external_urls", {}),
                        "release_date": a.get("release_date"),
                        "total_tracks": a.get("total_tracks"),
                        "album_type": a.get("album_type"),
                        "genres": a.get("genres", []),
                        "label": a.get("label"),
                        "popularity": a.get("popularity"),
                        "images": a.get("images", []),
                        "copyrights": a.get("copyrights", []),
                        "artists_detailed": [
                            {
                                "id": x["id"],
                                "name": x["name"],
                                "uri": x["uri"],
                                "external_urls": x.get("external_urls", {}),
                            }
                            for x in a["artists"]
                        ],
                        "tracks_detailed": [
                            {
                                "id": t["id"],
                                "name": t["name"],
                                "track_number": t["track_number"],
                                "disc_number": t["disc_number"],
                                "explicit": t["explicit"],
                                "duration_ms": t.get("duration_ms"),
                                "uri": t["uri"],
                                "external_urls": t.get("external_urls", {}),
                                "artists": [
                                    {"id": x["id"], "name": x["name"], "uri": x["uri"]}
                                    for x in t["artists"]
                                ],
                            }
                            for t in a["tracks"]["items"]
                        ],
                        "added_at": item.get("added_at"),
                    },
                }
            )
        results = sp.next(results) if results["next"] else None
    return albums


# ─── Download Manager ─────────────────────────────────────────────────────────


class DownloadManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_ready"):
            return
        self._ready = True
        self._lock = threading.Lock()
        self.tui = TUIManager()
        self.max_concurrent = int(os.getenv("MAX_CONCURRENT_DOWNLOADS", 4))
        self._max_file_size_mb = None
        self._progress = {
            "total_albums": 0,
            "completed_albums": 0,
            "total_songs": 0,
            "completed_songs": 0,
            "failed_downloads": 0,
            "skipped_songs": 0,
            "concurrent_downloads": 0,
            "threads": {},
        }

    def _update(self, **kw):
        with self._lock:
            self._progress.update(kw)
            mapping = {
                "total_albums": ("albums", "total"),
                "completed_albums": ("albums", "completed"),
                "total_songs": ("songs", "total"),
                "completed_songs": ("songs", "completed"),
                "failed_downloads": ("songs", "failed"),
                "skipped_songs": ("songs", "skipped"),
            }
            updates = {}
            for k, v in kw.items():
                if k in mapping:
                    section, field = mapping[k]
                    updates.setdefault(section, {})[field] = v
            for section, vals in updates.items():
                self.tui.update_state(**{section: vals})
            if updates:
                self.tui.refresh()

    def _update_thread(self, tid, **kw):
        with self._lock:
            self._progress["threads"].setdefault(
                tid, {"current_song": "", "status": "idle", "progress": 0, "speed": ""}
            )
            self._progress["threads"][tid].update(kw)
            self.tui.update_state(
                threads={t: d.copy() for t, d in self._progress["threads"].items()}
            )
            self.tui.refresh()

    def _progress_hook(self, d, tid):
        if d["status"] == "downloading":
            try:
                dl, total = (
                    d.get("downloaded_bytes", 0),
                    d.get("total_bytes") or d.get("total_bytes_estimate", 0),
                )
                pct = (
                    (dl / total * 100)
                    if total
                    else float(d.get("_percent_str", "0%").replace("%", "") or 0)
                )
                speed_bytes = _parse_speed(d.get("_speed_str", ""))
                self._update_thread(
                    tid,
                    progress=pct,
                    status="downloading",
                    speed=d.get("_speed_str", ""),
                    eta=d.get("_eta_str", ""),
                )
                with self._lock:
                    last = (
                        self._progress["threads"].get(tid, {}).get("last_downloaded", 0)
                    )
                    chunk = dl - last
                    if chunk > 0:
                        self.tui.update_state(
                            total_downloaded_bytes=self.tui.state.get(
                                "total_downloaded_bytes", 0
                            )
                            + chunk
                        )
                    self._progress["threads"][tid]["last_downloaded"] = dl
                self.tui.update_state(download_speed=speed_bytes)
                self.tui.refresh()
            except Exception:
                pass
        elif d["status"] == "finished":
            self._update_thread(tid, progress=100, status="processing")
        elif d["status"] == "error":
            self._update_thread(tid, status="failed")

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def search_youtube(self, song, artist, album=None, duration_ms=None):
        queries = [
            f"{artist} - {song}",
            f"{song} {artist} {album} official audio"
            if album
            else f"{song} {artist} official audio",
            f"{song} {artist} {album}" if album else f"{song} {artist}",
            f"{song} {artist}",
        ]
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,
            "default_search": "ytsearch3:",
            "socket_timeout": 10,
        }

        def try_query(query):
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    results = ydl.extract_info(f"ytsearch3:{query}", download=False)
                return (results or {}).get("entries") or []
            except Exception:
                return []

        with ThreadPoolExecutor(max_workers=4) as executor:
            all_entries = []
            for entries in executor.map(try_query, queries):
                all_entries.extend(entries)

        seen, unique_entries = set(), []
        for e in all_entries:
            if e.get("id") and e["id"] not in seen:
                seen.add(e["id"])
                unique_entries.append(e)

        if not unique_entries:
            return None

        if duration_ms:
            target = duration_ms / 1000

            def score(e):
                t = e.get("title", "").lower()
                bonus = (
                    -5
                    if any(w in t for w in ("official", "audio"))
                    else -3
                    if any(w in t for w in ("lyrics", "lyric"))
                    else 10
                    if any(w in t for w in ("cover", "remix", "live"))
                    else 0
                )
                return abs(e.get("duration", 0) - target) + bonus

            entry = min(unique_entries, key=score)
        else:
            entry = unique_entries[0]

        return f"https://www.youtube.com/watch?v={entry['id']}"

    def _song_exists(self, song_name, album_path):
        try:
            return any(
                f.startswith(song_name) and not f.startswith(".") and "_thread" not in f
                for f in os.listdir(album_path)
            )
        except OSError:
            return False

    def _download_with_retry(
        self, url, opts_base, album_path, temp_name, tid, retries=4
    ):
        formats = [
            "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio[ext=opus]/bestaudio/best",
            "bestaudio/best",
            "worstaudio/worst",
        ]
        for attempt in range(retries):
            for fmt in formats:
                try:
                    if attempt:
                        time.sleep(2**attempt)
                    opts = {**opts_base, "format": fmt}
                    with yt_dlp.YoutubeDL(opts) as ydl:
                        ydl.download([url])
                    found = [
                        f for f in os.listdir(album_path) if f.startswith(temp_name)
                    ]
                    if found:
                        return True, found[0]
                except Exception as e:
                    err = str(e).lower()
                    if "403" in err or "forbidden" in err:
                        time.sleep(3 + attempt * 2)
                    elif "format" not in err and "not available" not in err:
                        break
        return False, "Download failed"

    def download_song(self, song_info, album_path, tid=0):
        name = _sanitize(song_info["name"])
        artist = _sanitize(" & ".join(song_info["artists"]))
        album_name = song_info.get("album_name", "")
        duration_ms = song_info.get("duration_ms")

        self._update_thread(
            tid, current_song=f"{name} by {artist}", status="checking", progress=0
        )

        if self._song_exists(name, album_path):
            self._update_thread(tid, status="skipped", progress=100)
            self._update(
                completed_songs=self._progress["completed_songs"] + 1,
                skipped_songs=self._progress["skipped_songs"] + 1,
            )
            return True

        self._update_thread(tid, status="searching")
        url = self.search_youtube(name, artist, album_name, duration_ms)
        if not url:
            self._update_thread(tid, status="failed")
            self._update(failed_downloads=self._progress["failed_downloads"] + 1)
            return False

        self._update_thread(tid, status="downloading")
        temp = f"{name}_thread{tid}_{int(time.time())}"
        opts = {
            "outtmpl": os.path.join(album_path, f"{temp}.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "progress_hooks": [lambda d: self._progress_hook(d, tid)],
            "socket_timeout": 60,
            "retries": 5,
            "fragment_retries": 5,
            "skip_unavailable_fragments": True,
            "concurrent_fragment_downloads": 3,
            "http_headers": {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            },
        }

        success, result = self._download_with_retry(url, opts, album_path, temp, tid)
        if not success:
            self._update_thread(tid, status="failed")
            self._update(failed_downloads=self._progress["failed_downloads"] + 1)
            return False

        temp_path = os.path.join(album_path, result)
        ext = os.path.splitext(result)[1].lower()
        final_path = os.path.join(album_path, f"{name}.mp3")

        if ext != ".mp3":
            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        temp_path,
                        "-codec:a",
                        "libmp3lame",
                        "-b:a",
                        "320k",
                        "-map",
                        "a",
                        "-loglevel",
                        "error",
                        final_path,
                    ],
                    capture_output=True,
                )
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            except Exception:
                os.rename(temp_path, os.path.join(album_path, f"{name}{ext}"))
        else:
            os.rename(temp_path, final_path)

        self._update_thread(tid, status="completed", progress=100)
        self._update(completed_songs=self._progress["completed_songs"] + 1)
        return True

    def _download_image(self, url, path):
        try:
            r = requests.get(url, stream=True, timeout=30)
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            return True
        except Exception:
            return False

    def _save_album_metadata(self, album, album_path):
        try:
            data = {
                "basic_info": {
                    "name": album["name"],
                    "artists": album["artists"],
                    "total_tracks": len(album["tracks"]),
                },
                "spotify_metadata": album.get("metadata", {}),
                "download_info": {
                    "downloaded_at": time.strftime(
                        "%Y-%m-%d %H:%M:%S UTC", time.gmtime()
                    ),
                    "downloader_version": "1.1",
                    "tracks_list": album["tracks"],
                    "artwork_downloaded": {
                        "album_cover": "album.png",
                        "artist_image": "artist.png",
                    },
                },
            }
            with open(
                os.path.join(album_path, "album.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception:
            return False

    def _download_artwork(self, album, album_path):
        meta = album.get("metadata", {})
        if imgs := meta.get("images"):
            if not os.path.exists(os.path.join(album_path, "album.png")):
                self._download_image(
                    imgs[0]["url"], os.path.join(album_path, "album.png")
                )
        if (artists := meta.get("artists_detailed")) and sp:
            try:
                if not os.path.exists(os.path.join(album_path, "artist.png")):
                    artist_imgs = sp.artist(artists[0]["id"]).get("images", [])
                    if artist_imgs:
                        self._download_image(
                            artist_imgs[0]["url"],
                            os.path.join(album_path, "artist.png"),
                        )
            except Exception:
                pass

    def _artist_data_exists(self, artist_name):
        path = os.path.join(os.getcwd(), "artists", _sanitize(artist_name), "info.json")
        if not os.path.exists(path):
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not data.get("id") or not data.get("name"):
                return False
            return True
        except (json.JSONDecodeError, IOError, KeyError):
            return False

    def _save_artist_artwork(self, artist_name, image_url):
        path = os.path.join(os.getcwd(), "artists", _sanitize(artist_name))
        if imgs := [
            {"url": i["url"], "width": i["width"], "height": i["height"]}
            for i in self._get_spotify_artist_images(artist_name)
        ]:
            largest = max(imgs, key=lambda x: x.get("width", 0) * x.get("height", 0))
            self._download_image(largest["url"], os.path.join(path, "icon.png"))

    def _get_spotify_artist_images(self, artist_name):
        try:
            path = os.path.join(
                os.getcwd(), "artists", _sanitize(artist_name), "info.json"
            )
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data.get("images", [])
        except (json.JSONDecodeError, IOError, KeyError):
            pass
        return []

    def _album_metadata_exists(self, album, album_path):
        metadata_file = os.path.join(album_path, "album.json")
        if not os.path.exists(metadata_file):
            return False
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not data.get("basic_info") or not data.get("spotify_metadata"):
                return False
            return True
        except (json.JSONDecodeError, IOError, KeyError):
            return False

    def _save_album_metadata(self, album, album_path):
        try:
            data = {
                "basic_info": {
                    "name": album["name"],
                    "artists": album["artists"],
                    "total_tracks": len(album["tracks"]),
                },
                "spotify_metadata": album.get("metadata", {}),
                "download_info": {
                    "downloaded_at": time.strftime(
                        "%Y-%m-%d %H:%M:%S UTC", time.gmtime()
                    ),
                    "downloader_version": "1.1",
                    "tracks_list": album["tracks"],
                    "artwork_downloaded": {
                        "album_cover": "album.png",
                        "artist_image": "artist.png",
                    },
                },
            }
            with open(
                os.path.join(album_path, "album.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception:
            return False

    def fetch_and_save_artist_data(self, artist_id, artist_name):
        if not sp:
            return False
        try:
            if self._artist_data_exists(artist_name):
                return True
            path = os.path.join(os.getcwd(), "artists", _sanitize(artist_name))
            os.makedirs(path, exist_ok=True)
            details = sp.artist(artist_id)
            if imgs := details.get("images"):
                largest = max(
                    imgs, key=lambda x: x.get("width", 0) * x.get("height", 0)
                )
                self._download_image(largest["url"], os.path.join(path, "icon.png"))

            albums = []
            results = sp.artist_albums(artist_id, album_type="album", limit=50)
            while results:
                for a in results["items"]:
                    albums.append(
                        {
                            "name": a["name"],
                            "id": a["id"],
                            "release_date": a.get("release_date"),
                            "total_tracks": a.get("total_tracks"),
                            "cover_art": (a.get("images") or [{}])[0].get("url"),
                        }
                    )
                results = sp.next(results) if results["next"] else None

            with open(os.path.join(path, "info.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "name": details.get("name"),
                        "id": details.get("id"),
                        "genres": details.get("genres", []),
                        "followers": details.get("followers", {}).get("total", 0),
                        "popularity": details.get("popularity", 0),
                        "external_urls": details.get("external_urls", {}),
                        "albums": albums,
                        "images": [
                            {
                                "url": i["url"],
                                "width": i["width"],
                                "height": i["height"],
                            }
                            for i in details.get("images", [])
                        ],
                        "downloaded_at": time.strftime(
                            "%Y-%m-%d %H:%M:%S UTC", time.gmtime()
                        ),
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            return True
        except Exception:
            return False

    def _album_folder(self, album, base_path):
        name = _sanitize(album["name"])
        artist = _sanitize(" & ".join(album["artists"]))
        return os.path.join(base_path, f"{artist} - {name}")

    def _build_song_queue(self, albums, base_path):
        q = queue.Queue()
        skipped = 0
        for i, album in enumerate(albums):
            album_path = self._album_folder(album, base_path)
            track_details = album.get("metadata", {}).get("tracks_detailed", [])
            for j, track in enumerate(album["tracks"]):
                name = _sanitize(track)
                if self._song_exists(name, album_path):
                    skipped += 1
                    continue
                q.put(
                    {
                        "name": track,
                        "artists": album["artists"],
                        "album_name": _sanitize(album["name"]),
                        "artist_name": _sanitize(" & ".join(album["artists"])),
                        "album_path": album_path,
                        "album_idx": i,
                        "duration_ms": track_details[j].get("duration_ms")
                        if j < len(track_details)
                        else None,
                    }
                )
        if skipped:
            console.print(
                f"[yellow]Skipping {skipped} already-downloaded song{'s' if skipped != 1 else ''}[/yellow]"
            )
            self._update(skipped_songs=skipped, completed_songs=skipped)
        return q

    def _run_workers(self, dq, albums, on_result):
        completed_set = set()

        def worker(tid):
            while True:
                try:
                    info = dq.get(timeout=5)
                    result = self.download_song(info, info["album_path"], tid)
                    on_result(
                        info,
                        result,
                        self._song_exists(_sanitize(info["name"]), info["album_path"]),
                    )
                    dq.task_done()
                    self._check_album_complete(albums, info["album_idx"], completed_set)
                except queue.Empty:
                    break
                except Exception:
                    dq.task_done()

        threads = [
            threading.Thread(target=worker, args=(tid,), daemon=True)
            for tid in range(self.max_concurrent)
        ]
        for t in threads:
            t.start()
        dq.join()
        for t in threads:
            t.join(timeout=10)
        return completed_set

    def _check_album_complete(self, albums, idx, completed_set):
        if idx in completed_set:
            return
        album = albums[idx]
        path = self._album_folder(
            album, os.path.join(os.getcwd(), os.getenv("DOWNLOAD_PATH", "songs"))
        )
        try:
            songs = [
                f
                for f in os.listdir(path)
                if not f.startswith(".")
                and "_thread" not in f
                and f not in ("album.json", "album.png", "artist.png")
            ]
            if len(songs) >= len(album["tracks"]):
                completed_set.add(idx)
                self._update(completed_albums=len(completed_set))
        except OSError:
            pass

    def download_albums_cli(self, albums):
        base_path = os.path.join(os.getcwd(), os.getenv("DOWNLOAD_PATH", "songs"))
        os.makedirs(base_path, exist_ok=True)

        # Phase 1: Metadata + artwork
        console.print(
            "\n[bold yellow]Phase 1: Downloading metadata and artwork[/bold yellow]"
        )
        p1_lock = threading.Lock()
        p1_count = 0
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as prog:
            task = prog.add_task("Processing albums...", total=len(albums))

            def _process_album(album):
                nonlocal p1_count
                apath = self._album_folder(album, base_path)
                os.makedirs(apath, exist_ok=True)
                if not self._album_metadata_exists(album, apath):
                    self._save_album_metadata(album, apath)
                self._download_artwork(album, apath)
                with p1_lock:
                    p1_count += 1
                    prog.update(
                        task,
                        description=f"[{p1_count}/{len(albums)}] {album['name'][:40]}",
                        advance=1,
                    )

            with ThreadPoolExecutor(max_workers=min(10, len(albums))) as ex:
                list(ex.map(_process_album, albums))
        console.print("[green]✓[/green] Metadata complete")

        # Phase 2: Artists
        unique_artists = {
            a["id"]: a["name"]
            for album in albums
            for a in album.get("metadata", {}).get("artists_detailed", [])[:1]
            if a.get("id")
        }
        artists_to_fetch = {
            aid: aname
            for aid, aname in unique_artists.items()
            if not self._artist_data_exists(aname)
        }
        console.print(
            f"\n[bold yellow]Phase 2: Artist data ({len(unique_artists)} artists, {len(artists_to_fetch)} new)[/bold yellow]"
        )
        p2_lock = threading.Lock()
        p2_count = 0
        artist_items = list(artists_to_fetch.items())
        if artist_items:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as prog:
                task = prog.add_task("Fetching artists...", total=len(artist_items))

                def _process_artist(item):
                    nonlocal p2_count
                    aid, aname = item
                    self.fetch_and_save_artist_data(aid, aname)
                    with p2_lock:
                        p2_count += 1
                        prog.update(
                            task,
                            description=f"[{p2_count}/{len(artist_items)}] {aname[:40]}",
                            advance=1,
                        )

                with ThreadPoolExecutor(max_workers=min(10, len(artist_items))) as ex:
                    list(ex.map(_process_artist, artist_items))
        else:
            console.print("[yellow]All artist data already exists - skipping[/yellow]")
        console.print("[green]✓[/green] Artist data complete")

        # Phase 3: Songs
        console.print("\n[bold yellow]Phase 3: Downloading songs[/bold yellow]")
        total_songs = sum(len(a["tracks"]) for a in albums)
        counts = {"done": 0, "skipped": 0, "failed": 0}
        lock = threading.Lock()

        self.tui.update_state(
            albums={"total": len(albums), "completed": 0},
            songs={"total": total_songs, "completed": 0, "failed": 0, "skipped": 0},
        )
        self.tui.initialize()

        def on_result(info, result, was_existing):
            with lock:
                if was_existing or (
                    result
                    and not was_existing
                    and self._song_exists(_sanitize(info["name"]), info["album_path"])
                ):
                    pass  # already tracked by download_song via _update
            self.tui.refresh()

        dq = self._build_song_queue(albums, base_path)
        t0 = time.time()
        try:
            completed_set = self._run_workers(dq, albums, on_result)
        except KeyboardInterrupt:
            console.print("\n[yellow]Cancelled.[/yellow]")
        finally:
            self.tui.shutdown()

        elapsed = time.time() - t0
        p = self._progress
        console.print(
            Panel(Text("Download Complete!", style="bold green"), style="bold green")
        )
        console.print(
            f"Downloaded: [green]{p['completed_songs'] - p['skipped_songs']}[/green] | Skipped: [yellow]{p['skipped_songs']}[/yellow] | Failed: [red]{p['failed_downloads']}[/red]"
        )
        console.print(
            f"Time: [cyan]{elapsed:.1f}s[/cyan]"
            + (
                f" | Avg: [cyan]{elapsed / total_songs:.1f}s/song[/cyan]"
                if total_songs
                else ""
            )
        )

    def compress_all_music(self):
        base_path = os.path.join(os.getcwd(), os.getenv("DOWNLOAD_PATH", "songs"))
        if not os.path.exists(base_path):
            console.print(f"[red]Songs directory not found: {base_path}[/red]")
            return

        exts = {".mp3", ".m4a", ".webm", ".opus", ".ogg", ".wav", ".flac"}
        files = [
            os.path.join(r, f)
            for r, _, fs in os.walk(base_path)
            for f in fs
            if os.path.splitext(f)[1].lower() in exts
        ]
        console.print(f"Found {len(files)} audio files\n")

        total_orig = total_comp = 0
        for i, fp in enumerate(files):
            console.print(f"[{i + 1}/{len(files)}] {os.path.basename(fp)}")
            orig, comp = self._compress_file(fp)
            total_orig += orig
            total_comp += comp

        saved = total_orig - total_comp
        console.print(
            f"\n[bold]Complete![/bold] {_fmt_bytes(total_orig)} → {_fmt_bytes(total_comp)} (saved {_fmt_bytes(saved)}, {saved / total_orig * 100:.1f}%)"
            if total_orig
            else "\n[bold]Complete![/bold]"
        )

    def _compress_file(self, fp):
        orig = os.path.getsize(fp) if os.path.exists(fp) else 0
        if orig < 1 << 20:
            console.print(f"  Skip (small): {_fmt_bytes(orig)}")
            return orig, orig

        ext = os.path.splitext(fp)[1].lower()
        if self._max_file_size_mb and orig > self._max_file_size_mb * 2 * (1 << 20):
            bitrate, q = "96k", "5"
        elif orig > 8 << 20 or (
            self._max_file_size_mb and orig > self._max_file_size_mb * (1 << 20)
        ):
            bitrate, q = "128k", "4"
        elif ext == ".mp3":
            console.print("  Skip (already MP3 and small)")
            return orig, orig
        else:
            bitrate, q = "192k", "2"

        tmp = f"{os.path.splitext(fp)[0]}_compressed.mp3"
        try:
            r = subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    fp,
                    "-codec:a",
                    "libmp3lame",
                    "-b:a",
                    bitrate,
                    "-q:a",
                    q,
                    "-map",
                    "a",
                    "-y",
                    "-loglevel",
                    "error",
                    tmp,
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if r.returncode != 0 or not os.path.exists(tmp):
                return orig, orig

            comp = os.path.getsize(tmp)
            min_savings = 0.01 if orig > 4 << 20 else 0.05
            if comp < orig * (1 - min_savings):
                os.replace(fp, f"{os.path.splitext(fp)[0]}_orig{ext}")
                os.replace(tmp, fp)
                try:
                    os.remove(f"{os.path.splitext(fp)[0]}_orig{ext}")
                except Exception:
                    pass
                saved = orig - comp
                console.print(
                    f"  ✓ {_fmt_bytes(orig)} → {_fmt_bytes(comp)} (saved {_fmt_bytes(saved)}, {saved / orig * 100:.1f}%)"
                )
                return orig, comp
            else:
                os.remove(tmp)
                console.print("  Skip (compression not beneficial)")
                return orig, orig
        except Exception as e:
            if os.path.exists(tmp):
                os.remove(tmp)
            console.print(f"  Error: {e}")
            return orig, orig


# ─── CLI ──────────────────────────────────────────────────────────────────────


def _require_auth():
    if not load_spotify_token():
        console.print("[red]Not authenticated. Run with --auth first.[/red]")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Spotify Album Downloader")
    parser.add_argument("--auth", action="store_true", help="Authenticate with Spotify")
    parser.add_argument(
        "--download", action="store_true", help="Download all liked albums"
    )
    parser.add_argument(
        "--fetch-albums",
        action="store_true",
        help="Save liked albums to liked_albums.json",
    )
    parser.add_argument(
        "--fetch-artists", action="store_true", help="Fetch and save artist data"
    )
    parser.add_argument(
        "--compress", action="store_true", help="Compress all music files"
    )
    parser.add_argument(
        "--max-size", type=int, default=None, help="Max file size MB for compression"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Concurrent download threads (default: 4)",
    )
    parser.add_argument(
        "--replace-song",
        nargs=2,
        metavar=("SONG_NAME", "YOUTUBE_URL"),
        help="Replace a song by name using YouTube URL",
    )
    args = parser.parse_args()

    if args.auth:
        authenticate_cli()
        return

    if args.compress:
        console.print(
            Panel(Text("Compression Mode", style="bold yellow"), style="bold")
        )
        dm = DownloadManager()
        if args.max_size:
            dm._max_file_size_mb = args.max_size
        dm.compress_all_music()
        return

    if args.replace_song:
        song_name, youtube_url = args.replace_song
        console.print(f"\n[bold yellow]Replacing song: {song_name}[/bold yellow]")
        console.print(f"[cyan]YouTube URL:[/cyan] {youtube_url}\n")

        base_path = os.path.join(os.getcwd(), os.getenv("DOWNLOAD_PATH", "songs"))
        os.makedirs(base_path, exist_ok=True)

        # Remove existing file if present
        existing_file = os.path.join(base_path, f"{song_name}.mp3")
        if os.path.exists(existing_file):
            os.remove(existing_file)
            console.print(f"[yellow]Removed existing file: {song_name}.mp3[/yellow]")

        # Download directly from the provided YouTube URL
        album_path = os.path.join(base_path, song_name.split("/")[0])
        only_song_name = "/".join(song_name.split("/")[1:])
        temp = f"{only_song_name}_{int(time.time())}"
        opts = {
            "outtmpl": os.path.join(album_path, f"{temp}.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "format": "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best",
            "socket_timeout": 60,
            "retries": 5,
        }

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([youtube_url])

            # Find the downloaded file
            found = [f for f in os.listdir(album_path) if f.startswith(temp)]
            if found:
                temp_path = os.path.join(album_path, found[0])
                ext = os.path.splitext(found[0])[1].lower()
                final_path = os.path.join(album_path, f"{only_song_name}.mp3")

                if ext != ".mp3":
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-y",
                            "-i",
                            temp_path,
                            "-codec:a",
                            "libmp3lame",
                            "-b:a",
                            "320k",
                            "-map",
                            "a",
                            "-loglevel",
                            "error",
                            final_path,
                        ],
                        capture_output=True,
                        timeout=120,
                    )
                    os.remove(temp_path)
                else:
                    os.rename(temp_path, final_path)

                console.print(
                    f"\n[green]✓[/green] Song downloaded successfully: {song_name}.mp3"
                )
            else:
                console.print(f"\n[red]✗[/red] Download failed - no file found")
        except Exception as e:
            console.print(f"\n[red]✗[/red] Failed to download: {e}")
        return

    if args.fetch_albums:
        if not _require_auth():
            return
        albums = get_liked_albums()
        with open("liked_albums.json", "w") as f:
            json.dump(albums, f, indent=2)
        console.print(
            f"[green]✓[/green] Saved {len(albums)} albums to liked_albums.json"
        )
        return

    if args.fetch_artists:
        if not _require_auth():
            return
        albums = get_liked_albums()
        dm = DownloadManager()
        unique = {
            a["id"]: a["name"]
            for album in albums
            for a in album.get("metadata", {}).get("artists_detailed", [])[:1]
            if a.get("id")
        }
        artists_to_fetch = {
            aid: aname
            for aid, aname in unique.items()
            if not dm._artist_data_exists(aname)
        }
        console.print(
            f"Found {len(unique)} unique artists ({len(artists_to_fetch)} new)"
        )
        if artists_to_fetch:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as prog:
                task = prog.add_task("Fetching artists...", total=len(artists_to_fetch))
                for i, (aid, aname) in enumerate(artists_to_fetch.items()):
                    prog.update(
                        task,
                        description=f"[{i + 1}/{len(artists_to_fetch)}] {aname[:40]}",
                    )
                    dm.fetch_and_save_artist_data(aid, aname)
                    prog.advance(task)
            console.print(
                f"[green]✓[/green] Saved {len(artists_to_fetch)} artists to artists/"
            )
        else:
            console.print("[yellow]All artist data already exists - skipping[/yellow]")
        return
        albums = get_liked_albums()
        dm = DownloadManager()
        unique = {
            a["id"]: a["name"]
            for album in albums
            for a in album.get("metadata", {}).get("artists_detailed", [])[:1]
            if a.get("id")
        }
        artists_to_fetch = {
            aid: aname
            for aid, aname in unique.items()
            if not dm._artist_data_exists(aname)
        }
        console.print(
            f"Found {len(unique)} unique artists ({len(artists_to_fetch)} new)"
        )
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as prog:
            task = prog.add_task("Fetching artists...", total=len(artists_to_fetch))
            for i, (aid, aname) in enumerate(artists_to_fetch.items()):
                prog.update(
                    task, description=f"[{i + 1}/{len(artists_to_fetch)}] {aname[:40]}"
                )
                dm.fetch_and_save_artist_data(aid, aname)
                prog.advance(task)
        console.print(
            f"[green]✓[/green] Saved {len(artists_to_fetch)} artists to artists/"
        )
        return
        albums = get_liked_albums()
        dm = DownloadManager()
        unique = {
            a["id"]: a["name"]
            for album in albums
            for a in album.get("metadata", {}).get("artists_detailed", [])[:1]
            if a.get("id")
        }
        console.print(f"Found {len(unique)} unique artists")
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as prog:
            task = prog.add_task("Fetching artists...", total=len(unique))
            for i, (aid, aname) in enumerate(unique.items()):
                prog.update(task, description=f"[{i + 1}/{len(unique)}] {aname[:40]}")
                dm.fetch_and_save_artist_data(aid, aname)
                prog.advance(task)
        console.print(f"[green]✓[/green] Saved {len(unique)} artists to artists/")
        return

    if args.download:
        if not _require_auth():
            return
        albums = get_liked_albums()
        console.print(
            f"Found [cyan]{len(albums)}[/cyan] albums ({sum(len(a['tracks']) for a in albums)} songs)"
        )
        dm = DownloadManager()
        if args.threads:
            dm.max_concurrent = args.threads
        dm.download_albums_cli(albums)
        return

    console.print(
        Panel(Text("Spotify Album Downloader", style="bold cyan"), style="bold")
    )
    console.print("\n[bold]Usage:[/bold] python3 main.py [options]\n")
    console.print("[bold]Options:[/bold]")
    for flag, desc in [
        ("--auth", "Authenticate with Spotify"),
        ("--download", "Download all liked albums"),
        ("--fetch-albums", "Save liked albums list to JSON"),
        ("--fetch-artists", "Fetch artist data (bio, images, albums)"),
        ("--compress", "Compress all music files"),
        ("--replace-song SONG YOUTUBE_URL", "Replace a song by name using YouTube URL"),
        ("--max-size N", "Max file size MB (aggressive compression above this)"),
        ("--threads N", "Concurrent download threads (default: 4)"),
    ]:
        console.print(f"  {flag:<30} {desc}")
    console.print("\n[cyan]Example:[/cyan] python3 main.py --download --threads 8\n")


if __name__ == "__main__":
    main()
