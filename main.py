import os
import json
import time
import webbrowser
import re
import subprocess
import argparse
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import yt_dlp
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import requests
from dotenv import load_dotenv


load_dotenv()

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
REDIRECT_URI = os.getenv("REDIRECT_URI")

SCOPE = "user-library-read user-read-recently-played"

auth_manager = SpotifyOAuth(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri=REDIRECT_URI,
    scope=SCOPE,
    show_dialog=True,
)

sp = None


def save_token(token_info):

    with open(".spotify_token.json", "w") as f:
        json.dump(token_info, f)
    print("✓ Token saved to .spotify_token.json")


def load_spotify_token():

    global sp

    if not os.path.exists(".spotify_token.json"):
        return False

    try:
        with open(".spotify_token.json", "r") as f:
            token_info = json.load(f)

        if auth_manager.is_token_expired(token_info):
            print("Token expired, refreshing...")
            token_info = auth_manager.refresh_access_token(token_info["refresh_token"])
            save_token(token_info)

        sp = spotipy.Spotify(auth=token_info["access_token"])

        user = sp.current_user()
        if not user:
            print("Error: Failed to fetch user details")
            return False
        print(f"✓ Authenticated as: {user.get('display_name', 'Unknown')}")
        return True

    except Exception as e:
        print(f"Error loading token: {e}")
        return False


def authenticate_cli():

    global sp

    print("Opening browser for Spotify authentication...")
    print("Please authorize the application in your browser.")
    print()

    auth_url = auth_manager.get_authorize_url()
    print(f"Authorization URL: {auth_url}")
    print()

    try:
        webbrowser.open(auth_url)
        print("Browser opened. Please complete authorization.")
    except:
        print("Could not open browser automatically.")
        print("Please manually open the URL above.")

    print()
    print("After authorizing, you'll be redirected to a URL like:")
    print(f"  {REDIRECT_URI}?code=XXXXX...")
    print()
    auth_code = input(
        "Paste the FULL redirect URL here (or just the 'code' parameter): "
    ).strip()

    if "code=" in auth_code:
        import urllib.parse

        parsed = urllib.parse.urlparse(auth_code)
        params = urllib.parse.parse_qs(parsed.query)
        auth_code = params.get("code", [""])[0]

    if not auth_code:
        print("Error: No authorization code provided")
        return False

    try:
        print("\nExchanging code for token...")
        token_info = auth_manager.get_access_token(auth_code)

        if not token_info or "access_token" not in token_info:
            print("Error: Failed to get access token")
            return False

        sp = spotipy.Spotify(auth=token_info["access_token"])

        user = sp.current_user()
        if not user:
            print("Error: Failed to fetch user details")
            return False
        print(f"✓ Successfully authenticated as: {user.get('display_name', 'Unknown')}")

        save_token(token_info)
        print("✓ Authentication complete!")
        print()
        print("You can now use:")
        print("  --download      Download all liked albums")
        print("  --fetch-albums  Fetch and save album list")

        return True

    except Exception as e:
        print(f"Authentication failed: {e}")
        return False


def get_liked_albums():
    global sp
    if sp is None:
        raise Exception("Not authenticated. Please log in first.")

    print("Fetching liked albums...")
    albums = []
    try:
        results = sp.current_user_saved_albums(limit=50)
    except Exception as e:
        raise Exception(f"Failed to fetch albums from Spotify: {str(e)}")

    if not results:
        print("No results returned from Spotify")
        return albums

    while results:
        if not results.get("items"):
            break

        for item in results["items"]:
            album = item["album"]
            albums.append(
                {
                    "name": album["name"],
                    "artists": [a["name"] for a in album["artists"]],
                    "tracks": [t["name"] for t in album["tracks"]["items"]],
                    "metadata": {
                        "id": album["id"],
                        "uri": album["uri"],
                        "external_urls": album.get("external_urls", {}),
                        "release_date": album.get("release_date"),
                        "release_date_precision": album.get("release_date_precision"),
                        "total_tracks": album.get("total_tracks"),
                        "album_type": album.get("album_type"),
                        "genres": album.get("genres", []),
                        "label": album.get("label"),
                        "popularity": album.get("popularity"),
                        "images": album.get("images", []),
                        "copyrights": album.get("copyrights", []),
                        "artists_detailed": [
                            {
                                "id": artist["id"],
                                "name": artist["name"],
                                "uri": artist["uri"],
                                "external_urls": artist.get("external_urls", {}),
                            }
                            for artist in album["artists"]
                        ],
                        "tracks_detailed": [
                            {
                                "id": track["id"],
                                "name": track["name"],
                                "track_number": track["track_number"],
                                "disc_number": track["disc_number"],
                                "explicit": track["explicit"],
                                "duration_ms": track.get("duration_ms"),
                                "uri": track["uri"],
                                "external_urls": track.get("external_urls", {}),
                                "artists": [
                                    {
                                        "id": artist["id"],
                                        "name": artist["name"],
                                        "uri": artist["uri"],
                                    }
                                    for artist in track["artists"]
                                ],
                            }
                            for track in album["tracks"]["items"]
                        ],
                        "added_at": item.get("added_at"),
                    },
                }
            )
        if results["next"]:
            results = sp.next(results)
        else:
            break
    return albums


def get_recently_played():
    global sp
    if sp is None:
        raise Exception("Not authenticated. Please log in first.")

    print("Fetching recently played tracks...")
    tracks = []
    try:
        results = sp.current_user_recently_played(limit=50)
    except Exception as e:
        raise Exception(f"Failed to fetch recently played from Spotify: {str(e)}")

    if not results or not results.get("items"):
        print("No recently played tracks found")
        return tracks

    for item in results["items"]:
        track = item["track"]
        tracks.append(
            {
                "name": track["name"],
                "artists": [a["name"] for a in track["artists"]],
                "album": track["album"]["name"],
            }
        )
    return tracks


class DownloadManager:

    _instance = None
    _progress = {
        "current_album": "",
        "current_song": "",
        "album_progress": 0,
        "song_progress": 0,
        "total_albums": 0,
        "completed_albums": 0,
        "total_songs": 0,
        "completed_songs": 0,
        "failed_downloads": 0,
        "skipped_songs": 0,
        "status": "idle",
        "error": None,
        "concurrent_downloads": 0,
        "max_concurrent": int(os.getenv("MAX_CONCURRENT_DOWNLOADS", 4)),
        "threads": {},
        "metadata_progress": 0,
        "metadata_completed": 0,
        "compression": {
            "status": "idle",
            "total_files": 0,
            "completed_files": 0,
            "current_file": "",
            "progress": 0,
            "space_saved": 0,
            "original_size": 0,
            "compressed_size": 0,
        },
    }

    def __init__(self):
        self._download_lock = threading.Lock()
        self._progress_lock = threading.Lock()
        self._max_file_size_mb = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DownloadManager, cls).__new__(cls)
        return cls._instance

    def get_progress(self):
        return self._progress.copy()

    def update_progress(self, **kwargs):
        with self._progress_lock:
            self._progress.update(kwargs)
            if "completed_albums" in kwargs or "completed_songs" in kwargs:
                print(
                    f"Progress update: Albums {self._progress['completed_albums']}/{self._progress['total_albums']}, "
                    f"Songs {self._progress['completed_songs']}/{self._progress['total_songs']}"
                )

    def _progress_hook(self, d, thread_id):
        if d["status"] == "downloading":
            try:
                downloaded_bytes = d.get("downloaded_bytes", 0)
                total_bytes = d.get("total_bytes", 0) or d.get(
                    "total_bytes_estimate", 0
                )

                if total_bytes > 0:
                    progress_val = (downloaded_bytes / total_bytes) * 100
                else:
                    percent_str = d.get("_percent_str", "0%").replace("%", "").strip()
                    try:
                        progress_val = float(percent_str)
                    except:
                        progress_val = 0
                speed = d.get("_speed_str", "Unknown")
                eta = d.get("_eta_str", "Unknown")
                self.update_thread_progress(
                    thread_id,
                    progress=progress_val,
                    status="downloading",
                    speed=speed,
                    eta=eta,
                )
                if thread_id == 0:
                    print(
                        f"Thread {thread_id}: Download progress: {progress_val:.1f}% (Speed: {speed}, ETA: {eta})"
                    )
                if thread_id == 0:
                    self.update_progress(song_progress=progress_val)

            except Exception as e:
                print(f"Progress hook error for thread {thread_id}: {e}")
                pass
        elif d["status"] == "finished":
            print(f"Thread {thread_id}: Download finished")
            self.update_thread_progress(thread_id, progress=100, status="processing")
            if thread_id == 0:
                self.update_progress(song_progress=100)
        elif d["status"] == "error":
            print(f"Thread {thread_id}: Download error in progress hook")
            self.update_thread_progress(
                thread_id, status="failed", error="Download failed"
            )

    def update_thread_progress(self, thread_id, **kwargs):
        with self._progress_lock:
            if thread_id not in self._progress["threads"]:
                self._progress["threads"][thread_id] = {
                    "current_song": "",
                    "status": "idle",
                    "progress": 0,
                    "error": None,
                    "speed": "",
                    "eta": "",
                }
            self._progress["threads"][thread_id].update(kwargs)

    def sanitize_filename(self, filename):
        filename = re.sub(r'[<>:"/\\|?*]', "", filename)
        filename = filename.strip()
        return filename[:200]

    def search_youtube(self, song_name, artist_name, album_name=None, duration_ms=None):

        try:

            if album_name:
                query = f"{song_name} {artist_name} {album_name} official audio"
            else:
                query = f"{song_name} {artist_name} official audio"

            ydl_opts = {
                "quiet": True,
                "no_warnings": True,
                "extract_flat": False,
                "default_search": "ytsearch5:",
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                search_results = ydl.extract_info(f"ytsearch5:{query}", download=False)

                if (
                    not search_results
                    or "entries" not in search_results
                    or not search_results["entries"]
                ):
                    return None

                if duration_ms:
                    target_duration = duration_ms / 1000
                    best_match = None
                    best_score = float("inf")

                    for entry in search_results["entries"]:
                        if not entry:
                            continue

                        video_duration = entry.get("duration", 0)
                        title_lower = entry.get("title", "").lower()

                        duration_diff = abs(video_duration - target_duration)

                        title_bonus = 0
                        if "official" in title_lower or "audio" in title_lower:
                            title_bonus = -5
                        if "lyrics" in title_lower or "lyric video" in title_lower:
                            title_bonus = -3
                        if (
                            "cover" in title_lower
                            or "remix" in title_lower
                            or "live" in title_lower
                        ):
                            title_bonus = 10

                        score = duration_diff + title_bonus

                        if score < best_score:
                            best_score = score
                            best_match = entry

                    if best_match:
                        video_id = best_match["id"]
                        print(
                            f"  → Matched with duration {best_match.get('duration', 0)}s (target: {target_duration:.0f}s)"
                        )
                        return f"https://www.youtube.com/watch?v={video_id}"

                video_id = search_results["entries"][0]["id"]
                return f"https://www.youtube.com/watch?v={video_id}"

        except Exception as e:
            print(f"YouTube search error: {e}")
            return None

    def check_song_exists(self, song_name, album_path):

        try:
            base_filename = song_name
            existing_files = [
                f
                for f in os.listdir(album_path)
                if f.startswith(base_filename)
                and not f.startswith(".")
                and "_thread" not in f
            ]
            return len(existing_files) > 0
        except OSError:
            return False

    def download_song(self, song_info, album_path, thread_id=0):
        song_name = self.sanitize_filename(song_info["name"])
        artist_name = self.sanitize_filename(" & ".join(song_info["artists"]))
        album_name = song_info.get("album_name", "")
        duration_ms = song_info.get("duration_ms")

        start_time = time.time()
        self.update_thread_progress(
            thread_id,
            current_song=f"{song_name} by {artist_name}",
            status="checking",
            progress=0,
            error=None,
        )

        if self.check_song_exists(song_name, album_path):
            print(f"Thread {thread_id}: Skipping {song_name} - already exists")
            self.update_thread_progress(thread_id, status="skipped", progress=100)
            self.update_progress(
                completed_songs=self._progress["completed_songs"] + 1,
                skipped_songs=self._progress["skipped_songs"] + 1,
            )
            return True

        self.update_progress(
            current_song=f"[Thread {thread_id}] {song_name} by {artist_name}",
            concurrent_downloads=self._progress["concurrent_downloads"] + 1,
        )

        try:
            self.update_thread_progress(thread_id, status="searching")
            print(f"Thread {thread_id}: Starting search for {song_name}")

            search_start = time.time()
            youtube_url = self.search_youtube(
                song_name, artist_name, album_name, duration_ms
            )
            search_time = time.time() - search_start
            print(f"Thread {thread_id}: Search took {search_time:.2f}s")

            if not youtube_url:
                print(f"Thread {thread_id}: No YouTube link found for {song_name}")
                self.update_thread_progress(
                    thread_id, status="failed", error="No YouTube link found"
                )
                self.update_progress(
                    failed_downloads=self._progress["failed_downloads"] + 1,
                    concurrent_downloads=self._progress["concurrent_downloads"] - 1,
                )
                return False

            self.update_thread_progress(thread_id, status="downloading")
            print(
                f"Thread {thread_id}: Starting download for {song_name} from {youtube_url}"
            )
            temp_filename = f"{song_name}_thread{thread_id}_{int(time.time())}"
            ydl_opts = {
                "format": "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best",
                "outtmpl": os.path.join(album_path, f"{temp_filename}.%(ext)s"),
                "quiet": True,
                "no_warnings": True,
                "progress_hooks": [lambda d: self._progress_hook(d, thread_id)],
                "extractaudio": True,
                "audioformat": "mp3",
                "audioquality": "0",
                "prefer_ffmpeg": True,
                "socket_timeout": 30,
                "retries": 3,
                "fragment_retries": 3,
                "skip_unavailable_fragments": True,
                "concurrent_fragment_downloads": 2,
                "postprocessor_args": [
                    "-ar",
                    "44100",
                ],
            }

            download_start = time.time()

            try:
                ydl_opts["postprocessors"] = [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "mp3",
                        "preferredquality": "320",
                    }
                ]

                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([youtube_url])

                download_time = time.time() - download_start
                print(f"Thread {thread_id}: Download completed in {download_time:.2f}s")
                temp_files = [
                    f for f in os.listdir(album_path) if f.startswith(temp_filename)
                ]
                if temp_files:
                    temp_path = os.path.join(album_path, temp_files[0])
                    final_path = os.path.join(album_path, f"{song_name}.mp3")
                    os.rename(temp_path, final_path)
                    print(f"Thread {thread_id}: File renamed to {song_name}.mp3")

            except Exception as ffmpeg_error:
                if (
                    "ffmpeg" in str(ffmpeg_error).lower()
                    or "postprocessor" in str(ffmpeg_error).lower()
                ):
                    print(
                        f"Thread {thread_id}: FFmpeg issue, downloading original format for {song_name}"
                    )
                    ydl_opts_fallback = {
                        "format": "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio",
                        "outtmpl": os.path.join(album_path, f"{temp_filename}.%(ext)s"),
                        "quiet": True,
                        "no_warnings": True,
                        "progress_hooks": [lambda d: self._progress_hook(d, thread_id)],
                        "socket_timeout": 30,
                        "retries": 3,
                        "fragment_retries": 3,
                        "skip_unavailable_fragments": True,
                    }

                    with yt_dlp.YoutubeDL(ydl_opts_fallback) as ydl:
                        ydl.download([youtube_url])

                    download_time = time.time() - download_start
                    print(
                        f"Thread {thread_id}: Fallback download completed in {download_time:.2f}s"
                    )
                    temp_files = [
                        f for f in os.listdir(album_path) if f.startswith(temp_filename)
                    ]
                    if temp_files:
                        temp_path = os.path.join(album_path, temp_files[0])
                        extension = os.path.splitext(temp_files[0])[1]
                        final_path = os.path.join(album_path, f"{song_name}{extension}")
                        os.rename(temp_path, final_path)
                        print(
                            f"Thread {thread_id}: File renamed to {song_name}{extension}"
                        )
                else:
                    raise ffmpeg_error

            total_time = time.time() - start_time
            print(f"Thread {thread_id}: Total time for {song_name}: {total_time:.2f}s")

            self.update_thread_progress(thread_id, status="completed", progress=100)

            self.update_progress(
                completed_songs=self._progress["completed_songs"] + 1,
                concurrent_downloads=self._progress["concurrent_downloads"] - 1,
            )
            return True

        except Exception as e:
            error_time = time.time() - start_time
            print(
                f"Thread {thread_id}: Download error for {song_name} after {error_time:.2f}s: {e}"
            )
            self.update_thread_progress(thread_id, status="failed", error=str(e))
            self.update_progress(
                failed_downloads=self._progress["failed_downloads"] + 1,
                concurrent_downloads=self._progress["concurrent_downloads"] - 1,
            )
            return False

    def download_image(self, url, file_path):

        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"Downloaded image: {file_path}")
            return True

        except Exception as e:
            print(f"Error downloading image {url}: {e}")
            return False

    def download_album_artwork(self, album_info, album_path):

        global sp
        if sp is None:
            print("Warning: Not authenticated, skipping artist artwork download")
            return False

        try:
            metadata = album_info.get("metadata", {})
            images = metadata.get("images", [])
            if images:
                album_image_url = images[0].get("url")
                if album_image_url:
                    album_cover_path = os.path.join(album_path, "album.png")
                    self.download_image(album_image_url, album_cover_path)
            artists_detailed = metadata.get("artists_detailed", [])
            if artists_detailed:
                primary_artist = artists_detailed[0]
                artist_id = primary_artist.get("id")

                if artist_id:
                    try:
                        artist_details = sp.artist(artist_id)
                        if not artist_details:
                            print(f"Error fetching artist details")
                            return False
                        artist_images = artist_details.get("images", [])

                        if artist_images:
                            artist_image_url = artist_images[0].get("url")
                            if artist_image_url:
                                artist_image_path = os.path.join(
                                    album_path, "artist.png"
                                )
                                self.download_image(artist_image_url, artist_image_path)

                    except Exception as e:
                        print(f"Error fetching artist details: {e}")

            return True

        except Exception as e:
            print(f"Error downloading artwork: {e}")
            return False

    def save_album_metadata(self, album_info, album_path):

        try:
            album_json_path = os.path.join(album_path, "album.json")
            album_metadata = {
                "basic_info": {
                    "name": album_info["name"],
                    "artists": album_info["artists"],
                    "total_tracks": len(album_info["tracks"]),
                },
                "spotify_metadata": album_info.get("metadata", {}),
                "download_info": {
                    "downloaded_at": time.strftime(
                        "%Y-%m-%d %H:%M:%S UTC", time.gmtime()
                    ),
                    "last_updated": time.strftime(
                        "%Y-%m-%d %H:%M:%S UTC", time.gmtime()
                    ),
                    "downloader_version": "1.1",
                    "tracks_list": album_info["tracks"],
                    "artwork_downloaded": {
                        "album_cover": "album.png",
                        "artist_image": "artist.png",
                    },
                },
            }

            with open(album_json_path, "w", encoding="utf-8") as f:
                json.dump(album_metadata, f, indent=2, ensure_ascii=False)

            print(f"Saved album metadata to {album_json_path}")
            return True

        except Exception as e:
            print(f"Error saving album metadata: {e}")
            return False

    def process_album_metadata(self, album_info, album_path):

        try:
            print(f"Processing metadata for: {album_info['name']}")

            self.save_album_metadata(album_info, album_path)

            self.download_album_artwork(album_info, album_path)

            return True
        except Exception as e:
            print(f"Error processing album metadata: {e}")
            return False

    def download_all_metadata_parallel(self, albums, base_path):

        print(f"Starting parallel metadata download for {len(albums)} albums...")

        self.update_progress(
            status="downloading_metadata", metadata_progress=0, metadata_completed=0
        )

        metadata_tasks = []

        for album in albums:
            album_name = self.sanitize_filename(album["name"])
            artist_name = self.sanitize_filename(" & ".join(album["artists"]))
            album_folder = f"{artist_name} - {album_name}"
            album_path = os.path.join(base_path, album_folder)

            os.makedirs(album_path, exist_ok=True)

            metadata_tasks.append((album, album_path))

        max_metadata_workers = min(10, len(albums))
        completed_count = 0

        with ThreadPoolExecutor(max_workers=max_metadata_workers) as executor:
            future_to_album = {
                executor.submit(self.process_album_metadata, album, path): album["name"]
                for album, path in metadata_tasks
            }

            for future in as_completed(future_to_album):
                album_name = future_to_album[future]
                try:
                    result = future.result()
                    completed_count += 1
                    progress = (completed_count / len(albums)) * 100

                    self.update_progress(
                        metadata_completed=completed_count, metadata_progress=progress
                    )

                    print(
                        f"Metadata completed for: {album_name} ({completed_count}/{len(albums)})"
                    )

                except Exception as e:
                    print(f"Error processing metadata for {album_name}: {e}")

        print(f"Completed metadata download for all {len(albums)} albums")

    def download_albums(self, albums):
        self.update_progress(
            status="initializing",
            total_albums=len(albums),
            completed_albums=0,
            total_songs=sum(len(album["tracks"]) for album in albums),
            completed_songs=0,
            failed_downloads=0,
            skipped_songs=0,
            concurrent_downloads=0,
            error=None,
            threads={},
            metadata_progress=0,
            metadata_completed=0,
        )

        base_path = os.path.join(os.getcwd(), os.getenv("DOWNLOAD_PATH", "songs"))
        os.makedirs(base_path, exist_ok=True)

        print("Phase 1: Downloading metadata and artwork...")
        self.download_all_metadata_parallel(albums, base_path)

        print("Phase 2: Preparing song downloads...")
        self.update_progress(status="downloading")

        download_queue = queue.Queue()

        for album_idx, album in enumerate(albums):
            album_name = self.sanitize_filename(album["name"])
            artist_name = self.sanitize_filename(" & ".join(album["artists"]))
            album_folder = f"{artist_name} - {album_name}"
            album_path = os.path.join(base_path, album_folder)

            for track_idx, track in enumerate(album["tracks"]):

                track_details = album.get("metadata", {}).get("tracks_detailed", [])
                duration_ms = None
                if track_idx < len(track_details):
                    duration_ms = track_details[track_idx].get("duration_ms")

                song_info = {
                    "name": track,
                    "artists": album["artists"],
                    "album_name": album_name,
                    "artist_name": artist_name,
                    "album_path": album_path,
                    "album_idx": album_idx,
                    "duration_ms": duration_ms,
                }
                download_queue.put(song_info)

        print(f"Created download queue with {download_queue.qsize()} songs")
        completed_albums_set = set()

        def worker(thread_id):

            while True:
                try:
                    song_info = download_queue.get(timeout=5)
                    current_album = (
                        f"{song_info['album_name']} by {song_info['artist_name']}"
                    )
                    self.update_progress(current_album=current_album)
                    result = self.download_song(
                        song_info, song_info["album_path"], thread_id
                    )
                    download_queue.task_done()
                    self.check_album_completion(
                        albums, song_info["album_idx"], completed_albums_set
                    )

                except queue.Empty:
                    print(f"Thread {thread_id}: No more songs, exiting")
                    break
                except Exception as e:
                    print(f"Thread {thread_id}: Worker error: {e}")
                    download_queue.task_done()

        try:
            max_workers = self._progress["max_concurrent"]
            threads = []

            for thread_id in range(max_workers):
                thread = threading.Thread(target=worker, args=(thread_id,))
                thread.daemon = True
                thread.start()
                threads.append(thread)
                print(f"Started worker thread {thread_id}")

            download_queue.join()

            for thread in threads:
                thread.join(timeout=10)

            self.update_progress(
                status="completed",
                album_progress=100,
                song_progress=100,
                current_album="All downloads completed!",
                current_song="",
                concurrent_downloads=0,
                completed_albums=len(albums),
            )

            print(
                f"All downloads completed! Skipped {self._progress['skipped_songs']} existing songs."
            )

        except Exception as e:
            print(f"Download error: {e}")
            self.update_progress(status="error", error=str(e), concurrent_downloads=0)

    def download_albums_cli(self, albums):

        print(f"Initializing download of {len(albums)} albums...")
        print(
            f"Total songs to download: {sum(len(album['tracks']) for album in albums)}"
        )
        print()

        base_path = os.path.join(os.getcwd(), os.getenv("DOWNLOAD_PATH", "songs"))
        os.makedirs(base_path, exist_ok=True)

        print("=" * 60)
        print("Phase 1: Downloading metadata and artwork")
        print("=" * 60)

        metadata_start = time.time()

        for idx, album in enumerate(albums):
            album_name = self.sanitize_filename(album["name"])
            artist_name = self.sanitize_filename(" & ".join(album["artists"]))
            album_folder = f"{artist_name} - {album_name}"
            album_path = os.path.join(base_path, album_folder)

            os.makedirs(album_path, exist_ok=True)

            print(
                f"[{idx+1}/{len(albums)}] Processing: {album['name']} by {', '.join(album['artists'])}"
            )
            self.save_album_metadata(album, album_path)
            self.download_album_artwork(album, album_path)

        metadata_time = time.time() - metadata_start
        print(f"\n✓ Metadata phase complete in {metadata_time:.1f}s")
        print()

        print("=" * 60)
        print("Phase 2: Downloading songs")
        print("=" * 60)

        download_queue = queue.Queue()

        for album_idx, album in enumerate(albums):
            album_name = self.sanitize_filename(album["name"])
            artist_name = self.sanitize_filename(" & ".join(album["artists"]))
            album_folder = f"{artist_name} - {album_name}"
            album_path = os.path.join(base_path, album_folder)

            for song in album["tracks"]:
                song_info = {
                    "name": song,
                    "artists": album["artists"],
                    "album_name": album_name,
                    "artist_name": artist_name,
                    "album_path": album_path,
                    "album_idx": album_idx,
                }
                download_queue.put(song_info)

        total_songs = download_queue.qsize()
        completed_songs = [0]
        skipped_songs = [0]
        failed_songs = [0]
        completed_albums_set = set()
        progress_lock = threading.Lock()

        print(f"Songs in queue: {total_songs}")
        print()

        def worker(thread_id):

            while True:
                try:
                    song_info = download_queue.get(timeout=5)

                    song_name = self.sanitize_filename(song_info["name"])
                    already_exists = self.check_song_exists(
                        song_name, song_info["album_path"]
                    )

                    result = self.download_song(
                        song_info, song_info["album_path"], thread_id
                    )

                    with progress_lock:
                        if already_exists:
                            skipped_songs[0] += 1
                        elif result:
                            completed_songs[0] += 1
                        else:
                            failed_songs[0] += 1

                        total_done = (
                            completed_songs[0] + skipped_songs[0] + failed_songs[0]
                        )
                        if total_done % 10 == 0 or total_done == total_songs:
                            print(
                                f"Progress: {total_done}/{total_songs} ({completed_songs[0]} new, {skipped_songs[0]} skipped, {failed_songs[0]} failed)"
                            )

                    download_queue.task_done()
                    self.check_album_completion(
                        albums, song_info["album_idx"], completed_albums_set
                    )

                except queue.Empty:
                    break
                except Exception as e:
                    print(f"Thread {thread_id}: Worker error: {e}")
                    with progress_lock:
                        failed_songs[0] += 1
                    download_queue.task_done()

        download_start = time.time()
        max_workers = self._progress["max_concurrent"]
        threads = []

        for thread_id in range(max_workers):
            thread = threading.Thread(target=worker, args=(thread_id,))
            thread.daemon = True
            thread.start()
            threads.append(thread)

        download_queue.join()

        for thread in threads:
            thread.join(timeout=10)

        download_time = time.time() - download_start

        print()
        print("=" * 60)
        print("Download Complete!")
        print("=" * 60)
        print(f"Total albums: {len(albums)}")
        print(f"Completed albums: {len(completed_albums_set)}")
        print(f"Total songs: {total_songs}")
        print(f"Downloaded: {completed_songs[0]}")
        print(f"Skipped (existing): {skipped_songs[0]}")
        print(f"Failed: {failed_songs[0]}")
        print(f"Time taken: {download_time:.1f}s")
        if total_songs > 0:
            print(f"Average: {download_time/total_songs:.1f}s per song")
        print("=" * 60)

    def check_album_completion(self, albums, album_idx, completed_albums_set):

        if album_idx in completed_albums_set:
            return

        album = albums[album_idx]
        album_name = self.sanitize_filename(album["name"])
        artist_name = self.sanitize_filename(" & ".join(album["artists"]))
        album_folder = f"{artist_name} - {album_name}"
        album_path = os.path.join(os.getcwd(), "songs", album_folder)

        try:
            files_in_album = os.listdir(album_path)
            song_files = [
                f
                for f in files_in_album
                if not f.startswith(".")
                and "_thread" not in f
                and f not in ["album.json", "album.png", "artist.png"]
            ]

            if len(song_files) >= len(album["tracks"]):
                completed_albums_set.add(album_idx)
                completed_count = len(completed_albums_set)
                album_progress = (completed_count / len(albums)) * 100

                self.update_progress(
                    completed_albums=completed_count, album_progress=album_progress
                )

                print(
                    f"Album completed: {album_name} ({completed_count}/{len(albums)})"
                )
        except OSError:
            pass

    def get_file_size(self, filepath):

        try:
            return os.path.getsize(filepath)
        except:
            return 0

    def format_size(self, size_bytes):

        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"

    def compress_audio_file(self, filepath):

        temp_output = None
        original_size = 0
        compressed_size = 0
        try:
            original_size = self.get_file_size(filepath)

            if original_size < 1 * 1024 * 1024:
                print(
                    f"Skipping {os.path.basename(filepath)} - already small ({self.format_size(original_size)})"
                )
                return False, original_size, original_size

            filename, ext = os.path.splitext(filepath)
            temp_output = f"{filename}_compressed.mp3"

            original_size_mb = original_size / (1024 * 1024)

            if self._max_file_size_mb and original_size_mb > self._max_file_size_mb:

                if original_size_mb > self._max_file_size_mb * 2:
                    bitrate = "96k"
                    quality = "5"
                    print(
                        f"  Very large file ({self.format_size(original_size)}), using 96kbps (aggressive)"
                    )
                else:
                    bitrate = "128k"
                    quality = "4"
                    print(
                        f"  Large file ({self.format_size(original_size)}), using 128kbps"
                    )

            elif original_size > 8 * 1024 * 1024:
                bitrate = "128k"
                quality = "4"
                print(
                    f"  Large file detected ({self.format_size(original_size)}), using 128kbps"
                )
            else:

                if ext.lower() == ".mp3":
                    print(
                        f"Skipping {os.path.basename(filepath)} - already MP3 and small"
                    )
                    return False, original_size, original_size
                bitrate = "192k"
                quality = "2"

            cmd = [
                "ffmpeg",
                "-i",
                filepath,
                "-codec:a",
                "libmp3lame",
                "-b:a",
                bitrate,
                "-q:a",
                quality,
                "-map",
                "a",
                "-y",
                "-loglevel",
                "error",
                temp_output,
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                print(f"  FFmpeg error for {filepath}: {result.stderr}")
                if os.path.exists(temp_output):
                    os.remove(temp_output)
                return False, original_size, original_size

            compressed_size = self.get_file_size(temp_output)

            min_savings = 0.01 if original_size > 4 * 1024 * 1024 else 0.05

            if compressed_size < original_size * (1 - min_savings):

                backup_path = f"{filename}_original{ext}"

                os.replace(filepath, backup_path)
                os.replace(temp_output, filepath)

                try:
                    os.remove(backup_path)
                except:
                    pass

                space_saved = original_size - compressed_size
                savings_percent = (space_saved / original_size) * 100
                print(
                    f"  ✓ Compressed: {self.format_size(original_size)} → {self.format_size(compressed_size)} (saved {self.format_size(space_saved)}, {savings_percent:.1f}%)"
                )
                return True, original_size, compressed_size
            else:

                if os.path.exists(temp_output):
                    os.remove(temp_output)
                print(
                    f"  Skipping {os.path.basename(filepath)} - compression not beneficial"
                )
                return False, original_size, original_size

        except subprocess.TimeoutExpired:
            print(f"  Compression timeout for {filepath}")
            if temp_output and os.path.exists(temp_output):
                os.remove(temp_output)
            return False, original_size, original_size
        except Exception as e:
            if temp_output and os.path.exists(temp_output):
                print(f"  Error compressing {filepath}: {e}")
                os.remove(temp_output)
            return (
                False,
                original_size if "original_size" in locals() else 0,
                original_size if "original_size" in locals() else 0,
            )

    def compress_all_music(self):

        import subprocess

        with self._progress_lock:
            self._progress["compression"] = {
                "status": "scanning",
                "total_files": 0,
                "completed_files": 0,
                "current_file": "",
                "progress": 0,
                "space_saved": 0,
                "original_size": 0,
                "compressed_size": 0,
            }

        base_path = os.path.join(os.getcwd(), os.getenv("DOWNLOAD_PATH", "songs"))

        if not os.path.exists(base_path):
            with self._progress_lock:
                self._progress["compression"]["status"] = "error"
                self._progress["compression"][
                    "current_file"
                ] = "Songs directory not found"
            print(f"Error: Songs directory not found at {base_path}")
            return

        audio_extensions = [".mp3", ".m4a", ".webm", ".opus", ".ogg", ".wav", ".flac"]
        audio_files = []

        print("Scanning for audio files...")
        for root, dirs, files in os.walk(base_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in audio_extensions):
                    audio_files.append(os.path.join(root, file))

        total_files = len(audio_files)
        print(f"Found {total_files} audio files to compress\n")

        with self._progress_lock:
            self._progress["compression"]["total_files"] = total_files
            self._progress["compression"]["status"] = "compressing"

        if total_files == 0:
            with self._progress_lock:
                self._progress["compression"]["status"] = "completed"
                self._progress["compression"]["current_file"] = "No audio files found"
            print("No audio files found to compress")
            return

        total_original_size = 0
        total_compressed_size = 0
        completed = 0

        for filepath in audio_files:
            filename = os.path.basename(filepath)

            with self._progress_lock:
                self._progress["compression"]["current_file"] = filename
                self._progress["compression"]["completed_files"] = completed
                self._progress["compression"]["progress"] = (
                    completed / total_files
                ) * 100

            print(f"[{completed+1}/{total_files}] Processing: {filename}")

            success, original_size, compressed_size = self.compress_audio_file(filepath)

            total_original_size += original_size
            total_compressed_size += compressed_size
            completed += 1

            with self._progress_lock:
                self._progress["compression"]["original_size"] = total_original_size
                self._progress["compression"]["compressed_size"] = total_compressed_size
                self._progress["compression"]["space_saved"] = (
                    total_original_size - total_compressed_size
                )

        with self._progress_lock:
            self._progress["compression"]["status"] = "completed"
            self._progress["compression"]["completed_files"] = total_files
            self._progress["compression"]["progress"] = 100
            self._progress["compression"]["current_file"] = "Compression complete!"

        space_saved = total_original_size - total_compressed_size
        print(f"\n{'='*60}")
        print(f"Compression complete!")
        print(f"{'='*60}")
        print(f"Original size:    {self.format_size(total_original_size)}")
        print(f"Compressed size:  {self.format_size(total_compressed_size)}")
        print(
            f"Space saved:      {self.format_size(space_saved)} ({(space_saved/total_original_size*100):.1f}%)"
        )
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Spotify Album Downloader")
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Compress all existing music files to save space",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=None,
        help="Maximum file size in MB (files larger will be compressed more aggressively)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download all liked albums (requires authentication first)",
    )
    parser.add_argument(
        "--auth",
        action="store_true",
        help="Authenticate with Spotify (opens browser for OAuth)",
    )
    parser.add_argument(
        "--fetch-albums",
        action="store_true",
        help="Fetch and save list of liked albums to liked_albums.json",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Number of concurrent download threads (default: 4)",
    )

    args = parser.parse_args()

    if args.compress:
        print("Starting compression of all music files...")
        if args.max_size:
            print(f"Target maximum file size: {args.max_size}MB")
        print("Compression settings:")
        print("  - Files >8MB: 128kbps")
        print("  - Files 4-8MB: 160kbps")
        print("  - Files 2-4MB: 192kbps")
        print("  - Files <2MB: 192kbps (skip if already MP3)")
        print("Original files will be replaced after successful compression\n")

        download_manager = DownloadManager()
        if args.max_size:
            download_manager._max_file_size_mb = args.max_size
        download_manager.compress_all_music()

        return

    if args.auth:
        print("Starting Spotify authentication...")
        print("=" * 60)
        authenticate_cli()
        return

    if args.fetch_albums:
        if not load_spotify_token():
            print("Error: Not authenticated. Run with --auth first.")
            return
        print("Fetching liked albums from Spotify...")
        albums = get_liked_albums()
        with open("liked_albums.json", "w") as f:
            json.dump(albums, f, indent=2)
        print(f"✓ Saved {len(albums)} albums to liked_albums.json")
        return

    if args.download:
        if not load_spotify_token():
            print("Error: Not authenticated. Run with --auth first.")
            return

        print("Starting album download...")
        print("=" * 60)

        albums = get_liked_albums()
        print(f"Found {len(albums)} liked albums")
        print(f"Total songs: {sum(len(album['tracks']) for album in albums)}")

        download_manager = DownloadManager()
        if args.threads:
            download_manager._progress["max_concurrent"] = args.threads
            print(f"Using {args.threads} concurrent download threads")

        print("\nStarting downloads...\n")
        download_manager.download_albums_cli(albums)

        return

    print("Spotify Album Downloader")
    print("=" * 60)
    print("Usage: python3 main.py [options]")
    print()
    print("Options:")
    print("  --compress      Compress all existing music files to save space")
    print("  --max-size      Maximum file size in MB (files larger will be compressed more aggressively)")
    print("  --download      Download all liked albums (requires authentication first)")
    print("  --auth          Authenticate with Spotify (opens browser for OAuth)")
    print("  --fetch-albums  Fetch and save list of liked albums to liked_albums.json")
    print("  --threads       Number of concurrent download threads (default: 4)")
    print()
    print("Example: python3 main.py --download --threads 8")
    print()
    print("For more information, visit https://github.com/samuelngs/spotifydown")

if __name__ == "__main__":
    main()
