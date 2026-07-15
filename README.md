# Smart Document Scanner

A modular Python project that turns a phone webcam (or a Raspberry Pi Camera Module) into an automatic document scanner with a built-in web gallery.

## Features

- Live camera preview with on-screen page outline overlay
- Auto-detect the page edges (Canny + contour approximation)
- Perspective-warp + adaptive threshold cleanup
- Auto-numbered `captures/page_1.jpg`, `page_2.jpg`, ...
- One-click PDF generation and QR code for sharing
- Flask web UI listing captures and offering PDF/QR downloads
- Designed to be portable from Windows + DroidCam to a Raspberry Pi 5 + Camera Module 3 with a single flag change

## Project layout

```
DocumentScanner/
├── app.py              # main entry: camera loop + web UI bootstrap
├── camera.py           # Camera abstraction (OpenCV / picamera2)
├── detector.py         # DocumentDetector — finds the 4 corner points
├── scanner.py          # DocumentScanner — perspective warp + cleanup
├── pdf_generator.py    # PDFGenerator — combines pages into a PDF
├── qr_generator.py     # QRGenerator — builds a QR PNG
├── webserver.py        # Flask app factory + routes
├── requirements.txt
├── captures/           # captured page images (auto-numbered)
├── output/             # generated PDFs and QR PNGs
└── templates/          # HTML for the gallery UI
```

## Setup (Windows)

```powershell
cd "d:\Codes\KUET PROJECTS\3 1 Embedded\DocsMaker\DocumentScanner"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Start DroidCam on your phone and copy the IP Webcam / DroidCam URL.

```powershell
# Default webcam
python app.py

# DroidCam / IP Webcam URL
python app.py --source http://192.168.1.10:4747/video

# Disable the web UI if you only want the desktop window
python app.py --no-web
```

Then open <http://localhost:5000> in your browser to see the gallery.

### Hotkeys in the camera window

The OpenCV window is a tiny state machine with three screens. The active
screen is shown in the footer hint text.

| Screen          | Key       | Action                                                                                  |
| --------------- | --------- | --------------------------------------------------------------------------------------- |
| **Camera**      | `C`       | Capture the current page. If a document is detected it is warped + binarised; otherwise the whole frame is added to the PDF as-is. |
| **Camera**      | `D`       | Finish the current session — saves `output/scan_N.pdf`, writes a dummy QR code, and switches the window to the **PDF saved** screen. |
| **Camera**      | `N`       | If pages are pending, switch to the **Confirm new PDF** dialog (Y/N) inside the window. With zero pages captured, a toast reminds you the session is already fresh. |
| **Camera**      | `Q`       | Quit. If there are pages that haven't been finished yet, `Q` first runs `D` automatically; otherwise it asks for a save/discard confirmation. |
| **PDF saved**   | `Y` / `↵` | Acknowledge the saved PDF and return to the **Camera** screen for the next session.    |
| **PDF saved**   | `N`       | Open the **Confirm new PDF** dialog from the saved screen.                             |
| **PDF saved**   | `Q`       | Quit (the PDF is already on disk, so no save prompt is needed).                        |
| **PDF saved**   | `C`       | Ignored — there is no camera feed in this screen.                                       |
| **Confirm new** | `Y`       | Discard the current pages, reset the page counter, and return to **Camera**.           |
| **Confirm new** | `N` / `Esc` | Cancel the new-session request and return to **Camera** with the pages intact.       |
| **Confirm new** | any other | Ignored — the dialog waits for an explicit Y/N choice.                                 |

## Web UI — live 10-panel pipeline + remote control

The Flask server now auto-starts with the app (no need to wait for the first
`D`) and serves a **live** browser frontend at `http://<host>:5000`. Because the
host binds to `0.0.0.0`, any phone/laptop on the same Wi-Fi can open it.

The page shows the **same 10 pipeline images** the desktop window renders,
arranged as **5 on top + 5 on bottom**:

| Top row | Bottom row |
| --- | --- |
| Original | Selected contour |
| Gray | Colored page doc |
| Binary (Canny edges) | Grey page doc |
| Contours | Black & white |
| Present (what `C` saves) | Last captured (static) |

- **Click any panel** → a popup enlarges that stage's live stream. Every other
  panel keeps streaming and every control keeps working while the popup is open.
- **Full remote control**: the on-page buttons (and the `C` / `D` / `X` / `N` /
  `Q` keyboard shortcuts) drive the scanner over HTTP — capture, finish PDF,
  delete last page, new session, quit — so you can run it entirely from a phone.
- The desktop OpenCV window still works exactly as before; both surfaces share
  one session (a lock keeps page numbering contiguous when both fire at once).

Streaming uses browser-native MJPEG (`/stream/<stage>`, downscaled for the grid,
full-res in the popup) plus `/frame/<stage>.jpg` snapshots — no JavaScript
frameworks, no CDN.

Run **headless** (no monitor) with `--headless`: the camera loop and web UI run,
the OpenCV window is skipped, and the browser becomes the only interface.

## Migrating to Raspberry Pi 5 + 16 MP Camera Module

1. Install the Pi OS (Bookworm) system packages:

   ```bash
   sudo apt update
   sudo apt install -y python3-opencv python3-picamera2 ffmpeg
   ```

2. Create the virtualenv **with system packages visible** so it can import the
   apt-installed `picamera2` and `cv2` (this is the usual gotcha):

   ```bash
   cd ~/DocumentScanner
   python3 -m venv venv --system-site-packages
   source venv/bin/activate
   pip install -r requirements.txt
   pip install pydub pyalsaaudio        # optional: MP3 audio cues on the I2S amp
   ```

3. Run it. The web UI auto-starts on `http://<pi-ip>:5000`.

   ```bash
   # With a monitor attached (this is the command the project is driven with):
   python3 app.py --backend picamera2 --rotate 270 --autofocus

   # No monitor? Add --headless (OpenCV window skipped, web UI is the whole UI):
   python3 app.py --backend picamera2 --rotate 270 --autofocus --headless
   ```

   `camera.py` automatically picks the picamera2 code path — no other file needs
   to change.

4. Open the UI from your phone/laptop on the same Wi-Fi. Find the Pi's address
   with `hostname -I`, then browse to `http://<that-ip>:5000`. From there you can
   watch the live 10-panel pipeline and capture / finish / download PDFs.

## Audio cues (Pi 5 + MAX98357A I2S amp)

When the scanner runs on the Pi 5 with the optional MAX98357A mono amplifier
and a 5 W speaker, two layers of audio feedback are active:

- **Tone cues** (`sound.py`) — short synthesized WAV blips on capture / delete.
- **Spoken prompts** (`voice.py`) — TTS phrases via `espeak-ng`.
- **Long-form MP3 cues** (`mp3_player.py`) — user-supplied clips played
  directly on the I2S amp via `pydub` + `pyalsaaudio`.  Drop `captured.mp3`
  and `deleted.mp3` into the project root and they fire on the matching
  lifecycle event.

### Enabling the I2S amp

Add `dtoverlay=max98357a` to `/boot/firmware/config.txt` and reboot.  The
amplifier registers as card 2, device 0 (`plughw:2,0`).

### Installing the Pi-side dependencies

```bash
sudo apt update
sudo apt install -y ffmpeg                     # MP3 decoding (pydub shells out to it)
pip3 install pydub pyalsaaudio                 # Python audio pipeline
```

Drop your clips into the project root:

```
DocumentScanner/
├── captured.mp3    # plays on a successful page capture (manual C, auto, etc.)
└── deleted.mp3     # plays when the last page is removed via X
```

CLI overrides: `--mp3 / --no-mp3`, `--mp3-captured <path>`,
`--mp3-deleted <path>`, `--mp3-device plughw:2,0`, `--mp3-volume 8.0`.

The pipeline is the one we proved out manually:

```
MP3 -> ffmpeg (decode) -> pydub (mono, 48 kHz, S16_LE, +8 dB) ->
pyalsaaudio (plughw:2,0, 32768-byte chunks) -> MAX98357A -> speaker
```

## Generating the PDF and QR

From the web UI, click **Build PDF** to render `output/scan.pdf`. Click **Build QR**, enter a URL (e.g. the local server address printed by the app), and a `qrcode.png` is written into `output/`.

From the command line:

```python
from pathlib import Path
from pdf_generator import PDFGenerator
from qr_generator import QRGenerator

PDFGenerator(Path("captures"), Path("output")).build_pdf("scan.pdf")
QRGenerator(Path("output")).make("http://192.168.1.10:5000/download/scan.pdf")
```
