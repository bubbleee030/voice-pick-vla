"""Grab one JPEG frame from each live MJPEG camera stream and save to disk."""
import urllib.request, os, datetime, sys

URL = "http://127.0.0.1:8090/stream/{cam}/{st}"
TARGETS = [("cam1", "rgb"), ("cam2", "rgb"), ("claw", "rgb")]
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
outdir = "data/runtime/snapshots"
os.makedirs(outdir, exist_ok=True)

for cam, st in TARGETS:
    url = URL.format(cam=cam, st=st)
    try:
        r = urllib.request.urlopen(url, timeout=8)
        buf = b""
        # read until we have one full JPEG (FFD8 ... FFD9)
        while len(buf) < 4_000_000:
            chunk = r.read(8192)
            if not chunk:
                break
            buf += chunk
            s = buf.find(b"\xff\xd8")
            e = buf.find(b"\xff\xd9", s + 2) if s != -1 else -1
            if s != -1 and e != -1:
                jpg = buf[s:e + 2]
                path = os.path.join(outdir, f"{cam}_{st}_{ts}.jpg")
                with open(path, "wb") as f:
                    f.write(jpg)
                print(f"{cam}/{st}: saved {path} ({len(jpg)} bytes)")
                break
        else:
            print(f"{cam}/{st}: no full frame in buffer")
        r.close()
    except Exception as exc:
        print(f"{cam}/{st}: ERROR {exc}")
