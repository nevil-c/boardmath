"""List working camera indices and their native resolution (USB cam = 'HD Web Camera')."""
import cv2

for i in range(6):
    cap = cv2.VideoCapture(i)
    if not cap.isOpened():
        cap.release()
        continue
    ok, f = cap.read()
    print(i, "opened", end=" ")
    if ok and f is not None:
        print(f"frame {f.shape[1]}x{f.shape[0]} mean={f.mean():.1f}")
        cv2.imwrite(f"/tmp/cam{i}.jpg", f)
    else:
        print("no frame (permission?)")
    cap.release()
