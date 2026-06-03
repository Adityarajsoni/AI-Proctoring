
import cv2
import time
import argparse
import logging
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from .gaze_tracker import GazeTracker, shutdown

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
)
logger = logging.getLogger(__name__)

# Color map for each state — shown in the window overlay
_STATE_COLORS = {
    "focused":       (0,   220,  0),    # green
    "looking_left":  (0,   0,   220),   # red
    "looking_right": (0,   0,   220),   # red
    "looking_down":  (0,   0,   220),   # red
    "looking_up":    (0,   0,   220),   # red
    "pupil_left":    (0,   165, 255),   # orange
    "pupil_right":   (0,   165, 255),   # orange
    "no_face":       (100, 100, 100),   # grey
}


def run(source=0, show_window=True):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error("Could not open: %s", source)
        return

    tracker     = GazeTracker("TEST_STUDENT")
    INTERVAL    = 0.5    # 2 FPS
    last_time   = 0.0
    risk_events = []

    logger.info("=" * 60)
    logger.info("  Gaze Tracker  |  Q = quit")
    logger.info("  Colors: GREEN=focused  RED=away  ORANGE=pupil  GREY=no_face")
    logger.info("=" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()

        # ── 2 FPS processing ──────────────────────────────────
        if now - last_time >= INTERVAL:
            last_time = now
            event = tracker.process_frame(frame)

            g = tracker.last_gaze   # cached — no double processing
            if g:
                iv = f"{g.iris_vertical:.2f}"   if g.iris_vertical   is not None else "N/A"
                ih = f"{g.iris_horizontal:.2f}" if g.iris_horizontal  is not None else "N/A"

                if event:
                    risk_events.append(event)
                    logger.warning(
                        "RISK EVENT  event=%-30s  severity=%.2f  state=%s",
                        event["event"], event["severity"], event["gaze_state"],
                    )
                else:
                    logger.info(
                        "state=%-14s  yaw=%+6.1f°  pitch=%+6.1f°  iv=%-4s  ih=%-4s",
                        g.state, g.yaw, g.pitch, iv, ih,
                    )

        # ── Display every frame (smooth window) ───────────────
        if show_window:
            display = frame.copy()

            g = tracker.last_gaze
            if g:
                state = g.state
                color = _STATE_COLORS.get(state, (200, 200, 200))

                h_f, w_f = display.shape[:2]

                # Always draw border
                cv2.rectangle(display, (0, 0), (w_f - 1, h_f - 1), color, 12)

                # State label
                label = f" {state.upper()} "
                font  = cv2.FONT_HERSHEY_SIMPLEX
                (tw, th), _ = cv2.getTextSize(label, font, 1.0, 2)
                cv2.rectangle(display, (8, 8), (8 + tw + 4, 8 + th + 10), color, -1)
                cv2.putText(display, label, (10, 10 + th), font, 1.0, (0, 0, 0), 2)

                if state == "no_face":
                    # No face — just show the state, no angles
                    cv2.putText(display, "No face detected",
                                (10, 75), font, 0.6, (100, 100, 100), 1)
                else:
                    iv = f"{g.iris_vertical:.2f}"   if g.iris_vertical   is not None else "N/A"
                    ih = f"{g.iris_horizontal:.2f}" if g.iris_horizontal  is not None else "N/A"
                    cv2.putText(display,
                                f"yaw={g.yaw:+.1f}  pitch={g.pitch:+.1f}",
                                (10, 75), font, 0.52, (220, 220, 220), 1)
                    cv2.putText(display,
                                f"iris_v={iv}  iris_h={ih}",
                                (10, 95), font, 0.52, (220, 220, 220), 1)

                # Last risk event at bottom
                if risk_events:
                    last_evt = risk_events[-1]
                    cv2.putText(display,
                                f"! {last_evt['event']}  sev={last_evt['severity']}",
                                (10, h_f - 15), font, 0.55, (0, 0, 255), 2)

                # If a risk event fired, show it in red at bottom
                if risk_events:
                    last_evt = risk_events[-1]
                    cv2.putText(display,
                                f"! {last_evt['event']}  sev={last_evt['severity']}",
                                (10, h_f - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

            cv2.imshow("Gaze Tracker", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # ── Cleanup ───────────────────────────────────────────────
    cap.release()
    if show_window:
        cv2.destroyAllWindows()

    summary = tracker.end_session()
    shutdown()

    print("\n" + "=" * 60)
    print("  SESSION SUMMARY")
    print("=" * 60)
    print(f"  Student ID      : {summary['student_id']}")
    print(f"  Total frames    : {summary['total_frames_processed']}")
    print(f"  Away frames     : {summary['away_frames']}")
    print(f"  Attention rate  : {summary['attention_rate'] * 100:.1f}%")
    print(f"  Risk events     : {len(risk_events)}")
    if risk_events:
        print("\n  Events fired:")
        for e in risk_events:
            print(f"    [{e['event']}]  sev={e['severity']}  state={e['gaze_state']}")
    print("=" * 60)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video",      type=str, default=None)
    p.add_argument("--no-display", action="store_true")
    args = p.parse_args()
    run(source=args.video or 0, show_window=not args.no_display)