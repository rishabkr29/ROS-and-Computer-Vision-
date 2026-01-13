import json
import time
from typing import List, Tuple

import matplotlib.pyplot as plt
import requests


API_URL = "http://localhost:8080/wheel_rpm"
POLL_HZ = 5.0


def fetch_rpm():
    resp = requests.get(API_URL, timeout=1.0)
    resp.raise_for_status()
    return resp.json()


def main():
    timestamps: List[float] = []
    left: List[float] = []
    right: List[float] = []

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    line_left, = ax.plot([], [], 'b-', label="Left RPM", linewidth=2)
    line_right, = ax.plot([], [], 'r-', label="Right RPM", linewidth=2)
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("RPM", fontsize=12)
    ax.set_title("Wheel RPM Over Time", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    period = 1.0 / POLL_HZ
    start_time = time.time()
    first_stamp = None
    print(f"Polling {API_URL} at {POLL_HZ} Hz. Ctrl+C to stop.")
    print("Waiting for data...")

    try:
        while True:
            try:
                data = fetch_rpm()
                rpm_left = data.get("rpm_left", 0.0)
                rpm_right = data.get("rpm_right", 0.0)
                stamp = data.get("stamp", time.time())
                seq = data.get("seq", 0)
                
                # Use relative time from first data point
                if first_stamp is None:
                    first_stamp = stamp
                
                # Calculate relative time in seconds
                t = time.time() - start_time
                
                timestamps.append(t)
                left.append(rpm_left)
                right.append(rpm_right)
                
                # Update plot data
                line_left.set_data(timestamps, left)
                line_right.set_data(timestamps, right)
                
                # Auto-scale axes
                if len(timestamps) > 0:
                    ax.set_xlim(max(0, timestamps[-1] - 10), timestamps[-1] + 1)
                    all_rpm = left + right
                    if len(all_rpm) > 0:
                        min_rpm = min(all_rpm)
                        max_rpm = max(all_rpm)
                        margin = (max_rpm - min_rpm) * 0.1 if max_rpm != min_rpm else 1.0
                        ax.set_ylim(min_rpm - margin, max_rpm + margin)
                
                # Force redraw
                fig.canvas.draw()
                fig.canvas.flush_events()
                
                # Print status every 5 seconds
                if len(timestamps) % int(POLL_HZ * 5) == 1:
                    print(f"Data points: {len(timestamps)}, Latest: L={rpm_left:.2f} RPM, R={rpm_right:.2f} RPM (seq={seq})")
                
            except requests.exceptions.RequestException as e:
                print(f"Error fetching data: {e}")
                time.sleep(period)
                continue
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(period)
                continue

            time.sleep(period)
    except KeyboardInterrupt:
        print(f"\nStopped. Collected {len(timestamps)} data points.")
    finally:
        plt.ioff()
        if len(timestamps) > 0:
            print("Displaying final plot. Close window to exit.")
            plt.show(block=True)
        else:
            print("No data collected.")


if __name__ == "__main__":
    main()



