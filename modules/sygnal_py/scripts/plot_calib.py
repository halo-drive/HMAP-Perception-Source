#!/usr/bin/env python3
import json, sys, math
import matplotlib.pyplot as plt

def load(path):
    with open(path) as f:
        return json.load(f)

def line(x, m, b): return [m*xi + b for xi in x]

def main(path):
    data = load(path)
    pts = data["raw_points"]
    a   = data["analysis"]
    deadband = a.get("deadband", {})
    db_pct = deadband.get("deadband_percent", 0.0)

    # Extract series
    pct   = [p["cmd_percent"] for p in pts]
    norm  = [p["cmd_norm"]    for p in pts]
    angle = [p["sas_angle_deg"] for p in pts]
    settled_mask = [p["settled"] for p in pts]

    # Linear fit params
    pfit = a["fit_percent_to_angle"]
    nfit = a["fit_norm_to_angle"]

    # 1) Angle vs Percent
    x = pct
    xf = sorted(set(x))
    yf = line(xf, pfit["deg_per_percent"], pfit["offset_deg"])

    plt.figure()
    # settled vs not-settled markers
    xs = [xi for xi, s in zip(x, settled_mask) if s]
    ys = [yi for yi, s in zip(angle, settled_mask) if s]
    xu = [xi for xi, s in zip(x, settled_mask) if not s]
    yu = [yi for yi, s in zip(angle, settled_mask) if not s]

    if xu:
        plt.scatter(xu, yu, label="points (not settled)", marker="x")
    plt.scatter(xs, ys, label="points (settled)")
    plt.plot(xf, yf, label=f"fit: y = {pfit['deg_per_percent']:.2f}x + {pfit['offset_deg']:.1f}\n"
                           f"r² = {pfit['r_squared']:.3f}")
    if abs(db_pct) > 0:
        plt.axvspan(-db_pct, db_pct, alpha=0.1, label=f"deadband ±{db_pct:.2f}%")
    plt.title("Steering angle vs Command (%)")
    plt.xlabel("Command (%)")
    plt.ylabel("SAS angle (deg)")
    plt.grid(True)
    plt.legend()
    out1 = path.replace(".json", "_angle_vs_percent.png")
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    print("Saved:", out1)

    # 2) Angle vs Normalized
    x = norm
    xf = sorted(set(x))
    yf = line(xf, nfit["deg_per_norm"], nfit["offset_deg"])

    plt.figure()
    xs = [xi for xi, s in zip(x, settled_mask) if s]
    ys = [yi for yi, s in zip(angle, settled_mask) if s]
    xu = [xi for xi, s in zip(x, settled_mask) if not s]
    yu = [yi for yi, s in zip(angle, settled_mask) if not s]

    if xu:
        plt.scatter(xu, yu, label="points (not settled)", marker="x")
    plt.scatter(xs, ys, label="points (settled)")
    plt.plot(xf, yf, label=f"fit: y = {nfit['deg_per_norm']:.1f}x + {nfit['offset_deg']:.1f}\n"
                           f"r² = {nfit['r_squared']:.3f}")
    plt.title("Steering angle vs Command (normalized −1…+1)")
    plt.xlabel("Command (normalized)")
    plt.ylabel("SAS angle (deg)")
    plt.grid(True)
    plt.legend()
    out2 = path.replace(".json", "_angle_vs_norm.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    print("Saved:", out2)

    # 3) (Optional) Hysteresis bars (if you kept the same sequence shape)
    # We’ll overlay angle at +|p| vs −|p| for a few magnitudes:
    mags = [10,20,40,70,90,100]
    hv = []
    for m in mags:
        try:
            # closest points by commanded percent
            ppos = min(pts, key=lambda p: abs(p["cmd_percent"]-m))
            pneg = min(pts, key=lambda p: abs(p["cmd_percent"]+m))
            hv.append((m, ppos["sas_angle_deg"], -pneg["sas_angle_deg"]))
        except ValueError:
            pass
    if hv:
        plt.figure()
        ms   = [m for m,_,_ in hv]
        posA = [pa for _,pa,_ in hv]
        negA = [na for _,_,na in hv]
        plt.plot(ms, posA, marker="o", label="+cmd angle")
        plt.plot(ms, negA, marker="o", label="−cmd angle (sign-corrected)")
        plt.title("Hysteresis check (|angle| vs |command|)")
        plt.xlabel("|Command| (%)")
        plt.ylabel("|SAS angle| (deg)")
        plt.grid(True)
        plt.legend()
        out3 = path.replace(".json", "_hysteresis.png")
        plt.savefig(out3, dpi=150, bbox_inches="tight")
        print("Saved:", out3)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: plot_calibration.py <calibration.json>")
        sys.exit(1)
    main(sys.argv[1])
