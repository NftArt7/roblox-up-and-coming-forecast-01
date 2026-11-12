#!/usr/bin/env python3
"""Forecast Roblox's Up-and-Coming chart top 10 for every hour of the scored
window (make.fun roblox-up-and-coming-forecast-01).

Model: a per-game rank transition kernel fitted from the input history itself.
For each starting-rank bucket we fit, from all pairs of reads,
  * drift mu (ranks/hour) -- a weighted linear fit of mean rank change vs gap,
  * volatility sigma(dt) = s0 * dt**beta -- a log-log fit of the sd of rank
    change vs gap,
  * an exit hazard e(rank, dt) -- the empirical fraction of games that leave
    the chart, interpolated in log gap.
Each game on the last chart then gets a time-averaged place distribution:
for every hour h of the window, its rank is Normal(r0 + mu*dt + momentum,
sigma(dt)) truncated to the chart, times its survival probability; the
probabilities are averaged over the window's hours.  Momentum is the game's
recent rank velocity (last two gaps, shrunk by 0.5) decaying with a 12h
time constant -- rank velocity persists (corr ~0.22), while playing-level
changes do not predict rank movement (corr ~0.02), so playing is not used.

Because ranks are a permutation, independent per-game normals under-count
the mass at each place (the top places are always filled by *some* known
game).  Each place's probabilities are therefore renormalised to a total
mass of 0.85 -- the empirical share of a place held, over the window, by a
game already on the last chart (the rest is new games joining the chart,
which cannot be named and stay in the unnamed share).

Naming is greedy per place: games are named in decreasing probability while
a game's probability beats the unnamed per-game share, so the named set is
exactly the set worth naming under the scoring rule.  The top10 guess is
the 10 games with the highest total top-10 probability, ordered by their
modal place (then expected rank).

Deterministic: no randomness, no network, no model calls, stdlib only.
Reads $MAKEFUN_INPUT_DIR (else /input) history.ndjson + task.json, writes
$MAKEFUN_OUTPUT_DIR (else /output) forecast.json.
"""

import json
import math
import os
from collections import defaultdict
from datetime import datetime

# Rank buckets for the transition kernel.  Top ranks drift down slowly and
# tightly; deep ranks are volatile and exit often.
BUCKETS = [(1, 3), (4, 10), (11, 20), (21, 40), (41, 70), (71, 100)]
DT_BINS = [(0, 12), (12, 24), (24, 48), (48, 72), (72, 120), (120, 200)]

# Total probability mass assigned to already-known games at each place.
# The remainder is the unnamed share (new games joining during the window).
PLACE_MASS = 0.85

# Momentum: shrink factor for the recent rank velocity, and its decay time.
MOM_SHRINK = 0.5
MOM_TAU_H = 12.0


def bucket_of(rank):
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= rank <= hi:
            return i
    return len(BUCKETS) - 1


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def load_history(input_dir):
    rows = []
    with open(os.path.join(input_dir, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    ts = [parse_time(r["at"]) for r in rows]
    ranks = [{x["universe_id"]: x["rank"] for x in r["up_and_coming"]}
             for r in rows]
    return rows, ts, ranks


def dt_bin(dt):
    for i, (lo, hi) in enumerate(DT_BINS):
        if lo <= dt < hi:
            return i
    return None


def fit(rows, ts, ranks):
    """Fit drift / volatility / exit from all pairs of reads."""
    data = defaultdict(list)          # (bucket, dt_bin) -> list of drank
    exits = defaultdict(lambda: [0, 0])  # (bucket, dt_bin) -> [exited, total]
    n = len(rows)
    for i in range(n):
        for j in range(i + 1, n):
            dt = (ts[j] - ts[i]).total_seconds() / 3600.0
            db = dt_bin(dt)
            if db is None:
                continue
            ri, rj = ranks[i], ranks[j]
            for g, rk in ri.items():
                b = bucket_of(rk)
                exits[(b, db)][1] += 1
                if g in rj:
                    data[(b, db)].append(rj[g] - rk)
                else:
                    exits[(b, db)][0] += 1
    mu = {}
    s0 = {}
    beta = {}
    for b in range(len(BUCKETS)):
        pts = []
        for db, (lo, hi) in enumerate(DT_BINS):
            ds = data.get((b, db), [])
            if len(ds) >= 20:
                pts.append(((lo + hi) / 2.0, sum(ds) / len(ds), len(ds)))
        if len(pts) >= 3:
            num = sum(cnt * dt * m for dt, m, cnt in pts)
            den = sum(cnt * dt * dt for dt, m, cnt in pts)
            mu[b] = num / den if den else 0.0
        else:
            mu[b] = 0.0
        sp = []
        for db, (lo, hi) in enumerate(DT_BINS):
            ds = data.get((b, db), [])
            if len(ds) >= 20:
                m = sum(ds) / len(ds)
                sd = math.sqrt(sum((d - m) ** 2 for d in ds) / len(ds))
                sp.append((math.log((lo + hi) / 2.0),
                           math.log(max(sd, 0.3)), len(ds)))
        if len(sp) >= 3:
            wn = sum(cnt for _, _, cnt in sp)
            mx = sum(cnt * x for x, _, cnt in sp) / wn
            my = sum(cnt * y for _, y, cnt in sp) / wn
            num = sum(cnt * (x - mx) * (y - my) for x, y, cnt in sp)
            den = sum(cnt * (x - mx) ** 2 for x, y, cnt in sp)
            be = num / den if den else 0.5
            beta[b] = min(max(be, 0.3), 1.2)
            s0[b] = math.exp(my - beta[b] * mx)
        else:
            beta[b] = 0.7
            s0[b] = 1.0
    exit_pts = {}
    for b in range(len(BUCKETS)):
        pts = []
        for db, (lo, hi) in enumerate(DT_BINS):
            ex, tot = exits.get((b, db), [0, 0])
            if tot >= 20:
                pts.append((math.log((lo + hi) / 2.0), ex / tot))
        exit_pts[b] = pts
    return mu, s0, beta, exit_pts


def exit_prob(exit_pts, b, dt):
    pts = exit_pts[b]
    if not pts or dt <= 1:
        return 0.0
    x = math.log(dt)
    if x <= pts[0][0]:
        return pts[0][1] * max(0.0, min(1.0, x / pts[0][0])) \
            if pts[0][0] > 0 else pts[0][1]
    for i in range(len(pts) - 1):
        if pts[i][0] <= x <= pts[i + 1][0]:
            f = (x - pts[i][0]) / (pts[i + 1][0] - pts[i][0])
            return pts[i][1] + f * (pts[i + 1][1] - pts[i][1])
    return min(pts[-1][1], 0.85)


def forecast(rows, ts, ranks, hours):
    mu, s0, beta, exit_pts = fit(rows, ts, ranks)
    last = ranks[-1]
    games = list(last.keys())
    # Recent rank velocity (ranks/hour over the last two gaps, shrunk).
    vel = {}
    if len(rows) >= 3:
        for g in games:
            vs = []
            for i in (len(rows) - 2, len(rows) - 1):
                if g in ranks[i] and g in ranks[i - 1]:
                    dt = (ts[i] - ts[i - 1]).total_seconds() / 3600.0
                    if dt > 0:
                        vs.append((ranks[i][g] - ranks[i - 1][g]) / dt)
            if vs:
                vel[g] = MOM_SHRINK * sum(vs) / len(vs)
    # Time-averaged place probabilities.
    pk = {g: [0.0] * 11 for g in games}
    for g in games:
        r0 = last[g]
        b = bucket_of(r0)
        v0 = vel.get(g, 0.0)
        for h in range(hours):
            dt = h + 0.5
            m = r0 + mu[b] * dt + v0 * MOM_TAU_H * (1.0 - math.exp(-dt / MOM_TAU_H))
            sd = max(s0[b] * (dt ** beta[b]), 0.5)
            surv = 1.0 - exit_prob(exit_pts, b, dt)
            if surv <= 0:
                continue
            for k in range(1, 11):
                pk[g][k] += surv * (norm_cdf((k + 0.5 - m) / sd)
                                    - norm_cdf((k - 0.5 - m) / sd)) / hours
    # Renormalise each place to the empirical known-game mass.
    for k in range(1, 11):
        tot = sum(pk[g][k] for g in games)
        if tot > 0:
            f = PLACE_MASS / tot
            for g in games:
                pk[g][k] *= f
    # Greedy naming per place.
    probs = []
    for k in range(1, 11):
        cand = sorted(games, key=lambda g: (-pk[g][k], g))
        named = []
        ssum = 0.0
        for g in cand:
            p = pk[g][k]
            if p <= 0:
                break
            unnamed = (1.0 - ssum) / max(1, len(games) - len(named))
            if p > unnamed:
                named.append(g)
                ssum += p
            else:
                break
        for g in named:
            probs.append({"universe_id": int(g), "place": k,
                          "p": round(pk[g][k], 6)})
    # Top 10: highest total top-10 probability, ordered by modal place.
    total = {g: sum(pk[g][1:]) for g in games}
    exp_r = {g: sum(k * pk[g][k] for k in range(1, 11)) / max(total[g], 1e-9)
             for g in games}
    modal = {g: max(range(1, 11), key=lambda k: (pk[g][k], -k))
             for g in games}
    top = sorted(games, key=lambda g: (-total[g], exp_r[g], g))[:10]
    top10 = sorted(top, key=lambda g: (modal[g], exp_r[g], g))
    return {"probabilities": probs, "top10": [int(g) for g in top10]}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    with open(os.path.join(input_dir, "task.json")) as f:
        task = json.load(f)
    window = task.get("window", {})
    hours = int(window.get("slots") or 0)
    if hours <= 0:
        start = parse_time(window["from"])
        end = parse_time(window["until"])
        hours = max(1, int(round((end - start).total_seconds() / 3600.0)))
    rows, ts, ranks = load_history(input_dir)
    result = forecast(rows, ts, ranks, hours)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "forecast.json"), "w") as f:
        json.dump(result, f, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
