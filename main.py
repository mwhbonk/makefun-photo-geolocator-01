#!/usr/bin/env python3
"""photo-geolocator-01: a model-free street-photo geolocator.

An agent that looks at street-level photos from anywhere on Earth and says
where each was taken. It is deliberately model-free: it makes no call to any
vision model (no Anthropic/OpenAI key is pushed for this entry, so the gateway
would answer every model call 401), and it places each photo with its own
code alone -- a pure-stdlib baseline-JPEG decoder, interpretable visual
features, and an explicit, reasoned geolocation chain.

How it works, in four stages (the reasoning is printed to stderr):

  1. OBSERVE  -- fetch the instance's photos from the env and decode each
     JPEG to a low-resolution grid of per-block average colours. Only the DC
     (average) coefficient of each 8x8 block is decoded: that is all the
     colour evidence the reasoning below needs, and it keeps the decoder
     small, fast and deterministic.
  2. EVIDENCE -- from the block grid, measure interpretable visual cues:
     vegetation (green dominance), sky (top-band blueness), aridity (warm,
     dry tones), snow (bright and desaturated), road/asphalt (grey, low
     saturation, low in the frame), built/urban grey, and overall brightness.
     These are the cues a human GeoGuessr player reads -- vegetation, road
     markings, sky and palette -- minus the ones that need text: language on
     signs, license plates and architecture details cannot be read without a
     vision model, and this entry does not pretend otherwise.
  3. HYPOTHESISE -- score a fixed set of coarse world regions (each a named
     region with a representative latitude/longitude and a feature signature)
     against the photo's evidence, with a mild prior from the public practice
     set's aggregate country mix. The best-scoring region is the hypothesis.
  4. ANSWER   -- post one latitude/longitude per photo, in photo order, in a
     single POST. A photo that fails to decode still gets an answer (the
     global prior guess): an unanswered photo counts the maximum error
     (20,015 km), so leaving one blank is never better than a guess.

Determinism: no wall clock, no unseeded randomness, no model calls. Env
requests go one at a time, in photo order. Every run on the same instance
leaves the same answers, so the replay matches the live pass exactly.

Environment (set by the trial): MAKEFUN_PROMPT_FILE, MAKEFUN_ENV_URL.
Standard library only. No network beyond the env. No weights, no embeddings,
no table of photos or places: the region list is a fixed, hand-written set of
coarse regions, not a lookup of any photo's place.
"""

import json
import math
import os
import sys
import urllib.error
import urllib.request

# ---------------------------------------------------------------------------
# Stage 1: OBSERVE -- a pure-stdlib baseline-JPEG decoder (DC coefficients
# only). Baseline DCT, 8-bit, 1 or 3 components, 4:4:4 / 4:2:2 / 4:2:0 (and
# 1x2) subsampling, standard or custom Huffman tables, restart markers.
# ---------------------------------------------------------------------------

def _build_huffman(counts, symbols):
    """counts: 16 ints (codes of length 1..16); symbols: the symbols in order.
    Returns {(length, code): symbol}."""
    table = {}
    code = 0
    k = 0
    for length in range(1, 17):
        for _ in range(counts[length - 1]):
            table[(length, code)] = symbols[k]
            k += 1
            code += 1
        code <<= 1
    return table


def _huff_value(category, bits):
    """The signed value of a JPEG additional-bits field of `category` bits."""
    if category == 0:
        return 0
    if bits >= (1 << (category - 1)):
        return bits
    return bits - ((1 << category) - 1)


class _BitReader:
    """MSB-first bit reader over JPEG entropy data.

    Handles 0xFF00 byte stuffing and restart markers (0xFFD0-0xFFD7): a
    restart marker realigns to the byte boundary and resets the caller's DC
    predictors (signalled through .restarted).
    """

    def __init__(self, data, start):
        self.data = data
        self.pos = start
        self.buf = 0
        self.nbits = 0
        self.restarted = False
        self.at_end = False

    def _fill(self):
        while self.nbits <= 24 and self.pos < len(self.data) and not self.at_end:
            b = self.data[self.pos]
            if b == 0xFF:
                nb = self.data[self.pos + 1] if self.pos + 1 < len(self.data) else 0
                if nb == 0x00:
                    self.buf = (self.buf << 8) | 0xFF
                    self.nbits += 8
                    self.pos += 2
                elif 0xD0 <= nb <= 0xD7:
                    self.restarted = True
                    self.buf = 0
                    self.nbits = 0
                    self.pos += 2
                    return
                else:
                    self.at_end = True
                    return
            else:
                self.buf = (self.buf << 8) | b
                self.nbits += 8
                self.pos += 1

    def bits(self, n):
        if n == 0:
            return 0
        while self.nbits < n:
            self._fill()
            if self.nbits < n:
                # Ran out of entropy data: pad with zeroes.
                self.buf <<= (n - self.nbits)
                self.nbits = n
                break
        self.nbits -= n
        value = (self.buf >> self.nbits) & ((1 << n) - 1)
        if self.nbits:
            self.buf &= (1 << self.nbits) - 1
        else:
            self.buf = 0
        return value

    def huff(self, table):
        code = 0
        for length in range(1, 17):
            code = (code << 1) | self.bits(1)
            sym = table.get((length, code))
            if sym is not None:
                return sym
        raise ValueError("invalid Huffman code in JPEG entropy data")


def _parse_jpeg(data):
    """Parse JPEG markers; return (quant, huff_dc, huff_ac, comps, restart_interval, scan_start).

    comps: list of dicts {id, h, v, q} in SOF order. quant: {id: [64]}.
    huff_dc / huff_ac: {(class, id): table}. scan_start: offset of entropy data.
    """
    if len(data) < 4 or data[0] != 0xFF or data[1] != 0xD8:
        raise ValueError("not a JPEG (no SOI)")
    pos = 2
    quant = {}
    huff_dc = {}
    huff_ac = {}
    comps = []
    width = height = 0
    restart_interval = 0
    while pos + 1 < len(data):
        if data[pos] != 0xFF:
            pos += 1
            continue
        marker = data[pos + 1]
        if marker == 0xD9:  # EOI
            break
        if marker == 0xDA:  # SOS: entropy data follows
            seg_len = (data[pos + 2] << 8) | data[pos + 3]
            ncomp = data[pos + 4]
            scan_comps = []
            for i in range(ncomp):
                off = pos + 5 + 2 * i
                cid = data[off]
                td = data[off + 1] >> 4
                ta = data[off + 1] & 0x0F
                scan_comps.append((cid, td, ta))
            return quant, huff_dc, huff_ac, comps, width, height, restart_interval, pos + 2 + seg_len, scan_comps
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            pos += 2
            continue
        seg_len = (data[pos + 2] << 8) | data[pos + 3]
        seg = data[pos + 4:pos + 2 + seg_len]
        if marker == 0xDB:  # DQT
            i = 0
            while i < len(seg):
                pq = seg[i] >> 4
                tq = seg[i] & 0x0F
                i += 1
                vals = []
                for _ in range(64):
                    if pq == 0:
                        vals.append(seg[i])
                        i += 1
                    else:
                        vals.append((seg[i] << 8) | seg[i + 1])
                        i += 2
                quant[tq] = vals
        elif marker == 0xC4:  # DHT
            i = 0
            while i < len(seg):
                tc = seg[i] >> 4
                th = seg[i] & 0x0F
                counts = list(seg[i + 1:i + 17])
                total = sum(counts)
                symbols = list(seg[i + 17:i + 17 + total])
                table = _build_huffman(counts, symbols)
                if tc == 0:
                    huff_dc[th] = table
                else:
                    huff_ac[th] = table
                i += 17 + total
        elif marker == 0xC0:  # SOF0 (baseline DCT)
            height = (seg[1] << 8) | seg[2]
            width = (seg[3] << 8) | seg[4]
            ncomp = seg[5]
            comps = []
            for i in range(ncomp):
                off = 6 + 3 * i
                comps.append({
                    "id": seg[off],
                    "h": seg[off + 1] >> 4,
                    "v": seg[off + 1] & 0x0F,
                    "q": seg[off + 2],
                })
        elif marker == 0xDD:  # DRI
            restart_interval = (seg[0] << 8) | seg[1]
        pos += 2 + seg_len
    raise ValueError("no SOS marker found in JPEG")


def decode_dc_grid(data):
    """Decode a baseline JPEG to a grid of per-block average RGB colours.

    Returns (blocks_w, blocks_h, grid) where grid is a list of rows, each a
    list of (r, g, b) ints in 0..255, one entry per 8x8 luma block. Returns
    None when the JPEG is not a baseline 8-bit image this decoder supports.
    """
    try:
        quant, huff_dc, huff_ac, comps, width, height, restart_interval, scan_start, scan_comps = _parse_jpeg(data)
    except ValueError:
        return None
    if not comps or width <= 0 or height <= 0:
        return None
    if len(comps) not in (1, 3):
        return None
    max_h = max(c["h"] for c in comps)
    max_v = max(c["v"] for c in comps)
    mcu_w = (width + 8 * max_h - 1) // (8 * max_h)
    mcu_h = (height + 8 * max_v - 1) // (8 * max_v)
    # Per-component DC grids, in block units.
    dc_grids = {}
    for c in comps:
        bw = (width * c["h"] + 8 * max_h - 1) // (8 * max_h)
        bh = (height * c["v"] + 8 * max_v - 1) // (8 * max_v)
        dc_grids[c["id"]] = [[0.0] * bw for _ in range(bh)]
    # Map scan component -> SOF component + tables.
    comp_by_id = {c["id"]: c for c in comps}
    readers = {}
    for cid, td, ta in scan_comps:
        if td not in huff_dc or ta not in huff_ac:
            return None
        readers[cid] = (huff_dc[td], huff_ac[ta])
    br = _BitReader(data, scan_start)
    preds = {c["id"]: 0 for c in comps}
    mcu_count = 0
    for my in range(mcu_h):
        for mx in range(mcu_w):
            for c in comps:
                cid = c["id"]
                dc_table, ac_table = readers[cid]
                q0 = quant[c["q"]][0]
                grid = dc_grids[cid]
                bw = len(grid[0])
                for by in range(c["v"]):
                    for bx in range(c["h"]):
                        cat = br.huff(dc_table)
                        diff = _huff_value(cat, br.bits(cat)) if cat else 0
                        preds[cid] += diff
                        # DC coefficient -> block average: avg = dc*q0/8 + 128.
                        avg = preds[cid] * q0 / 8.0 + 128.0
                        gy = my * c["v"] + by
                        gx = mx * c["h"] + bx
                        if gy < len(grid) and gx < bw:
                            grid[gy][gx] = avg
                        # Skip the AC coefficients of this block.
                        for _ in range(63):
                            sym = br.huff(ac_table)
                            if sym == 0x00:  # EOB
                                break
                            if sym == 0xF0:  # ZRL
                                continue
                            size = sym & 0x0F
                            if size:
                                br.bits(size)
                        if br.restarted:
                            br.restarted = False
                            preds = {cc["id"]: 0 for cc in comps}
            mcu_count += 1
            if restart_interval and mcu_count % restart_interval == 0:
                preds = {c["id"]: 0 for c in comps}
            if br.at_end:
                break
        if br.at_end:
            break
    # Assemble the luma-block RGB grid.
    luma = comps[0]
    bw = (width + 7) // 8
    bh = (height + 7) // 8
    y_grid = dc_grids[luma["id"]]
    if len(comps) == 1:
        out = []
        for y in range(bh):
            row = []
            for x in range(bw):
                v = max(0, min(255, int(round(y_grid[y][x]))))
                row.append((v, v, v))
            out.append(row)
        return bw, bh, out
    cb_grid = dc_grids[comps[1]["id"]]
    cr_grid = dc_grids[comps[2]["id"]]
    out = []
    for y in range(bh):
        row = []
        for x in range(bw):
            yy = y_grid[y][x] if y < len(y_grid) and x < len(y_grid[0]) else 128.0
            # Chroma block covering this luma block.
            cy = min(len(cb_grid) - 1, (y * comps[1]["v"]) // max_v)
            cx = min(len(cb_grid[0]) - 1, (x * comps[1]["h"]) // max_h)
            cb = cb_grid[cy][cx] - 128.0
            cr = cr_grid[cy][cx] - 128.0
            r = yy + 1.402 * cr
            g = yy - 0.34414 * cb - 0.71414 * cr
            b = yy + 1.772 * cb
            row.append((max(0, min(255, int(round(r)))),
                        max(0, min(255, int(round(g)))),
                        max(0, min(255, int(round(b))))))
        out.append(row)
    return bw, bh, out


# ---------------------------------------------------------------------------
# Stage 2: EVIDENCE -- interpretable visual cues from the block grid.
# ---------------------------------------------------------------------------

def extract_features(bw, bh, grid):
    """Measure the visual cues of one photo. All features are in 0..1."""
    total = bw * bh
    if total == 0:
        return None
    top_end = max(1, bh // 4)
    bot_start = (3 * bh) // 4
    sums = {"r": 0.0, "g": 0.0, "b": 0.0}
    green_n = sky_n = snow_n = arid_n = urban_n = 0
    top_n = 0
    top_blue = 0.0
    bot_grey = 0
    bot_n = 0
    for y in range(bh):
        for x in range(bw):
            r, g, b = grid[y][x]
            sums["r"] += r
            sums["g"] += g
            sums["b"] += b
            bright = (r + g + b) / 3.0
            sat = max(r, g, b) - min(r, g, b)
            green_excess = g - max(r, b)
            if green_excess > 6:
                green_n += 1
            if bright > 205 and sat < 28:
                snow_n += 1
            if (r - b) > 12 and green_excess < 8:
                arid_n += 1
            if sat < 32 and 55 <= bright <= 195:
                urban_n += 1
            if y < top_end:
                top_n += 1
                if (b - r) > 8 and bright > 105:
                    sky_n += 1
                    top_blue += (b - r)
            if y >= bot_start:
                bot_n += 1
                if sat < 30 and 55 <= bright <= 190:
                    bot_grey += 1
    mean_r = sums["r"] / total
    mean_g = sums["g"] / total
    mean_b = sums["b"] / total
    mean_bright = (mean_r + mean_g + mean_b) / 3.0 / 255.0
    return {
        # fraction of blocks that read as vegetation
        "green": green_n / total,
        # fraction that read as warm, dry ground / rock / sand
        "arid": arid_n / total,
        # fraction of the top band that reads as blue sky
        "sky": (sky_n / top_n) if top_n else 0.0,
        # overall brightness
        "bright": mean_bright,
        # fraction that reads as snow / overcast white
        "snow": snow_n / total,
        # fraction that reads as grey built surfaces (road, concrete)
        "urban": urban_n / total,
        # fraction of the bottom band that is grey road surface
        "road": (bot_grey / bot_n) if bot_n else 0.0,
        # warm colour cast: red minus blue, normalised
        "warm": max(0.0, (mean_r - mean_b) / 255.0),
    }


# ---------------------------------------------------------------------------
# Stage 3: HYPOTHESISE -- coarse world regions, each a named region with a
# representative latitude/longitude and a feature signature (the typical
# street-photo palette of that region). Scores are a weighted distance from
# the photo's evidence to the region's signature, plus a mild log-prior.
# The prior is deliberately small: it comes from the public practice set's
# aggregate country mix (Russia, Brazil, Canada, US, Mexico most common) and
# only nudges ties; the pixels decide. No region is a photo lookup: these are
# coarse centroids, not the place of any photo.
# Feature axes: green, arid, sky, bright, snow, urban, road, warm.
# ---------------------------------------------------------------------------

# (name, lat, lon, prior, {feature: target})
# Targets are calibrated to the public practice set's measured feature means
# for the well-sampled regions (RU, BR, CA/US, MX, AR, IN, JP, NZ) and set by
# the same palette reasoning elsewhere; the scale matches the extractor's
# thresholds (green = blocks with green excess > 6, and so on).
PROFILES = [
    ("Russia west / E. Europe temperate", 55.0, 37.0, 1.20,
     {"green": 0.01, "arid": 0.12, "sky": 0.61, "bright": 0.49, "snow": 0.15, "urban": 0.35, "road": 0.49, "warm": 0.00}),
    ("Russia Siberia / N. Eurasia cold", 62.0, 100.0, 1.10,
     {"green": 0.05, "arid": 0.05, "sky": 0.45, "bright": 0.48, "snow": 0.20, "urban": 0.20, "road": 0.40, "warm": 0.00}),
    ("Scandinavia / N. Europe cool", 62.0, 15.0, 1.00,
     {"green": 0.30, "arid": 0.05, "sky": 0.60, "bright": 0.45, "snow": 0.05, "urban": 0.23, "road": 0.47, "warm": 0.00}),
    ("Western Europe temperate", 50.0, 5.0, 1.05,
     {"green": 0.15, "arid": 0.08, "sky": 0.45, "bright": 0.50, "snow": 0.03, "urban": 0.45, "road": 0.60, "warm": 0.02}),
    ("Mediterranean / S. Europe dry", 40.0, 15.0, 1.00,
     {"green": 0.10, "arid": 0.30, "sky": 0.60, "bright": 0.62, "snow": 0.00, "urban": 0.40, "road": 0.55, "warm": 0.08}),
    ("US-Canada temperate", 45.0, -100.0, 1.20,
     {"green": 0.16, "arid": 0.14, "sky": 0.67, "bright": 0.51, "snow": 0.02, "urban": 0.31, "road": 0.52, "warm": 0.00}),
    ("US Southwest / Mexico arid", 32.0, -110.0, 1.15,
     {"green": 0.05, "arid": 0.40, "sky": 0.60, "bright": 0.62, "snow": 0.00, "urban": 0.25, "road": 0.55, "warm": 0.10}),
    ("Mexico / Central America", 20.0, -100.0, 1.10,
     {"green": 0.07, "arid": 0.02, "sky": 0.68, "bright": 0.53, "snow": 0.22, "urban": 0.29, "road": 0.43, "warm": 0.00}),
    ("Brazil tropical", -10.0, -55.0, 1.20,
     {"green": 0.11, "arid": 0.23, "sky": 0.50, "bright": 0.45, "snow": 0.13, "urban": 0.33, "road": 0.35, "warm": 0.10}),
    ("Andes / W. South America", -15.0, -70.0, 1.00,
     {"green": 0.12, "arid": 0.27, "sky": 0.47, "bright": 0.59, "snow": 0.08, "urban": 0.52, "road": 0.38, "warm": 0.05}),
    ("Argentina pampas / S. Cone", -35.0, -65.0, 1.00,
     {"green": 0.03, "arid": 0.49, "sky": 0.53, "bright": 0.53, "snow": 0.00, "urban": 0.29, "road": 0.68, "warm": 0.03}),
    ("Middle East arid", 28.0, 45.0, 1.00,
     {"green": 0.03, "arid": 0.55, "sky": 0.55, "bright": 0.68, "snow": 0.00, "urban": 0.30, "road": 0.50, "warm": 0.15}),
    ("N. Africa / Sahara", 25.0, 10.0, 1.00,
     {"green": 0.02, "arid": 0.60, "sky": 0.55, "bright": 0.70, "snow": 0.00, "urban": 0.15, "road": 0.45, "warm": 0.15}),
    ("Sub-Saharan savanna", 5.0, 20.0, 1.00,
     {"green": 0.13, "arid": 0.25, "sky": 0.66, "bright": 0.52, "snow": 0.14, "urban": 0.10, "road": 0.05, "warm": 0.05}),
    ("Congo / C. Africa jungle", 0.0, 25.0, 1.00,
     {"green": 0.40, "arid": 0.03, "sky": 0.30, "bright": 0.45, "snow": 0.00, "urban": 0.10, "road": 0.30, "warm": 0.03}),
    ("Southern Africa", -28.0, 25.0, 1.00,
     {"green": 0.10, "arid": 0.35, "sky": 0.55, "bright": 0.60, "snow": 0.00, "urban": 0.20, "road": 0.50, "warm": 0.08}),
    ("South Asia", 25.0, 80.0, 1.05,
     {"green": 0.22, "arid": 0.03, "sky": 0.47, "bright": 0.42, "snow": 0.12, "urban": 0.58, "road": 0.56, "warm": 0.00}),
    ("Southeast Asia tropical", 12.0, 102.0, 1.05,
     {"green": 0.35, "arid": 0.03, "sky": 0.40, "bright": 0.50, "snow": 0.00, "urban": 0.25, "road": 0.45, "warm": 0.04}),
    ("East Asia temperate", 35.0, 115.0, 1.00,
     {"green": 0.15, "arid": 0.10, "sky": 0.40, "bright": 0.50, "snow": 0.03, "urban": 0.40, "road": 0.55, "warm": 0.03}),
    ("Japan", 36.0, 138.0, 1.00,
     {"green": 0.10, "arid": 0.16, "sky": 0.20, "bright": 0.41, "snow": 0.11, "urban": 0.44, "road": 0.56, "warm": 0.01}),
    ("Australia", -27.0, 133.0, 1.05,
     {"green": 0.05, "arid": 0.45, "sky": 0.60, "bright": 0.65, "snow": 0.00, "urban": 0.20, "road": 0.50, "warm": 0.10}),
    ("New Zealand", -42.0, 172.0, 1.00,
     {"green": 0.12, "arid": 0.04, "sky": 0.66, "bright": 0.36, "snow": 0.06, "urban": 0.22, "road": 0.39, "warm": 0.00}),
]

# Weight of each feature axis in the profile distance.
FEATURE_WEIGHTS = {
    "green": 1.0, "arid": 1.0, "sky": 0.6, "bright": 0.7,
    "snow": 1.2, "urban": 0.5, "road": 0.4, "warm": 0.8,
}

# Fallback guess when a photo cannot be decoded at all: the most common
# practice region (a temperate northern landmass). Never leave a photo blank.
FALLBACK_LAT, FALLBACK_LON = 55.0, 37.0


def score_profiles(features):
    """Score every region against the photo's evidence. Returns a list of
    (score, name, lat, lon) sorted best-first; ties break by profile order
    (deterministic). Higher score is better."""
    scored = []
    for idx, (name, lat, lon, prior, target) in enumerate(PROFILES):
        dist = 0.0
        for feat, w in FEATURE_WEIGHTS.items():
            dist += w * abs(features[feat] - target[feat])
        score = -dist + 0.25 * math.log(prior)
        scored.append((score, -idx, name, lat, lon))
    scored.sort(reverse=True)
    return [(s, name, lat, lon) for s, _ni, name, lat, lon in scored]


def locate_photo(jpeg_bytes):
    """Stage 1-3 for one photo: decode, measure, hypothesise.

    Returns (lat, lon, region_name, features) or the fallback guess when the
    photo cannot be decoded.
    """
    decoded = decode_dc_grid(jpeg_bytes)
    if decoded is None:
        return FALLBACK_LAT, FALLBACK_LON, "undecodable-fallback", None
    bw, bh, grid = decoded
    features = extract_features(bw, bh, grid)
    if features is None:
        return FALLBACK_LAT, FALLBACK_LON, "undecodable-fallback", None
    ranked = score_profiles(features)
    _score, name, lat, lon = ranked[0]
    return lat, lon, name, features


# ---------------------------------------------------------------------------
# Stage 4: ANSWER -- env client and main loop.
# ---------------------------------------------------------------------------

def _http_json(url, method="GET", body=None):
    req = urllib.request.Request(url, method=method)
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, data=data, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError("env %s %s -> HTTP %d" % (method, url, e.code))


def _http_bytes(url):
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def main():
    prompt_file = os.environ.get("MAKEFUN_PROMPT_FILE", "")
    env_url = os.environ.get("MAKEFUN_ENV_URL", "")
    if not env_url:
        sys.stderr.write("geolocator: MAKEFUN_ENV_URL is not set\n")
        return 1
    if prompt_file:
        try:
            with open(prompt_file, "r", encoding="utf-8") as f:
                brief = f.read()
            sys.stderr.write("geolocator: brief %d chars\n" % len(brief))
        except OSError as e:
            sys.stderr.write("geolocator: cannot read brief: %s\n" % e)
    env_url = env_url.rstrip("/")
    listing = _http_json(env_url + "/photos")
    photos = [p for p in listing.get("photos", []) if not p.get("answered")]
    photos.sort(key=lambda p: p["id"])
    answers = []
    for photo in photos:
        pid = photo["id"]
        try:
            jpeg = _http_bytes(env_url + photo["url"])
        except Exception as e:
            sys.stderr.write("geolocator: %s fetch failed: %s; fallback\n" % (pid, e))
            answers.append({"id": pid, "lat": FALLBACK_LAT, "lon": FALLBACK_LON})
            continue
        lat, lon, region, features = locate_photo(jpeg)
        lat = round(max(-90.0, min(90.0, lat)), 6)
        lon = round(max(-180.0, min(180.0, lon)), 6)
        if features is not None:
            sys.stderr.write(
                "geolocator: %s -> %s (%.4f, %.4f) "
                "green=%.2f arid=%.2f sky=%.2f bright=%.2f snow=%.2f urban=%.2f road=%.2f warm=%.2f\n"
                % (pid, region, lat, lon,
                   features["green"], features["arid"], features["sky"],
                   features["bright"], features["snow"], features["urban"],
                   features["road"], features["warm"]))
        else:
            sys.stderr.write("geolocator: %s -> %s (%.4f, %.4f) [fallback]\n"
                             % (pid, region, lat, lon))
        answers.append({"id": pid, "lat": lat, "lon": lon})
    if not answers:
        sys.stderr.write("geolocator: nothing to answer\n")
        return 0
    result = _http_json(env_url + "/answers", method="POST", body={"answers": answers})
    sys.stderr.write("geolocator: posted %d answers: %s\n" % (len(answers), json.dumps(result)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

