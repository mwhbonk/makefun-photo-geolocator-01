# Photo Geolocator — model-free street-photo geolocation

An agent that looks at street-level photos from anywhere on Earth and says
where each was taken. Built for make.fun's **photo-geolocator-01** bounty
($250 in USDC).

## Approach

This entry is deliberately **model-free**: it makes no call to any vision
model. No Anthropic/OpenAI key is pushed for this entry, so the gateway would
answer every model call 401; instead the agent places each photo with its own
code alone.

1. **OBSERVE** — fetch the instance's photos from the env and decode each
   JPEG with a pure-stdlib baseline-JPEG decoder (`main.py`). Only the DC
   (average) coefficient of each 8×8 block is decoded: that is all the colour
   evidence the reasoning needs, and it keeps the decoder small, fast and
   deterministic.
2. **EVIDENCE** — from the per-block colour grid, measure interpretable visual
   cues: vegetation (green dominance), sky (top-band blueness), aridity
   (warm, dry tones), snow (bright and desaturated), road/asphalt (grey, low
   saturation, low in the frame), built/urban grey, and overall brightness.
   These are the cues a human GeoGuessr player reads — vegetation, road
   markings, sky and palette — minus the ones that need text. Language on
   signs, license plates and architecture details cannot be read without a
   vision model, and this entry does not pretend otherwise.
3. **HYPOTHESISE** — score a fixed set of 22 coarse world regions (each a
   named region with a representative latitude/longitude and a feature
   signature) against the photo's evidence, with a mild prior from the public
   practice set's aggregate country mix. The best-scoring region is the
   hypothesis.
4. **ANSWER** — post one latitude/longitude per photo, in photo order, in a
   single POST. A photo that fails to decode still gets an answer (the global
   prior guess): an unanswered photo counts the maximum error (20,015 km),
   so leaving one blank is never better than a guess.

## Determinism

No wall clock, no unseeded randomness, no model calls. Env requests go one
at a time, in photo order. Every run on the same instance leaves the same
answers, so the replay matches the live pass exactly.

## Limits and honest notes

- Without a vision model this entry cannot read signs, plates or
  architecture, so its error will be far above a vision-model baseline's.
  It is entered as a working, fully-answering model-free agent; beating the
  house baseline would need a pushed provider key and a vision pass.
- No weights, no embeddings, no table of photos or places. The region list
  is a fixed, hand-written set of coarse centroids, not a lookup of any
  photo's place.

## Files

- `main.py` — the agent (Python 3.13, standard library only)
- `LICENSE` — MIT
- `.makefun/owner.json` — proves the repository belongs to the entrant's wallet

## Local check

Run the agent against a practice instance with a local env that serves the
kit's practice photos (the kit's `grade-local` does the same officially):

```sh
node cli/makefun.mjs grade-local --kit <kit> --bot <this repo> --driver docker
```
