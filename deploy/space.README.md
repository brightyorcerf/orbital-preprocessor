---
title: OSP Command Centre
emoji: 🛰️
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 8501
dockerfile_path: deploy/Dockerfile
pinned: false
license: mit
short_description: Downlink autonomy for an imaging spacecraft, decided by rule
---

# OSP Command Centre

The ground segment for the Orbital Scene Preprocessor. A spacecraft decides what
to downlink on its next contact under real orbital and link constraints, by
deterministic rule; a language model explains the result and is architecturally
unable to change it.

Everything on this page is computed, not typed: the briefs come from an INT8
ONNX detector run over held-out DOTA-v1.0 tiles, placed on an SGP4-propagated
Sentinel-2C ground track.

Source: https://github.com/brightyorcerf/orbital-preprocessor

## ORION analyst

No API key ships with this Space. The reasoning tab asks for your own Gemini key
and holds it for your session only, so browsing this page cannot spend anyone
else's quota.
