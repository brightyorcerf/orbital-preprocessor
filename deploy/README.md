# Deploying the OSP Command Centre

The dashboard is the artifact most visitors will actually look at, so it needs to
be reachable without a login and without a key.

## What is in here

| File | Purpose |
|---|---|
| `Dockerfile` | Dashboard-only image, CPU, 1.46 GB. Not the mission container. |

The dependency manifest lives at `ground/requirements.txt`, next to the
entrypoint, and is shared by both deploy paths below. It used to be duplicated
here as `requirements-dashboard.txt`; the copies drifted and the deployed page
died on `ModuleNotFoundError: streamlit_folium`, so there is now one file.

The root `Dockerfile` and root `requirements.txt` target the OrbitLab payload:
they install torch, ultralytics, sentence-transformers and
`onnxruntime-gpu`, which is roughly 3.8 GB of wheels and a ~10 GB image. The
dashboard reaches none of that, because it serves the committed corpus in
`data/briefs/` rather than running the detector.

## Streamlit Community Cloud — the deployment target

Entrypoint: `ground/dashboard.py`.

Community Cloud resolves dependencies from the entrypoint's own directory
before falling back to the repo root, so it installs `ground/requirements.txt`
and never sees the root manifest. This is load-bearing rather than tidy: the
free tier caps an app at **1 GB**, and the root manifest is roughly twice that,
so pointing Community Cloud at it does not merely build slowly — the install is
killed partway and the app boots against a half-populated environment. The
symptom is a long "Your app is in the oven" stall followed by a
`ModuleNotFoundError` for whichever package lost the race.

Two settings decide whether the link works for someone reading it in an email:

- Set the app's visibility to **public**. A Streamlit app defaults to private,
  and a private app answers every request with a `303` to `/-/login`, which is
  indistinguishable from a dead link to anyone following it.
- Leave `GEMINI_API_KEY` **unset**. ORION then prompts each visitor for their
  own key in the sidebar. Setting a shared key exposes the owner's quota to
  anyone who finds the URL.

## Container — local and self-hosted

Community Cloud does not build this image; it installs the manifest directly.
The image exists for running the dashboard locally and on any host that takes a
Dockerfile.

```bash
docker build -f deploy/Dockerfile -t osp-dashboard .
docker run --rm -p 8501:8501 osp-dashboard
curl -s localhost:8501/_stcore/health    # -> ok
```

It reads `PORT` rather than hardcoding one, so the same build runs unmodified
anywhere that injects a port (Cloud Run, Fly, Render).

Hugging Face Spaces is no longer an option: as of July 2026 the Docker SDK and
the free CPU Basic tier both require a paid personal plan.
