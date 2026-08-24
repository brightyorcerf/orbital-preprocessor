# Deploying the OSP Command Centre

The dashboard is the artifact most visitors will actually look at, so it needs to
be reachable without a login and without a key.

## What is in here

| File | Purpose |
|---|---|
| `Dockerfile` | Dashboard-only image, CPU, 1.46 GB. Not the mission container. |
| `requirements-dashboard.txt` | The subset the page actually imports. |

The root `Dockerfile` and root `requirements.txt` target the OrbitLab payload:
they install torch, ultralytics, faiss, sentence-transformers and
`onnxruntime-gpu`, which is roughly 3.8 GB of wheels and a ~10 GB image. The
dashboard reaches none of that, because it serves the committed corpus in
`data/briefs/` rather than running the detector.

## Verify locally first

```bash
docker build -f deploy/Dockerfile -t osp-dashboard .
docker run --rm -p 8501:8501 osp-dashboard
curl -s localhost:8501/_stcore/health    # -> ok
```

## Hugging Face Space (Docker SDK)

Deterministic, because the Space builds this exact Dockerfile.

1. Create a Space, SDK **Docker**, hardware **CPU basic (free)**.
2. Point it at this repository, or push the repo to the Space remote.
3. Add `dockerfile_path: deploy/Dockerfile` to the Space's `README.md`
   front matter.
4. Spaces route to port **7860**; the image reads `PORT`, so set `PORT=7860`
   in the Space variables. No code change needed.

## Streamlit Community Cloud

Fewer moving parts, but it installs from a `requirements.txt` it discovers
itself rather than from `deploy/Dockerfile`. Confirm which file it picks up
before relying on it: the root manifest pulls ~2 GB and will not fit the free
tier's resource cap.

Whichever platform is used, after the first deploy:

- Set the app's visibility to **public**. A Streamlit app defaults to private,
  and a private app answers every request with a `303` to `/-/login`, which is
  indistinguishable from a dead link to anyone following it from an email.
- Leave `GEMINI_API_KEY` **unset**. ORION then prompts each visitor for their
  own key in the sidebar. Setting a shared key exposes the owner's quota to
  anyone who finds the URL.
