# Deploying the OSP Command Centre

The dashboard is the artifact most visitors will actually look at, so it needs to
be reachable without a login and without a key.

## What is in here

| File | Purpose |
|---|---|
| `Dockerfile` | Dashboard-only image, CPU, 1.46 GB. Not the mission container. |
| `requirements-dashboard.txt` | The subset the page actually imports. |

The root `Dockerfile` and root `requirements.txt` target the OrbitLab payload:
they install torch, ultralytics, sentence-transformers and
`onnxruntime-gpu`, which is roughly 3.8 GB of wheels and a ~10 GB image. The
dashboard reaches none of that, because it serves the committed corpus in
`data/briefs/` rather than running the detector.

## Verify locally first

```bash
docker build -f deploy/Dockerfile -t osp-dashboard .
docker run --rm -p 8501:8501 osp-dashboard
curl -s localhost:8501/_stcore/health    # -> ok
```

## Hugging Face Space (Docker SDK) — the deployment target

Deterministic, because the Space builds this exact Dockerfile rather than
resolving dependencies on its own.

1. Create a Space: SDK **Docker**, hardware **CPU basic (free)**.
2. Copy `deploy/space.README.md` to the Space repository's `README.md`. Its
   front matter carries `sdk: docker`, `dockerfile_path: deploy/Dockerfile` and
   `app_port: 8501`, which is the port this image already listens on, so there
   is nothing further to configure.
3. Push this repository to the Space remote:

   ```bash
   git remote add space https://huggingface.co/spaces/<user>/osp-command-centre
   git push space main
   ```

4. Leave the Space's variables and secrets empty. See the note on keys below.

The image reads `PORT`, so the same build also runs anywhere that injects one
(Cloud Run, Fly) without an edit.

After the first deploy:

- Set the app's visibility to **public**. A Streamlit app defaults to private,
  and a private app answers every request with a `303` to `/-/login`, which is
  indistinguishable from a dead link to anyone following it from an email.
- Leave `GEMINI_API_KEY` **unset**. ORION then prompts each visitor for their
  own key in the sidebar. Setting a shared key exposes the owner's quota to
  anyone who finds the URL.
