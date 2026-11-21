# Deploy AcciDEN on Vercel

This version has no dashboard login. Sidebar buttons switch Admin, Ambulance, and Police inside the current tab without opening tabs or reloading the camera. Saved incident history and videos are shared through private Vercel Blob storage. No Supabase account is used.

## GitHub upload

Create an empty GitHub repository and install Git LFS. From the extracted project folder, replace the URL below and run:

```powershell
git init -b main
git lfs install
git add .
git commit -m "Prepare same-tab dashboard app"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

The 109 MiB model requires Git LFS. Upload the extracted folder through Git, not the ZIP through GitHub's web uploader.

## Vercel setup

1. Import your repository. Framework: Other. Build: `npm run build`. Output: `dist`.
2. Create a PRIVATE Vercel Blob store in Storage and connect it to this project. Enable the read-write token environment variable.
3. Upload `model/epoch61.pt` through the Blob dashboard. Copy its private Blob URL.
4. Ensure `BLOB_READ_WRITE_TOKEN` is configured. Set `MODEL_URL` to the model's private Blob URL. No APP_PASSWORD is needed. Leave API_BASE_URL unset.
5. Enable Fluid Compute. If required for PyTorch and available in your account, set `VERCEL_SUPPORT_LARGE_FUNCTIONS=1`.
6. Deploy. Test actual model inference and playback. Large-function availability and free-plan CPU, memory, storage, and transfer limits must be confirmed in your account; successful live inference has not yet been verified.

## Use the dashboards

Open `/admin`. Click the left sidebar buttons to change dashboards in the SAME tab. Browser Back and Forward work. Camera state stays in the page while you switch views; keep the page open while evidence records and uploads. Hidden/background browser pages can throttle detection.

For other devices, use the SAME deployed domain: `/ambulance` and `/police`. Admin has Copy link buttons. No password is required. All devices share one Blob manifest; incidents, videos, and decisions refresh every 15 seconds and when the page regains focus.

Admin has Delete video buttons next to saved incident IDs and in Video storage management. Deletion removes the saved object and retains incident history. Ambulance and Police show the deletion after refresh. Dispatch remains simulated.

As requested, dashboard APIs are open: anyone who can reach the deployed app can read evidence, submit decisions, and delete videos. Blob credentials remain server-side.

## Local use

Extract the new ZIP, stop the earlier server, and run START_UPDATED_APP.bat. Use the browser tab it opens. Close old duplicate tabs once. The launcher opens only Admin on startup; it opens no extra tabs when that server is already running. Local incident data is separate from cloud incident data.

## Checks

```powershell
npm ci
npm run check
npm test
npm run build
```

Eleven frontend tests and five backend API tests pass. The API integration test uses separate clients and simulated persistent storage to verify shared evidence, playback, independent decisions, and deletion. It does not contact a live Blob store or run YOLO. GitHub Actions also contains backend checks.
