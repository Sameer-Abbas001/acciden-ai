# Start here

1. Extract this ZIP into a new folder.
2. For local use, stop your earlier server and run START_UPDATED_APP.bat. Close old duplicate tabs once.
3. Click Admin, Ambulance, or Police in the left sidebar. They now switch in the SAME tab. No password or Connect button is required.
4. For GitHub upload and Vercel deployment, follow DEPLOYMENT.md.

Vercel still needs a private Blob store, BLOB_READ_WRITE_TOKEN, and MODEL_URL. No APP_PASSWORD is required. Use the same deployed domain on other devices to share incidents and videos automatically.

Admin retains Delete video controls. Build, eleven frontend checks, and five API tests passed; live Vercel/YOLO inference remains unverified.

The three dashboard navigation buttons are visible only on Admin. Ambulance and Police hide the buttons and Operations heading. Browser Back or the direct /admin URL can return to Admin.
