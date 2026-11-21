# AcciDEN

Accident detection with Admin, Ambulance, and Police dashboards.

- Sidebar buttons switch dashboards inside the current tab, preserving page and camera state.
- No password or Connect controls.
- Admin can delete saved accident videos while retaining incident history.
- Devices on the same deployed domain share incidents, videos, and independent agency responses, refreshing every 15 seconds.
- Frontend and Python API run on Vercel; persistent data uses private Vercel Blob storage.

See START_HERE.md and DEPLOYMENT.md for setup. Live model inference and free-plan resource compatibility still need verification on your Vercel account.

The three dashboard navigation buttons are visible only on Admin. Ambulance and Police hide the buttons and Operations heading. Browser Back or the direct /admin URL can return to Admin.
