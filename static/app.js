'use strict';
const API_BASE = (window.ACCIDEN_CONFIG?.apiBaseUrl || '').replace(/\/$/, '');
const backendUrl = path => API_BASE ? new URL(path, `${API_BASE}/`).href : path;
const $ = id => document.getElementById(id);
const escapeHTML = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const state = { view: 'vision', incidents: [], selected: new URLSearchParams(location.search).get('incident'), active: false, recording: false, detecting: false, boxes: [], stream: null, url: null, generation: 0, timer: null, refreshing: false };
function currentView() {
  const route = location.pathname.replace(/\/$/, '').replace(/\.html$/, '');
  if (route === '/police') return 'police';
  if (route === '/ambulance') return 'ambulance';
  if (route === '/admin' || route === '/vision') return 'vision';
  return new URLSearchParams(location.search).get('view') || 'vision';
}
const source = $('source'), canvas = $('monitor'), ctx = canvas.getContext('2d');
const capture = document.createElement('canvas'), captureCtx = capture.getContext('2d');
let toastTimer, recordedAbort = null, currentEvidence = null;
function notify(message, error = false) { $('toast').textContent = message; $('toast').classList.toggle('error', error); $('toast').hidden = false; clearTimeout(toastTimer); toastTimer = setTimeout(() => $('toast').hidden = true, error ? 10000 : 5000); }
async function api(url, options = {}) {
  const response = await fetch(backendUrl(url), {...options, cache: 'no-store', headers:options.headers});
  const body = await response.text();
  let data;
  try { data = JSON.parse(body); }
  catch { throw new Error(`The backend returned an invalid response (HTTP ${response.status}). Check the Vercel runtime logs.`); }
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : `Request failed (${response.status})`);
  return data;
}
function form(values) { const data = new FormData(); Object.entries(values).forEach(([key, value]) => data.append(key, value)); return data; }
function responseStatus(incident, agency) { return incident.responses?.[agency]?.status || 'PENDING'; }
function pill(status, label = status.replaceAll('_', ' ').toLowerCase()) { return `<span class="pill ${escapeHTML(status.toLowerCase())}">${escapeHTML(label)}</span>`; }
function applyTheme(theme) { document.documentElement.dataset.theme = theme; $('theme').textContent = theme === 'dark' ? '☀' : '☾'; $('theme').setAttribute('aria-label', `Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`); try { localStorage.setItem('acciden-theme', theme); } catch {} }
let savedTheme; try { savedTheme = localStorage.getItem('acciden-theme'); } catch {}
applyTheme(savedTheme || 'dark');
$('theme').onclick = () => applyTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark');
function setView(view, updateHistory = true) {
  state.view = ['vision', 'ambulance', 'police'].includes(view) ? view : 'vision';
  const vision = state.view === 'vision', police = state.view === 'police';
  $('dashboardNav').hidden = !vision;
  $('dashboardNavLabel').hidden = !vision;
  document.querySelectorAll('[data-view]').forEach(link => { const active = link.dataset.view === state.view; link.classList.toggle('active', active); if (active) link.setAttribute('aria-current', 'page'); else link.removeAttribute('aria-current'); });
  $('visionPanel').hidden = !vision; $('responsePanel').hidden = vision;
  $('breadcrumb').textContent = vision ? 'Admin dashboard' : police ? 'Police dashboard' : 'Ambulance dashboard';
  document.title = `AcciDEN — ${$('breadcrumb').textContent}`;
  $('pageTitle').innerHTML = vision ? 'See it. Detect it. <em>Respond.</em>' : police ? 'Secure the scene. <em>Protect lives.</em>' : 'Ready to respond. <em>Ready to care.</em>';
  $('pageSubtitle').textContent = vision ? 'Real-time accident intelligence, connected to the teams that matter.' : `Review shared accident evidence and coordinate your ${police ? 'police' : 'medical'} response.`;
  $('agencyTitle').textContent = police ? 'Police response' : 'Ambulance response';
  $('queueTitle').textContent = vision ? 'Incident activity' : 'Response queue';
  if (updateHistory) history.pushState({}, '', state.view === 'vision' ? '/admin' : '/' + state.view);
  if (!vision && !state.selected && state.incidents.length) state.selected = state.incidents[0].id;
  render(); renderReview(); renderStorage();
}
document.querySelectorAll('[data-view],.agency-tiles a').forEach(link => {
  link.removeAttribute('target'); link.removeAttribute('rel');
  link.title = 'Switch dashboard';
  link.onclick = event => { event.preventDefault(); AccidenNavigation.navigate(link.href); };
});
window.addEventListener('acciden:navigate', event => {
  if (event.detail.incident) { state.selected = event.detail.incident; $('notes').value = ''; }
  setView(event.detail.view, false);
  refresh();
});
window.addEventListener('focus', () => refresh());
document.addEventListener('visibilitychange', () => { if (!document.hidden) refresh(); });
window.onpopstate = () => { state.selected = new URLSearchParams(location.search).get('incident'); setView(currentView(), false); refresh(); };
function render() {
  const items = state.incidents, agencies = state.view === 'vision' ? ['ambulance', 'police'] : [state.view];
  $('total').textContent = items.length; $('navCount').textContent = items.length;
  $('pending').textContent = items.filter(i => agencies.some(a => responseStatus(i, a) === 'PENDING')).length;
  $('clips').textContent = items.filter(i => i.has_clip).length;
  $('dispatched').textContent = items.reduce((n, i) => n + agencies.filter(a => responseStatus(i, a) === 'DISPATCHED').length, 0);
  $('incidentCount').textContent = items.length;
  const query = $('search').value.toLowerCase(), filter = $('filter').value;
  const filtered = items.filter(i => `${i.id} ${i.location || ''}`.toLowerCase().includes(query)).filter(i => filter === 'all' || (filter === 'ready' && i.has_clip) || (filter === 'pending' && agencies.some(a => responseStatus(i, a) === 'PENDING')) || (filter === 'dispatched' && agencies.some(a => responseStatus(i, a) === 'DISPATCHED')));
  $('incidentRows').innerHTML = filtered.map(i => `<tr class="${i.id === state.selected ? 'selected' : ''}" data-id="${escapeHTML(i.id)}"><td><b>${escapeHTML(i.id)}</b><small>${escapeHTML(i.timestamp)}</small>${state.view === 'vision' && (i.has_clip || i.object_path) ? `<button class="delete-video-button" data-delete-video="${escapeHTML(i.id)}">Delete video</button>` : ''}</td><td>${escapeHTML(i.location || 'Location not provided')}</td><td>${typeof i.confidence === 'number' ? `${i.confidence.toFixed(1)}%` : '—'}</td><td>${pill(i.has_clip ? 'READY' : 'PENDING', i.has_clip ? '8s · ready' : i.status === 'RECORDING_FAILED' ? 'Recording failed' : i.status === 'RECORDING_EVIDENCE' ? 'Recording…' : 'Awaiting video')}</td><td>${pill(responseStatus(i, 'ambulance'))}</td><td>${pill(responseStatus(i, 'police'))}</td><td><button class="review-button" data-id="${escapeHTML(i.id)}">Review</button></td></tr>`).join('');
  $('tableEmpty').hidden = filtered.length > 0;
  $('tableEmpty').querySelector('h3').textContent = items.length ? 'No incidents match this view.' : "All clear. We're watching with you.";
  $('tableEmpty').querySelector('p').textContent = items.length ? 'Try another search or filter.' : 'Detected incidents will appear here, ready for both response teams.';
  $('tableSummary').textContent = `${filtered.length} of ${items.length} incidents`;
}
$('incidentRows').onclick = async event => { const deleteButton = event.target.closest('[data-delete-video]'); if (deleteButton) { deleteVideo(deleteButton); return; } const row = event.target.closest('[data-id]'); if (!row) return; if (state.view === 'vision') { AccidenNavigation.navigate(`/ambulance?incident=${encodeURIComponent(row.dataset.id)}`); return; } state.selected = row.dataset.id; $('notes').value = ''; render(); renderReview(); $('responsePanel').scrollIntoView({behavior:'smooth', block:'start'}); };
$('search').oninput = render; $('filter').onchange = render;
function renderReview() {
  if (state.view === 'vision') return;
  const incident = state.incidents.find(i => i.id === state.selected);
  if (!incident) return;
  $('reviewId').textContent = incident.id;
  $('incidentDetails').innerHTML = `<h3 class="detail-location">${escapeHTML(incident.location || 'Location not provided')}</h3><p class="detail-time">${escapeHTML(incident.timestamp)}</p><div class="detail-row"><span>Detection confidence</span><b>${typeof incident.confidence === 'number' ? incident.confidence.toFixed(1) + '%' : 'Unavailable'}</b></div><div class="detail-row"><span>Ambulance</span>${pill(responseStatus(incident,'ambulance'))}</div><div class="detail-row"><span>Police</span>${pill(responseStatus(incident,'police'))}</div>${incident.responses?.[state.view]?.notes ? `<p class="setting-note">${escapeHTML(incident.responses[state.view].notes)}</p>` : ''}`;
  const url = incident.has_clip && incident.clip_url ? backendUrl(incident.clip_url) : null;
  if (currentEvidence !== url) { $('evidence').pause(); if (url) $('evidence').src = url; else { $('evidence').removeAttribute('src'); $('evidence').load(); } currentEvidence = url; $('evidence').playbackRate = 1; $('slowButton').textContent = '0.5× slow motion'; }
  $('evidence').hidden = !url; $('evidenceEmpty').hidden = !!url;
  $('evidenceEmpty').querySelector('h3').textContent = incident.video_deleted ? 'Video deleted by Admin' : incident.status === 'RECORDING_FAILED' ? 'Evidence recording failed' : 'Evidence is on its way';
  $('evidenceEmpty').querySelector('p').textContent = incident.video_deleted ? 'The saved video was removed to free storage. Incident history is retained.' : incident.status === 'RECORDING_FAILED' ? 'The source ended or recording failed. Review is unavailable without a playable clip.' : 'The detection workspace is recording and uploading the 8-second clip. This view refreshes automatically.';
  $('slowButton').disabled = !url; $('frameButton').disabled = !url; $('download').hidden = !url; if (url) $('download').href = url;
  const status = responseStatus(incident, state.view), pending = status === 'PENDING';
  $('reviewActions').hidden = !url || (status !== 'PENDING' && status !== 'DISPATCHED');
  $('confirmButton').hidden = !pending; $('rejectButton').hidden = !pending; $('resolveButton').hidden = status !== 'DISPATCHED';
  $('notes').hidden = !pending; document.querySelector('label[for=notes]').hidden = !pending;
}
$('slowButton').onclick = () => { const slow = $('evidence').playbackRate === 1; $('evidence').playbackRate = slow ? .5 : 1; $('slowButton').textContent = slow ? '1× normal speed' : '0.5× slow motion'; };
$('frameButton').onclick = () => { $('evidence').pause(); $('evidence').currentTime = Math.min($('evidence').duration || 8, $('evidence').currentTime + 1/30); };
$('evidence').onerror = () => notify('This evidence video could not be played. Check the saved file and server connection.', true);
async function review(action) { const buttons = [$('confirmButton'), $('rejectButton'), $('resolveButton')]; buttons.forEach(b => b.disabled = true); try { await api(`/api/response/${state.view}/${action}`, {method:'POST', body:form({incident_id:state.selected, notes:$('notes').value})}); await refresh(); notify(action === 'confirm' ? `${state.view === 'police' ? 'Police' : 'Ambulance'} response dispatched in local simulation.` : action === 'resolve' ? 'Response marked resolved.' : 'Incident rejected for this agency only.'); } catch (error) { notify(error.message, true); } finally { buttons.forEach(b => b.disabled = false); } }
$('confirmButton').onclick = () => review('confirm'); $('rejectButton').onclick = () => review('reject'); $('resolveButton').onclick = () => review('resolve');
async function refresh(manual = false) { if (state.refreshing) return; state.refreshing = true; try { const data = await api(`/api/response/${state.view === 'police' ? 'police' : 'ambulance'}/incidents`); state.incidents = data.incidents; $('connection').textContent = '● Server connected'; $('connection').className = 'connection online'; if (!state.selected && state.view !== 'vision' && state.incidents.length) state.selected = state.incidents[0].id; render(); renderReview(); renderStorage(); if (manual) notify('Incident activity is up to date.'); } catch (error) { $('connection').textContent = 'Server disconnected'; $('connection').className = 'connection'; if (manual) notify(error.message, true); } finally { state.refreshing = false; } }
$('refresh').onclick = () => refresh(true);
$('confidence').oninput = () => $('confidenceValue').textContent = `${$('confidence').value}%`;
function paint() {
  if (state.active && source.readyState >= 2) {
    ctx.drawImage(source, 0, 0, canvas.width, canvas.height);
    for (const box of state.boxes) { const [x1,y1,x2,y2] = box.bbox; ctx.strokeStyle = box.is_accident ? (box.confirmed === false ? '#ffc86e' : '#ff695e') : '#72e4bd'; ctx.lineWidth = 3; ctx.strokeRect(x1,y1,x2-x1,y2-y1); const label = `${box.is_accident ? (box.confirmed === false ? 'ACCIDENT CANDIDATE' : 'ACCIDENT') : 'VEHICLE'} ${box.confidence_pct}%`; ctx.font = 'bold 15px sans-serif'; const width = ctx.measureText(label).width + 14; ctx.fillStyle = ctx.strokeStyle; ctx.fillRect(x1, Math.max(0,y1-26), width, 26); ctx.fillStyle = '#071212'; ctx.fillText(label,x1+7,Math.max(18,y1-8)); }
    ctx.fillStyle = '#071212c9'; ctx.fillRect(8, canvas.height-30, 260, 24); ctx.fillStyle = '#fff'; ctx.font = '12px monospace'; ctx.fillText(new Date().toLocaleString(),16,canvas.height-13);
  }
  requestAnimationFrame(paint);
}
paint();
function releaseSource() { state.generation++; state.active = false; state.boxes = []; clearTimeout(state.timer); source.pause(); source.srcObject = null; source.removeAttribute('src'); source.load(); state.stream?.getTracks().forEach(t => t.stop()); state.stream = null; if (state.url) URL.revokeObjectURL(state.url); state.url = null; }
function sourceControls(disabled) { $('cameraButton').disabled = disabled; $('upload').disabled = disabled; $('stopButton').disabled = disabled || !state.active; }
function stopMonitoring() { if (state.recording) { notify('Please wait for the 8-second evidence capture and upload to finish.'); return; } releaseSource(); $('monitorStatus').textContent = 'MONITOR STOPPED'; $('sourceTag').textContent = 'STOPPED'; $('feedState').textContent = 'Monitoring stopped'; $('stopButton').disabled = true; }
$('stopButton').onclick = stopMonitoring;
async function startSource(file = null) {
  if (state.recording) return;
  if (!window.MediaRecorder || !canvas.captureStream) { notify('Use a current Chrome or Edge browser to record evidence.',true); return; }
  releaseSource(); const generation = state.generation; sourceControls(true);
  try {
    if (file) { state.url = URL.createObjectURL(file); source.src = state.url; } else { state.stream = await navigator.mediaDevices.getUserMedia({video:{width:{ideal:1280},height:{ideal:720}},audio:false}); if (generation !== state.generation) { state.stream.getTracks().forEach(t => t.stop()); return; } source.srcObject = state.stream; state.stream.getVideoTracks()[0].onended = () => { recordedAbort?.('Camera disconnected before recording completed.'); stopMonitoring(); }; }
    await source.play(); if (generation !== state.generation) return;
    canvas.width = Math.min(1280, source.videoWidth); canvas.height = Math.round(canvas.width * source.videoHeight/source.videoWidth); capture.width = canvas.width; capture.height = canvas.height;
    state.active = true; $('monitor').hidden = false; $('feedEmpty').hidden = true; $('sourceTag').textContent = file ? 'VIDEO FILE' : 'LIVE CAMERA'; $('sourceName').textContent = file?.name || 'Live camera feed'; $('monitorStatus').textContent = '● DETECTION ACTIVE'; $('feedState').textContent = 'Analyzing video'; sourceControls(false); detect();
  } catch (error) { releaseSource(); sourceControls(false); $('feedState').textContent = 'Source unavailable'; notify(`Could not start video: ${error.message}`, true); }
}
$('cameraButton').onclick = () => startSource();
$('upload').onchange = event => { const file = event.target.files[0]; if (file) startSource(file); event.target.value = ''; };
source.onended = () => { if (state.recording) recordedAbort?.('The source video ended before 8 seconds could be captured. Use footage with at least 8 seconds after detection.'); state.active = false; $('monitorStatus').textContent = 'VIDEO ENDED'; $('feedState').textContent = 'Video complete — choose another source'; $('stopButton').disabled = true; };
async function detect() {
  if (!state.active || state.recording || state.detecting) return;
  const generation = state.generation; state.detecting = true;
  try {
    const started = performance.now();
    const frames = [];
    for (let sample = 0; sample < 3; sample++) {
      if (generation !== state.generation || !state.active || source.readyState < 2) return;
      captureCtx.drawImage(source,0,0,capture.width,capture.height);
      frames.push(capture.toDataURL('image/jpeg',.7));
      if (sample < 2) await new Promise(resolve => setTimeout(resolve,160));
    }
    const data = await api('/api/detect', {method:'POST', body:form({frames_json:JSON.stringify(frames), conf:Number($('confidence').value)/100, return_annotated:false, location:$('location').value.trim() || 'Location not provided'})});
    if (generation !== state.generation || !state.active) { if (data.incident) await api(`/api/incidents/${data.incident.id}/recording-failed`, {method:'POST'}); return; }
    state.boxes = data.detections; $('inference').textContent = `${Math.round(performance.now()-started)} ms inference`; $('feedState').textContent = data.summary.accident_detected ? 'Possible accident — repeated across frames' : data.summary.candidate_detected ? 'Unconfirmed accident candidate' : 'Monitoring traffic';
    if (data.incident) { await refresh(); await recordIncident(data.incident); }
  } catch (error) { $('feedState').textContent = 'Detection unavailable'; notify(`Detection: ${error.message}`, true); }
  finally { state.detecting = false; if (state.active) state.timer = setTimeout(detect, 700); }
}
async function recordIncident(incident) {
  state.recording = true; sourceControls(true); $('recording').hidden = false; $('recordProgress').style.width = '0%'; $('recordingText').textContent = 'Recording · 8 seconds remaining'; notify('Possible accident seen across frames. Capturing evidence for review.');
  let recordStream, ticker;
  try {
    const mime = ['video/webm;codecs=vp9','video/webm;codecs=vp8','video/mp4','video/webm'].find(type => MediaRecorder.isTypeSupported(type));
    if (!mime) throw new Error('No supported video recording format in this browser.');
    recordStream = canvas.captureStream(30);
    const recorder = new MediaRecorder(recordStream,{mimeType:mime,videoBitsPerSecond:1000000});
    const chunks = [];
    const blob = await new Promise((resolve,reject) => {
      let aborted = null, timeout; let started = performance.now();
      const stop = () => { clearTimeout(timeout); clearInterval(ticker); if (recorder.state !== 'inactive') recorder.stop(); };
      recordedAbort = message => { aborted = new Error(message); stop(); };
      recorder.ondataavailable = event => { if (event.data.size) chunks.push(event.data); };
      recorder.onerror = () => { aborted = new Error('Browser recording failed.'); stop(); reject(aborted); };
      recorder.onstop = () => aborted ? reject(aborted) : resolve(new Blob(chunks,{type:mime}));
      recorder.onstart = () => { started = performance.now(); timeout = setTimeout(stop,8250); };
      recorder.start(250);
      ticker = setInterval(() => { const elapsed = performance.now()-started; $('recordProgress').style.width = `${Math.min(100,elapsed/80)}%`; $('recordingText').textContent = `Recording · ${Math.max(0,Math.ceil((8000-elapsed)/1000))} seconds remaining`; },100);
    });
    recordedAbort = null; recordStream.getTracks().forEach(t => t.stop()); $('recordProgress').style.width = '100%'; $('recordingText').textContent = 'Uploading to both dashboards…';
    const body = form({incident_id:incident.id,confidence:incident.confidence,location:incident.location,vehicles_count:incident.vehicles_involved}); body.append('video_file',blob,`evidence.${mime.includes('mp4') ? 'mp4' : 'webm'}`);
    await api('/api/clips/upload',{method:'POST',body}); notify('8-second evidence delivered to Ambulance and Police.');
  } catch (error) { notify(`Evidence failed: ${error.message}`,true); try { await api(`/api/incidents/${incident.id}/recording-failed`,{method:'POST'}); } catch {} }
  finally { clearInterval(ticker); recordedAbort = null; recordStream?.getTracks().forEach(t => t.stop()); state.recording = false; $('recording').hidden = true; sourceControls(false); await refresh(); }
}
window.addEventListener('beforeunload', event => { if (state.recording) { event.preventDefault(); event.returnValue = ''; } });
function updateClock() { const now = new Date(); $('today').textContent = now.toLocaleDateString(undefined,{day:'numeric',month:'short',year:'numeric'}); $('clock').textContent = now.toLocaleTimeString(undefined,{hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false}); }
setView(currentView(),false); refresh(); updateClock(); setInterval(refresh,15000); setInterval(updateClock,1000);

function renderStorage() {
  $('storagePanel').hidden = state.view !== 'vision';
  $('sharingPanel').hidden = state.view !== 'vision';
  $('manageVideos').hidden = state.view !== 'vision';
  const videos = state.incidents.filter(i => i.object_path || i.has_clip);
  $('storageSummary').textContent = videos.length + ' saved videos · ' + (videos.reduce((total, i) => total + (i.clip_bytes || 0), 0) / 1048576).toFixed(2) + ' MB';
  $('storageRows').innerHTML = videos.length ? videos.map(i => '<tr><td>' + escapeHTML(i.id) + '</td><td>' + escapeHTML(i.location) + '</td><td>' + ((i.clip_bytes || 0)/1048576).toFixed(2) + ' MB</td><td><button data-delete-video="' + escapeHTML(i.id) + '">Delete video</button></td></tr>').join('') : '<tr><td colspan="4">No saved videos. Delete buttons appear after an accident recording is saved.</td></tr>';
}
async function deleteVideo(button) {
  if (state.view !== 'vision' || !button || !confirm('Permanently delete this accident video? The incident history will remain.')) return;
  button.disabled = true;
  try { await api('/api/incidents/' + encodeURIComponent(button.dataset.deleteVideo) + '/video', {method:'DELETE'}); await refresh(); notify('Video deleted from storage.'); }
  catch (error) { notify(error.message, true); button.disabled = false; }
}
$('storageRows').onclick = event => deleteVideo(event.target.closest('[data-delete-video]'));
$('manageVideos').onclick = () => { $('storagePanel').scrollIntoView({behavior:'smooth', block:'start'}); };

document.querySelectorAll('[data-copy-dashboard]').forEach(button => {
  button.onclick = async () => {
    const url = new URL('/' + button.dataset.copyDashboard, location.origin).href;
    try { await navigator.clipboard.writeText(url); notify('Dashboard link copied. Share it with the other device.'); }
    catch { notify('Copy the dashboard URL shown beside this button.'); }
  };
});
for (const role of ['ambulance', 'police']) $('share-' + role).value = new URL('/' + role, location.origin).href;
