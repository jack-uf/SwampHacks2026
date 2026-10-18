import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import './wireframe-live.css';

type Data = Record<string, any>;
const $ = <T extends Element = HTMLElement>(selector: string, root: ParentNode = document) => root.querySelector<T>(selector)!;
const all = <T extends Element = HTMLElement>(selector: string, root: ParentNode = document) => Array.from(root.querySelectorAll<T>(selector));
const boxes = (id: string) => all<HTMLElement>(`#${id} article.box`);
const esc = (value: unknown) => String(value ?? 'Unavailable').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]!));
const label = (value: unknown) => esc(String(value ?? 'unknown').replaceAll('_', ' '));
const date = (value: any) => value ? new Date(value).toLocaleString() : 'Unavailable';
const option = (value: string, text: string) => `<option value="${esc(value)}">${esc(text)}</option>`;
const hazards = ['heat','storm','flood','lightning','wildfire','wildlife','unstable_terrain','equipment_obstruction','other'];
let sites: Data[] = [], missions: Data[] = [], selected = '', mission: Data | null = null, detail: Data | null = null;
let siteRequest = 0, missionRequest = 0, liveLightningConfigured = false;
const status = document.createElement('p');
status.className = 'live-status'; status.setAttribute('role', 'status'); status.setAttribute('aria-live', 'polite');
$('#wireframe main').prepend(status);
function notify(message: string, error = false) { status.textContent = message; status.setAttribute('role', error ? 'alert' : 'status'); }
async function api(path: string, body?: unknown, method = 'GET') {
  const headers: Record<string,string> = {};
  const token = sessionStorage.getItem('planit-token');
  if (token) headers.Authorization = `Bearer ${token}`;
  if (body && !(body instanceof FormData)) headers['Content-Type'] = 'application/json';
  const response = await fetch(`/api${path}`, {method, headers, body: body instanceof FormData ? body : body ? JSON.stringify(body) : undefined});
  if (!response.ok) {
    const error = await response.json().catch(() => ({detail: response.statusText}));
    throw new Error(`${response.status}: ${typeof error.detail === 'string' ? error.detail : JSON.stringify(error.detail)}`);
  }
  return response.json();
}
async function run(button: HTMLButtonElement | null, task: () => Promise<void>) {
  if (button) button.disabled = true;
  try { await task(); } catch (error) { notify(String(error), true); }
  finally { if (button) button.disabled = false; }
}
function content(box: HTMLElement, html: string) { const title = $('h3', box).outerHTML; box.innerHTML = title + html; }
function form(box: HTMLElement, buttonText: string, submit: (form: HTMLFormElement) => Promise<void>) {
  const node = document.createElement('form');
  while (box.children.length > 1) node.append(box.children[1]);
  node.insertAdjacentHTML('beforeend', `<button type="submit">${buttonText}</button>`);
  box.append(node);
  node.addEventListener('submit', event => { event.preventDefault(); void run($<HTMLButtonElement>('button[type=submit]', node), () => submit(node)); });
  return node;
}
function field(name: string, text: string, attributes = '') { return `<label>${text}<input name="${name}" ${attributes} /></label>`; }
function safeLink(url: any, text: string) {
  try { const parsed = new URL(url); if (!['http:','https:'].includes(parsed.protocol)) return esc(text); return `<a href="${esc(parsed.href)}" target="_blank" rel="noreferrer">${esc(text)}</a>`; } catch { return esc(text); }
}
all<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>('#wireframe [disabled]').forEach(node => node.disabled = false);
$('.masthead .tag').textContent = 'Connected workspace';
$('.masthead a').textContent = 'API documentation'; $('.masthead a').setAttribute('href', '/docs');
$('#overview .section-heading .caption').textContent = 'Select a facility to inspect its conditions and reports.';
$('#wireframe footer').textContent = 'Decision support for supervisor review. Missing feeds remain unavailable; AI suggestions never set ratings.';
const overviewBoxes = boxes('overview');
const mapNode = $('#overview .map'); mapNode.innerHTML = '';
const map = L.map(mapNode).setView([37.4,-96.5], 4);
L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {attribution:'&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap contributors</a>'}).addTo(map);
const markers = L.layerGroup().addTo(map);
const overlapNode = $('#overlaps .map'); overlapNode.innerHTML = '';
const overlapMap = L.map(overlapNode).setView([37.4,-96.5], 4);
L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {attribution:'&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap contributors</a>'}).addTo(overlapMap);
const overlapMarkers = L.layerGroup().addTo(overlapMap);
new ResizeObserver(() => { map.invalidateSize(); overlapMap.invalidateSize(); }).observe($('#wireframe main'));
const watchlist = overviewBoxes[5];
all('.row', watchlist).forEach(row => row.remove());
const siteList = document.createElement('div'); siteList.className = 'live-list'; $('label', watchlist).after(siteList);
const search = $('input', watchlist) as HTMLInputElement;
search.addEventListener('input', renderSites);
function renderSites() {
  siteList.innerHTML = sites.filter(site => `${site.name} ${site.state} ${site.operator}`.toLowerCase().includes(search.value.toLowerCase())).map(site => `<button type="button" class="row" data-site="${esc(site.id)}" aria-pressed="${site.id === selected}"><span><strong>${esc(site.name)}</strong><br><small>${esc(site.county)}, ${esc(site.state)}</small></span><span>${site.stale ? 'Stale' : esc(site.index?.label ?? 'Unknown')}<br><small>${site.index ? `${Math.round(site.index.coverage * 100)}% coverage` : 'Unassessed'}</small></span></button>`).join('') || '<p>No facilities match.</p>';
  all<HTMLButtonElement>('[data-site]', siteList).forEach(button => button.onclick = () => void run(button, () => selectSite(button.dataset.site!)));
}
function stats() {
  const values = [sites.length, missions.filter(m => !m.ended).length, mission?.jha.readiness.headline ?? 'Select mission', detail ? detail.reports.filter((r: Data) => r.status === 'pending').length : 'Select facility'];
  all('#overview .stats .value').forEach((node,i) => node.textContent = String(values[i]));
  all('#overview .stats .caption')[3].textContent = 'Pending reports at selected facility';
}
const missionBox = boxes('missions')[0];
const missionInputs = all<HTMLInputElement | HTMLSelectElement>('input,select', missionBox);
['site_id','name','activity','crew_size','shift_start','shift_end'].forEach((name,i) => { missionInputs[i].name = name; missionInputs[i].required = true; });
const [missionSite,,activity] = missionInputs as HTMLSelectElement[];
(missionInputs[1] as HTMLInputElement).minLength = 3;
(missionInputs[1] as HTMLInputElement).maxLength = 120;
const crew = missionInputs[3] as HTMLInputElement; crew.type = 'number'; crew.min = '1'; crew.max = '200'; crew.value = '4';
const start = new Date(); start.setDate(start.getDate()+1); start.setHours(7,0,0,0);
const localDate = (d: Date) => new Date(d.getTime()-d.getTimezoneOffset()*60000).toISOString().slice(0,16);
missionInputs[4].value = localDate(start); missionInputs[5].value = localDate(new Date(start.getTime()+8*3600000));
$('.fields', missionBox).insertAdjacentHTML('beforeend', field('supervisor','Responsible supervisor','required minlength="2" maxlength="80"'));
const missionForm = form(missionBox, 'Save mission and build JHA', async node => {
  const body = Object.fromEntries(new FormData(node)); const site = sites.find(s => s.id === body.site_id)!;
  const saved = await api('/missions', {...body, crew_size:Number(body.crew_size),latitude:site.latitude,longitude:site.longitude,shift_start:new Date(String(body.shift_start)).toISOString(),shift_end:new Date(String(body.shift_end)).toISOString()}, 'POST');
  await loadMissions(); await openMission(saved.id); notify('Mission saved with cited JHA.');
});
missionSite.onchange = () => void run(null, () => selectSite(missionSite.value));
const summary = boxes('missions')[1];
summary.insertAdjacentHTML('afterbegin','<label class="mission-picker">Open mission<select id="mission-picker"><option value="">Choose a mission</option></select></label>');
const missionPicker = $<HTMLSelectElement>('#mission-picker');
missionPicker.onchange = () => { if (missionPicker.value) void run(null, () => openMission(missionPicker.value)); };
const decisionBox = boxes('missions')[2];
const decisionFields = $('.fields',decisionBox);
decisionFields.innerHTML = `<label>Hazard / JHA version<select name="hazard" required></select></label><label>Decision<select name="decision">${option('accept','Accept rating')}${option('mitigate','Mitigate with controls')}${option('override','Override rating')}</select></label>${field('supervisor','Supervisor name','required minlength="2" maxlength="80"')}<label>Override rating<select name="rating">${['low','elevated','high'].map(x=>option(x,x)).join('')}</select></label><label class="wide">Controls<input name="mitigation" maxlength="1000" /></label><label class="wide">Rationale<textarea name="rationale" required minlength="5" maxlength="1000"></textarea></label>`;
const decisionForm = form(decisionBox, 'Save decision', async node => {
  if (!mission) throw new Error('Select a mission first.');
  const data = Object.fromEntries(new FormData(node));
  await api(`/missions/${mission.mission.id}/decisions`, {...data,version:mission.versions.at(-1).version,rating:data.decision === 'override' ? data.rating : null,mitigation:data.decision === 'mitigate' ? data.mitigation : null}, 'POST');
  await openMission(mission.mission.id); notify('Supervisor decision recorded.');
});
function decisionState() {
  const current = mission?.jha.entries.find((e: Data) => e.hazard === $<HTMLSelectElement>('[name=hazard]',decisionForm).value);
  const decision = $<HTMLSelectElement>('[name=decision]',decisionForm);
  $('option[value=override]',decision).toggleAttribute('disabled',current?.rating === 'stop_work');
  if (current?.rating === 'stop_work' && decision.value === 'override') decision.value = 'mitigate';
  const controls = $<HTMLInputElement>('[name=mitigation]',decisionForm); controls.required = decision.value === 'mitigate'; controls.disabled = decision.value !== 'mitigate';
  $<HTMLSelectElement>('[name=rating]',decisionForm).disabled = decision.value !== 'override';
  $<HTMLButtonElement>('button[type=submit]',decisionForm).disabled = !mission;
}
decisionForm.onchange = decisionState;
const reportBox = boxes('reports')[0];
$('.fields',reportBox).innerHTML = `<label>Mission / Facility<input name="facility" readonly /></label>${field('latitude','Latitude','type="number" step="any" required min="-90" max="90"')}${field('longitude','Longitude','type="number" step="any" required min="-180" max="180"')}<label class="wide">Observation<textarea name="description" required minlength="10" maxlength="2000" placeholder="Describe the conditions or hazard"></textarea></label><label>Hazard categories<select name="classification">${hazards.map(x=>option(x,x.replaceAll('_',' '))).join('')}</select></label><label>Photo<input name="image" type="file" accept="image/jpeg,image/png,image/webp" /></label>`;
$<HTMLInputElement>('input[type=checkbox]',reportBox).name = 'jev_consent';
$('.note',reportBox).textContent = 'Reports start pending. Only verified reports count toward ratings. Photos stay local.';
const reportForm = form(reportBox, 'Submit observation', async node => {
  if (!selected) throw new Error('Select a facility first.');
  const data = new FormData(node); data.delete('facility'); data.set('categories','[]'); data.set('jev_consent',String($<HTMLInputElement>('[name=jev_consent]',node).checked));
  const image = data.get('image'); if (image instanceof File && !image.size) data.delete('image');
  if (image instanceof File && image.size > 5*1024*1024) throw new Error('Photo must be at most 5 MB.');
  await api(`/sites/${selected}/reports`,data,'POST');
  $<HTMLTextAreaElement>('textarea',node).value = ''; $<HTMLInputElement>('[name=image]',node).value = '';
  $<HTMLInputElement>('[name=jev_consent]',node).checked = false;
  await selectSite(selected); if (mission) await openMission(mission.mission.id); notify('Observation saved as pending review.');
});
const supervisorBox = boxes('briefings')[2];
const tokenInput = $<HTMLInputElement>('input',supervisorBox); tokenInput.name = 'token'; tokenInput.autocomplete = 'off'; tokenInput.value = sessionStorage.getItem('planit-token') || '';
const tokenForm = form(supervisorBox,'Save session access',async () => { sessionStorage.setItem('planit-token', tokenInput.value.trim()); notify('Supervisor token saved for this browser session.'); });
tokenForm.insertAdjacentHTML('beforeend','<button type="button" id="clear-access">Clear access</button><p><a href="/docs" target="_blank" rel="noreferrer">API documentation</a> / <button type="button" id="show-audit">View audit</button></p><div id="audit-results"></div>');
$('#clear-access').addEventListener('click',() => { sessionStorage.removeItem('planit-token'); tokenInput.value=''; notify('Session access cleared.'); });
$('#show-audit').addEventListener('click',() => void run($<HTMLButtonElement>('#show-audit'),async () => { const rows = await api('/audit'); $('#audit-results').textContent = JSON.stringify(rows,null,2); }));
function evidence(items: Data[]) { return items.map(item => `<p><strong>${esc(item.kind)} / ${esc(item.source)}</strong><br>${esc(item.summary)}${item.counts === false ? ' / Not counted' : ''}</p>`).join('') || 'Unavailable'; }
async function selectSite(id: string) {
  selected = id; missionSite.value = id; renderSites();
  const site = sites.find(s=>s.id===id)!; if (!site) return;
  $<HTMLInputElement>('[name=facility]',reportForm).value = site.name;
  $<HTMLInputElement>('[name=latitude]',reportForm).value = site.latitude;
  $<HTMLInputElement>('[name=longitude]',reportForm).value = site.longitude;
  $('#conditions .caption').textContent = `${site.name} / ${site.operator} / ${site.county}, ${site.state} / ${site.latitude}, ${site.longitude} / ${site.capacity_mw} MW / ${site.technology}`;
  const request = ++siteRequest; detail = null; stats();
  boxes('conditions').forEach(box=> { $('.value',box).textContent = 'Loading'; });
  const data = await api(`/sites/${id}`); if (request !== siteRequest) return;
  detail = data; renderConditions(); renderReports(); stats(); await loadBriefings(id);
}
function renderConditions() {
  if (!detail) return;
  const readings = boxes('conditions'), weather = detail.weather || {}, s = weather.summary || {};
  const values = [`${s.apparent_max_c ?? 'Unavailable'} °C / ${s.gust_max_kmh ?? 'Unavailable'} km/h`,`${s.rain_24h_mm ?? 'Unavailable'} mm / ${s.soil_moisture ?? 'Unavailable'}`,detail.lightning?.live?.active ? 'STOP WORK' : liveLightningConfigured ? 'No active hold' : 'Unavailable'];
  values.forEach((value,i)=>$('.value',readings[i]).textContent=value);
  all('p',readings[0])[1].textContent = `Predicted / ${weather.source || 'Unavailable'} / ${date(weather.feed?.fetched_at)}`;
  all('p',readings[1])[1].textContent = `Predicted / ${weather.source || 'Unavailable'} / ${date(weather.feed?.fetched_at)}`;
  if (detail.lightning?.live?.active) all('p',readings[2])[0].textContent = `Stop work until ${date(detail.lightning.live.review_after)}; supervisor review required.`;
  [readings[3],readings[4]].forEach((box,i)=> { const items = detail!.nearby?.[i ? 'wildfire':'flood'] || []; $('.value',box).textContent = items.map((e:Data)=>e.rating || e.kind).join(' / ') || 'Unavailable'; all('p',box)[0].innerHTML = evidence(items); });
  const alerts = detail.alerts?.items || [];
  $('.value',readings[5]).textContent = String(alerts.length);
  all('p',readings[5])[0].textContent = alerts.map((a:Data)=>a.headline || a.event).join(' / ') || (detail.alerts?.items != null ? 'No active official alerts' : 'Alert feed unavailable');
  all('p',readings[5])[1].textContent = detail.model?.reason || 'County-scoped model results remain advisory. Missing model inputs are unavailable.';
  const sourceBox = boxes('briefings')[1];
  content(sourceBox,`<p>${esc(weather.source)} / predicted</p><p>Retrieved ${esc(date(weather.feed?.fetched_at))}</p><p>Assessment ${esc(date(detail.assessed_at))}</p><button type="button" id="refresh-feeds">Refresh feeds</button>`);
  $('#refresh-feeds').onclick = () => void run($<HTMLButtonElement>('#refresh-feeds'),async()=> {await api('/refresh',{},'POST'); notify('Refresh requested. Use the refresh button after feeds finish to inspect new data.'); await selectSite(selected); });
}
function renderReports() {
  const reviewBox = boxes('reports')[1], reports = detail?.reports || [];
  content(reviewBox, `<div class="live-list">${reports.map((r:Data)=>`<article class="report-item"><p><strong>${esc(r.id.slice(0,8))} / ${esc(r.status)}</strong> / ${esc(date(r.created_at))}</p><p>${esc(r.description)}</p>${r.image ? `<a href="/uploads/${encodeURIComponent(r.image)}" target="_blank" rel="noreferrer"><img class="report-photo" src="/uploads/${encodeURIComponent(r.image)}" alt="Crew observation" /></a>`:''}<p>Manual labels: ${esc(r.triage?.manual_categories?.join(', ') || r.classification)}</p><p>Advisory Jev: ${esc(r.triage?.provider?.reason || r.triage?.provider?.status || 'Unavailable')} / ${esc(r.triage?.provider?.categories?.join(', ') || 'No accepted suggestions')} / ${r.triage?.provider?.abstained ? 'Abstained' : 'Supervisor review required'}</p><div class="review-actions">${['verified','rejected','resolved'].map(state=>`<button type="button" data-review="${esc(r.id)}" data-status="${state}" ${r.status===state?'disabled':''}>Mark ${state}</button>`).join('')}</div></article>`).join('') || '<div class="placeholder">No reports for the selected facility.</div>'}</div><p class="note">AI suggestions do not verify reports or set ratings.</p>`);
  all<HTMLButtonElement>('[data-review]',reviewBox).forEach(button=>button.onclick=()=>void run(button,async()=> { await api(`/reports/${button.dataset.review}`,{status:button.dataset.status},'PATCH'); await selectSite(selected); if(mission) await openMission(mission.mission.id); notify(`Report marked ${button.dataset.status}.`); }));
  all<HTMLElement>('.report-item', reviewBox).forEach((article, i) => {
    const report = reports[i];
    const editor = document.createElement('details');
    editor.innerHTML = `<summary>Correct hazard labels</summary><form><label>Categories (comma-separated)<input name="categories" required value="${esc((report.triage?.manual_categories || [report.classification]).join(', '))}" /></label><p class="note">Supported: ${hazards.join(', ')}</p><button type="submit">Save labels</button></form>`;
    article.append(editor);
    const node = $('form', editor) as HTMLFormElement;
    node.onsubmit = event => {
      event.preventDefault();
      void run($<HTMLButtonElement>('button', node), async () => {
        const categories = $<HTMLInputElement>('input', node).value.split(',').map(value => value.trim()).filter(Boolean);
        if (!categories.length || categories.some(value => !hazards.includes(value))) throw new Error('Enter at least one supported hazard category.');
        await api(`/reports/${report.id}/triage`, {categories: [...new Set(categories)]}, 'PATCH');
        await selectSite(selected);
        if (mission) await openMission(mission.mission.id);
        notify('Supervisor hazard labels saved.');
      });
    };
  });
}
async function loadMissions() {
  missions = await api('/missions');
  missionPicker.innerHTML = option('','Choose a mission') + missions.map(m=>option(m.id,`${m.name} / ${m.site_name}`)).join('');
  if(mission) missionPicker.value = mission.mission.id;
  stats();
}
async function openMission(id: string) {
  const request = ++missionRequest;
  const data = await api(`/missions/${id}`); if(request !== missionRequest) return;
  mission = data; missionPicker.value=id;
  if(selected !== data.mission.site_id) await selectSite(data.mission.site_id);
  $('#missions .caption').textContent = `${data.mission.name} / ${data.jha.readiness.headline} / JHA v${data.versions.at(-1).version}`;
  const rows = all('.row',summary);
  rows[0].innerHTML = `<span>Shift / Activity</span><span>${esc(date(data.mission.shift_start))} to ${esc(date(data.mission.shift_end))} / ${esc(data.mission.activity)}</span>`;
  rows[1].innerHTML = `<span>Predicted shift coverage</span><span>${esc(data.conditions.hours_available)} / ${esc(data.conditions.hours_requested)} hours</span>`;
  rows[2].innerHTML = `<span>Version / Trigger</span><span>v${esc(data.versions.at(-1).version)} / ${esc(data.versions.at(-1)?.trigger)}</span>`;
  $('.placeholder',summary).innerHTML = data.versions.map((v:Data)=>`<details><summary>v${esc(v.version)} / ${esc(v.trigger)}</summary><pre>${esc(JSON.stringify(v.diff,null,2))}</pre></details>`).join('');
  $('#missions tbody').innerHTML = data.jha.entries.map((entry:Data)=> {const d=data.decisions.find((d:Data)=>d.hazard===entry.hazard); return `<tr><td>${label(entry.hazard)}</td><td>${label(entry.rating)}</td><td>${entry.evidence.map((e:Data)=>esc(e.kind)).join('<br>')}</td><td>${evidence(entry.evidence)}</td><td>${d ? `${label(d.decision)} / ${esc(d.actor || d.supervisor)} / v${esc(d.version)} / ${d.stale?'Stale':'Current'}<br>${esc(d.rationale)}<br>${esc(d.mitigation || '')}<br>${esc(date(d.created_at))}`:'No decision'}</td></tr>`;}).join('');
  $<HTMLSelectElement>('[name=hazard]',decisionForm).innerHTML = data.jha.entries.map((e:Data)=>option(e.hazard,`${e.hazard.replaceAll('_',' ')} / ${e.rating} / v${data.versions.at(-1).version}`)).join(''); decisionState(); stats();
  renderMemory();
}
function renderMemory() {
  const records = mission?.memory || [], memoryBoxes = boxes('memory');
  content(memoryBoxes[0],records.map((m:Data,i:number)=>`<button type="button" class="row" data-memory="${i}"><span>${esc(m.record.event_type || m.record.hazard || m.record.source)} / ${esc(m.record.narrative || 'No narrative')}</span><span>${Math.round(m.score*100)}% relevance</span></button>`).join('') || '<div class="placeholder">No matching records for this mission.</div>');
  const show=(i:number)=>{const m=records[i]; if(!m) return; content(memoryBoxes[1],Object.entries(m.factors).map(([name,value])=>{const f=value as Data;return `<div class="row"><span>${label(name)}</span><span>${Math.round(f.score*100)}% / ${esc(f.detail)}</span></div>`;}).join('')+`<p class="note">${esc(m.record.source)}. Memory informs review and never changes ratings.</p>`);};
  all<HTMLButtonElement>('[data-memory]').forEach(button=>button.onclick=()=>show(Number(button.dataset.memory))); show(0);
}
async function loadBriefings(id: string) {
  const rows=await api(`/sites/${id}/briefings`); if(id!==selected) return;
  const box=boxes('briefings')[0];
  content(box,`<div class="live-list">${rows.map((b:Data)=>`<details><summary>${esc(b.local_date)} / ${esc(b.payload.site_name)}</summary><p>${esc(date(b.generated_at))} / ${esc(b.delivery)}</p><ul>${b.payload.actions.map((a:string)=>`<li>${esc(a)}</li>`).join('')}</ul><p>${esc(b.payload.note)}</p></details>`).join('') || '<div class="placeholder">No saved briefings for this facility.</div>'}</div><button type="button" id="generate-briefing">Generate next-day briefing</button>`);
  $('#generate-briefing').onclick=()=>void run($<HTMLButtonElement>('#generate-briefing'),async()=>{const siteId=selected;await api(`/sites/${siteId}/briefings`,{},'POST');await loadBriefings(siteId);notify('Next-day briefing saved.');});
}
const overlapBoxes=boxes('overlaps');
overlapBoxes[0].insertAdjacentHTML('beforeend','<label class="check"><input id="utilities-only" type="checkbox" checked />Electric utilities only</label><label class="check"><input id="cross-state" type="checkbox" />Cross-state pairs only</label>');
async function loadOverlaps() {
  const query=new URLSearchParams({utilities_only:String($<HTMLInputElement>('#utilities-only').checked),cross_state:String($<HTMLInputElement>('#cross-state').checked),limit:'100'});
  const data=await api(`/plan-overlaps?${query}`);
  content(overlapBoxes[1],`<p>${esc(data.total)} pairs / ${esc(data.entities)} entities</p><div class="live-list">${data.pairs.map((p:Data,i:number)=>`<button class="row" type="button" data-pair="${i}"><span><strong>${esc(p.a.plant_name)} / ${esc(p.a.entity_name)}</strong><br>Planned ${esc(p.a.planned_month)}<br><strong>${esc(p.b.plant_name)} / ${esc(p.b.entity_name)}</strong><br>Planned ${esc(p.b.planned_month)}</span><span>${esc(p.distance_miles)} miles<br>${esc(p.months_apart)} months<br>${label(p.kind)}${p.possible_affiliates ? '<br>Possible affiliates (name heuristic)' : ''}</span></button>`).join('') || '<p>No matching pairs.</p>'}</div><p class="note">${safeLink(data.source.url,'EIA source')} / ${esc(data.note)}</p><p class="note">Suspect coordinates excluded: ${data.suspect_coordinates.map((p:Data)=>esc(p.plant_name)).join(', ') || 'None'}</p>`);
  overlapMarkers.clearLayers();
  data.pairs.forEach((p:Data)=>{[p.a,p.b].forEach(point=>L.circleMarker([point.latitude,point.longitude],{color:'#000',fillColor:'#fff',fillOpacity:1,radius:4}).bindTooltip(esc(point.plant_name)).addTo(overlapMarkers));});
  all<HTMLButtonElement>('[data-pair]').forEach(button=>button.onclick=()=>{const p=data.pairs[Number(button.dataset.pair)];overlapMap.fitBounds([[p.a.latitude,p.a.longitude],[p.b.latitude,p.b.longitude]],{padding:[30,30],maxZoom:10});});
}
$('#utilities-only').onchange = $('#cross-state').onchange = ()=>void run(null,loadOverlaps);
async function init() {
  notify('Loading facilities and missions...'); decisionState();
  const [inventory,activities,health]=await Promise.all([api('/sites'),api('/activities'),api('/health')]); sites=inventory.sites; liveLightningConfigured=health.lightning_live_configured;
  missionSite.innerHTML=sites.map(s=>option(s.id,`${s.name} / ${s.state}`)).join(''); activity.innerHTML=activities.map((a:Data)=>option(a.id,a.label)).join('');
  markers.clearLayers(); sites.forEach(site=>L.circleMarker([site.latitude,site.longitude],{color:'#000',fillColor:'#fff',fillOpacity:1,radius:5}).bindTooltip(esc(site.name)).on('click',()=>void run(null,()=>selectSite(site.id))).addTo(markers));
  renderSites(); await loadMissions();
  if(sites.length) await selectSite(sites[0].id);
  if (mission || missions.length) await openMission(mission?.mission.id || missions[0].id);
  else {
    $('#missions .caption').textContent = 'Plan a mission to build its first JHA.';
    all('.row span:last-child', summary).forEach(node => node.textContent = 'No mission selected');
    $('.placeholder', summary).textContent = 'Version history appears after a mission is saved.';
    $('#missions tbody').innerHTML = '<tr><td colspan="5">No mission selected. Plan a mission to load cited hazard entries.</td></tr>';
    boxes('memory').forEach(box => content(box, '<div class="placeholder">Select a mission to inspect relevant historical records.</div>'));
  }
  notify('Ready. Select a facility or mission. Supervisor access is required to save changes.');
  void run(null,loadOverlaps);
}
const retry=document.createElement('button');retry.type='button';retry.textContent='Reload data';retry.className='reload-data';status.after(retry);retry.onclick=()=>void run(retry,init);
void run(null,init);
