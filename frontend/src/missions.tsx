import React, { useEffect, useRef, useState } from "react";
import L from "leaflet";
import {
  Check,
  ClipboardList,
  CloudRain,
  GitCompare,
  History,
  MapPin,
  Plus,
  RefreshCw,
  Shield,
  Thermometer,
  TriangleAlert,
  Wind,
  X,
  Zap,
} from "lucide-react";

type Api = (path: string, options?: RequestInit) => Promise<any>;
type Action = (fn: () => Promise<any>, message: string) => Promise<boolean>;

const fmt = (v: number | null | undefined, digits = 1) => (v == null ? "—" : v.toFixed(digits));
const zoned = (v: string | null | undefined, tz?: string) =>
  v
    ? new Date(v).toLocaleString([], { timeZone: tz, month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", timeZoneName: "short" })
    : "—";
const KIND_LABEL: Record<string, string> = {
  observed: "Observed",
  reported: "Reported · unverified",
  predicted: "Predicted",
  official_alert: "Official alert",
  simulated: "Simulated",
  unavailable: "Unavailable",
  ai_suggestion: "AI suggestion",
  historical: "Historical",
};
const RATING_LABEL: Record<string, string> = {
  stop_work: "Stop work",
  high: "High",
  elevated: "Elevated",
  low: "Low",
  unknown: "Unknown",
};
const localInput = (d: Date) => new Date(d.getTime() - d.getTimezoneOffset() * 60000).toISOString().slice(0, 16);

export function KindBadge({ kind }: { kind: string }) {
  return <span className={"kind-badge kind-" + kind}>{KIND_LABEL[kind] || kind}</span>;
}

function FactorBars({ factors }: { factors: Record<string, { score: number; detail: string }> }) {
  return (
    <div className="factor-bars">
      {Object.entries(factors).map(([name, f]) => (
        <div key={name} title={f.detail}>
          <span>{name.replaceAll("_", " ")}</span>
          <div className="bar-track">
            <span style={{ width: `${Math.round(f.score * 100)}%` }} />
          </div>
          <small>{f.detail}</small>
        </div>
      ))}
    </div>
  );
}

export function MissionsView({
  api,
  action,
  busy,
  sites,
  selectedSite,
  revision,
  onReport,
}: {
  api: Api;
  action: Action;
  busy: boolean;
  sites: any[];
  selectedSite: string | null;
  revision: number;
  onReport: (siteId: string) => void;
}) {
  const [list, setList] = useState<any[]>([]);
  const [activities, setActivities] = useState<any[]>([]);
  const [current, setCurrent] = useState<string | null>(null);
  const [detail, setDetail] = useState<any>(null);
  const [planning, setPlanning] = useState(false);
  const [error, setError] = useState("");
  const currentRef = useRef(current);
  currentRef.current = current;
  async function loadList() {
    try {
      setList(await api("/missions"));
    } catch (e) {
      setError(String(e));
    }
  }
  async function loadDetail(id: string) {
    try {
      const d = await api("/missions/" + id);
      if (currentRef.current === id) setDetail(d);
    } catch (e) {
      setError(String(e));
    }
  }
  useEffect(() => {
    api("/activities").then(setActivities).catch((e) => setError(String(e)));
  }, []);
  useEffect(() => {
    loadList();
    if (current) loadDetail(current);
  }, [revision]);
  useEffect(() => {
    setDetail(null);
    if (!current) return;
    loadDetail(current);
    const timer = setInterval(() => loadDetail(current), 30000);
    return () => clearInterval(timer);
  }, [current]);
  useEffect(() => {
    if (!current && list.length) setCurrent(list[list.length - 1].id);
  }, [list]);
  return (
    <div className="missions-layout">
      <section className="panel mission-list">
        <div className="panel-header">
          <div>
            <h2>
              Missions <span className="small-count">{list.length}</span>
            </h2>
            <p>Planned work tied to a facility and shift</p>
          </div>
          <button className="button primary" disabled={busy} onClick={() => setPlanning(!planning)}>
            <Plus size={15} />
            Plan mission
          </button>
        </div>
        {error && (
          <p className="form-error" role="alert">
            {error}
          </p>
        )}
        {planning && (
          <MissionForm
            sites={sites}
            activities={activities}
            selectedSite={selectedSite}
            busy={busy}
            onSubmit={async (body) => {
              let created: any = null;
              const ok = await action(async () => {
                created = await api("/missions", { method: "POST", body: JSON.stringify(body) });
              }, "Mission saved. Its first Job Hazard Analysis version was computed from current sources.");
              if (ok && created) {
                setPlanning(false);
                await loadList();
                setCurrent(created.id);
              }
            }}
          />
        )}
        <div className="mission-rows">
          {[...list].reverse().map((m) => (
            <button key={m.id} className={"mission-row" + (m.id === current ? " selected" : "")} onClick={() => setCurrent(m.id)}>
              <span className={"rating-dot rating-" + (m.readiness?.status || "unknown")} />
              <span>
                {m.name}
                <small>
                  {m.site_name} · {zoned(m.shift_start)} {m.ended ? "· ended" : ""}
                </small>
              </span>
              <span className="tiny-tag">v{m.version ?? "—"}</span>
            </button>
          ))}
          {!list.length && <p className="empty">No missions yet. Plan one to build its hazard analysis.</p>}
        </div>
      </section>
      <section className="panel mission-detail">
        {!current ? (
          <div className="empty">Select or plan a mission.</div>
        ) : !detail ? (
          <div className="loading">
            <RefreshCw className="spin" size={20} />
            Loading the mission's hazard analysis…
          </div>
        ) : (
          <MissionDetail
            detail={detail}
            api={api}
            action={action}
            busy={busy}
            onReport={() => onReport(detail.site.id)}
          />
        )}
      </section>
    </div>
  );
}

function MissionForm({
  sites,
  activities,
  selectedSite,
  busy,
  onSubmit,
}: {
  sites: any[];
  activities: any[];
  selectedSite: string | null;
  busy: boolean;
  onSubmit: (body: any) => Promise<void>;
}) {
  const start = new Date();
  start.setDate(start.getDate() + 1);
  start.setHours(7, 0, 0, 0);
  const [siteId, setSiteId] = useState(selectedSite || sites[0]?.id || "");
  const site = sites.find((s) => s.id === siteId);
  const [lat, setLat] = useState(String(site?.latitude ?? ""));
  const [lon, setLon] = useState(String(site?.longitude ?? ""));
  useEffect(() => {
    if (site) {
      setLat(String(site.latitude));
      setLon(String(site.longitude));
    }
  }, [siteId]);
  return (
    <form
      className="mission-form"
      onSubmit={(e) => {
        e.preventDefault();
        const f = new FormData(e.currentTarget);
        onSubmit({
          site_id: siteId,
          name: String(f.get("name")),
          activity: String(f.get("activity")),
          latitude: Number(lat),
          longitude: Number(lon),
          shift_start: new Date(String(f.get("shift_start"))).toISOString(),
          shift_end: new Date(String(f.get("shift_end"))).toISOString(),
          crew_size: Number(f.get("crew_size")),
          supervisor: String(f.get("supervisor")),
        });
      }}
    >
      <label className="field">
        Facility
        <select value={siteId} onChange={(e) => setSiteId(e.target.value)} required>
          {sites.map((s) => (
            <option key={s.id} value={s.id}>
              {s.name} · {s.state}
            </option>
          ))}
        </select>
      </label>
      <label className="field">
        Mission name
        <input name="name" required minLength={3} maxLength={120} placeholder="e.g. Feeder 12 trench repair" />
      </label>
      <label className="field">
        Activity
        <select name="activity" required>
          {activities.map((a) => (
            <option key={a.id} value={a.id}>
              {a.label}
            </option>
          ))}
        </select>
      </label>
      <div className="form-grid">
        <label className="field">
          Shift start (your local time)
          <input name="shift_start" type="datetime-local" required defaultValue={localInput(start)} />
        </label>
        <label className="field">
          Shift end
          <input name="shift_end" type="datetime-local" required defaultValue={localInput(new Date(start.getTime() + 8 * 3600000))} />
        </label>
      </div>
      <div className="form-grid">
        <label className="field">
          Work latitude
          <input type="number" step="any" value={lat} onChange={(e) => setLat(e.target.value)} required />
        </label>
        <label className="field">
          Work longitude
          <input type="number" step="any" value={lon} onChange={(e) => setLon(e.target.value)} required />
        </label>
      </div>
      <div className="form-grid">
        <label className="field">
          Crew size
          <input name="crew_size" type="number" min={1} max={200} defaultValue={4} required />
        </label>
        <label className="field">
          Responsible supervisor
          <input name="supervisor" required minLength={2} maxLength={80} />
        </label>
      </div>
      <p className="body-note">Work location starts at the facility and must stay within 50 miles. Shifts last at most 24 hours.</p>
      <button className="button primary full" disabled={busy}>
        Save mission and build JHA
      </button>
    </form>
  );
}

function Readiness({ readiness, undecided }: { readiness: any; undecided: string[] }) {
  if (readiness.status === "stop_work")
    return (
      <div className="stop-banner">
        <Zap size={18} />
        <strong>{readiness.headline}</strong>
        <span>Deterministic rule. No AI output or supervisor override can lift it.</span>
      </div>
    );
  if (readiness.status === "simulated_stop_work")
    return (
      <div className="replay-banner mission-banner">
        <Zap size={18} />
        <strong>{readiness.headline}</strong>
        <span>Same deterministic rule, fed by a labeled replay strike.</span>
      </div>
    );
  return (
    <div className={"inline-note mission-readiness rating-" + readiness.status}>
      <Shield size={15} />
      <span>
        <b>{RATING_LABEL[readiness.status]}</b> · {readiness.headline}
        {readiness.status === "high" &&
          (undecided.length
            ? ` Still without a current decision: ${undecided.join(", ")}.`
            : " Every high-rated hazard has a current supervisor decision.")}
        {readiness.unknown_hazards.length > 0 && ` Unknown inputs: ${readiness.unknown_hazards.join(", ").replaceAll("_", " ")}.`}
      </span>
    </div>
  );
}

function MissionDetail({ detail, api, action, busy, onReport }: { detail: any; api: Api; action: Action; busy: boolean; onReport: () => void }) {
  const { mission, site, conditions, jha, versions, decisions } = detail;
  const tz = site.timezone;
  const [deciding, setDeciding] = useState<string | null>(null);
  return (
    <>
      <div className="panel-header">
        <div>
          <div className="eyebrow">
            MISSION · {detail.activity.label.toUpperCase()} · JHA v{versions[versions.length - 1].version}
          </div>
          <h2 className="site-heading">{mission.name}</h2>
          <p>
            {site.name} ({site.county}, {site.state}) · {zoned(mission.shift_start, tz)} to {zoned(mission.shift_end, tz)} · crew of{" "}
            {mission.crew_size} · supervisor {mission.supervisor}
            {mission.ended && " · shift ended, analysis frozen"}
          </p>
        </div>
        <button className="button secondary" disabled={busy || mission.ended} onClick={onReport}>
          <Plus size={15} />
          Crew report
        </button>
      </div>
      <div className="mission-body">
        <Readiness
          readiness={jha.readiness}
          undecided={jha.entries
            .filter((e: any) => e.rating === "high" && !decisions.some((d: any) => d.hazard === e.hazard && !d.stale))
            .map((e: any) => e.label)}
        />
        {detail.checked && (
          <p className={"body-note" + (detail.checked.stale ? " warning-text" : "")}>
            JHA v{versions[versions.length - 1].version} unchanged since {zoned(versions[versions.length - 1].created_at, tz)}. Last re-checked{" "}
            {zoned(detail.checked.checked_at, tz)} against the site assessment from {zoned(detail.checked.assessed_at, tz)}
            {detail.checked.stale ? ". That assessment is stale because reassessment failed; treat this analysis as out of date." : "."}
          </p>
        )}
        <h3 className="section-subtitle">
          Shift conditions <KindBadge kind={conditions.kind} />
        </h3>
        <div className="condition-cards">
          <Card icon={<Thermometer />} label="Apparent max" value={fmt(conditions.apparent_max_c) + "°C"} note="Screening, not WBGT" />
          <Card icon={<Wind />} label="Gust max" value={fmt(conditions.gust_max_kmh, 0) + " km/h"} note={`${detail.activity.label} limits apply`} />
          <Card icon={<CloudRain />} label="Shift rainfall" value={fmt(conditions.rain_mm) + " mm"} note={`${conditions.wet_hours ?? "—"} wet hours`} />
        </div>
        <p className="body-note">
          {conditions.hours_available} of {conditions.hours_requested} shift hours covered by the {conditions.source} forecast fetched{" "}
          {zoned(conditions.fetched_at, tz)}. {conditions.note} Missing hours stay unknown.
        </p>

        <div className="section-title jha-title">
          <div>
            <h3>
              <ClipboardList size={15} /> Job Hazard Analysis
            </h3>
            <p>Deterministic ratings. Every line cites its source. AI suggestions are listed but never set a rating.</p>
          </div>
        </div>
        <div className="kind-legend">
          {Object.keys(KIND_LABEL)
            .filter((k) => k !== "historical")
            .map((k) => (
              <KindBadge key={k} kind={k} />
            ))}
        </div>
        <div className="jha-entries">
          {jha.entries.map((e: any) => {
            const decision = decisions.find((d: any) => d.hazard === e.hazard);
            return (
              <article key={e.hazard} className={"jha-entry rating-" + e.rating}>
                <header>
                  <strong>{e.label}</strong>
                  <span className={"rating-pill rating-" + e.rating}>
                    {RATING_LABEL[e.rating]}
                    {e.simulated && " · simulated"}
                  </span>
                  {e.incomplete && <span className="tiny-tag">INCOMPLETE INPUTS</span>}
                  {e.needs_review && <span className="tiny-tag">NEEDS REVIEW</span>}
                  <small>activity sensitivity {fmt(e.sensitivity, 1)}</small>
                </header>
                <ul className="evidence-list">
                  {e.evidence.map((x: any) => (
                    <li key={x.id}>
                      <KindBadge kind={x.kind} />
                      <span>
                        <b>{x.source}</b> {x.summary}
                        {x.rating && <em> → {RATING_LABEL[x.rating]}</em>}
                        {!x.counts && <em> · not counted</em>}
                        {x.ai_agrees && <em> · Jev agreed</em>}
                      </span>
                      {x.kind === "reported" && x.ref && (
                        <span className="evidence-actions">
                          <button
                            disabled={busy}
                            onClick={() =>
                              action(
                                () => api("/reports/" + x.ref, { method: "PATCH", body: JSON.stringify({ status: "verified" }) }),
                                "Report verified. The mission JHA was recomputed.",
                              )
                            }
                          >
                            <Check size={12} />
                            Verify
                          </button>
                          <button
                            disabled={busy}
                            onClick={() =>
                              action(
                                () => api("/reports/" + x.ref, { method: "PATCH", body: JSON.stringify({ status: "rejected" }) }),
                                "Report rejected. The mission JHA was recomputed.",
                              )
                            }
                          >
                            <X size={12} />
                            Reject
                          </button>
                        </span>
                      )}
                    </li>
                  ))}
                </ul>
                <details>
                  <summary>Controls</summary>
                  <ul>
                    {e.controls.map((c: string) => (
                      <li key={c}>{c}</li>
                    ))}
                  </ul>
                </details>
                <div className="decision-line">
                  {decision ? (
                    <span className={decision.stale ? "decision stale" : "decision"}>
                      {decision.decision === "override" ? `Override to ${RATING_LABEL[decision.rating]}` : decision.decision === "mitigate" ? "Mitigated" : "Accepted"} by{" "}
                      {decision.supervisor}
                      {decision.identity === "token" ? " (personal token)" : " (name typed under the shared token)"} on v{decision.version},{" "}
                      {zoned(decision.created_at, tz)}: "{decision.rationale}"
                      {decision.mitigation && ` Controls: ${decision.mitigation}`}
                      {decision.stale && " · conditions changed since, review again"}
                    </span>
                  ) : (
                    <span className="decision none">No supervisor decision recorded.</span>
                  )}
                  {!mission.ended && (
                    <button className="text-button" onClick={() => setDeciding(deciding === e.hazard ? null : e.hazard)}>
                      Record decision
                    </button>
                  )}
                </div>
                {deciding === e.hazard && (
                  <DecisionForm
                    entry={e}
                    busy={busy}
                    onSubmit={async (body) => {
                      const ok = await action(
                        () =>
                          api(`/missions/${mission.id}/decisions`, {
                            method: "POST",
                            body: JSON.stringify({ ...body, version: versions[versions.length - 1].version }),
                          }),
                        "Supervisor decision recorded against the current JHA version.",
                      );
                      if (ok) setDeciding(null);
                    }}
                  />
                )}
              </article>
            );
          })}
        </div>

        <h3 className="section-subtitle">
          <History size={15} /> Version history <span className="small-count">{versions.length}</span>
        </h3>
        <ol className="version-list">
          {[...versions].reverse().map((v: any) => (
            <li key={v.version}>
              <div>
                <b>v{v.version}</b> <span className={"rating-pill rating-" + v.readiness.status}>{RATING_LABEL[v.readiness.status] || "Simulated stop"}</span>
                <small>{zoned(v.created_at, tz)}</small>
              </div>
              <p>{v.trigger}</p>
              {v.version > 1 && (
                <ul>
                  {v.diff.map((d: any) => (
                    <li key={d.hazard}>
                      <b>{d.hazard.replaceAll("_", " ")}</b>{" "}
                      {d.change === "rating" ? `${RATING_LABEL[d.before]} → ${RATING_LABEL[d.after]}` : d.change === "evidence" ? `evidence changed (${RATING_LABEL[d.after]})` : `${d.change} (${RATING_LABEL[d.after || d.before]})`}
                      {d.evidence_added.map((x: string) => (
                        <span key={"+" + x} className="diff-add">+ {x}</span>
                      ))}
                      {d.evidence_removed.map((x: string) => (
                        <span key={"-" + x} className="diff-remove">− {x}</span>
                      ))}
                    </li>
                  ))}
                </ul>
              )}
            </li>
          ))}
        </ol>

        <h3 className="section-subtitle">Site Hazard Memory</h3>
        <p className="body-note">
          {detail.memory_source.noaa} NOAA records and {detail.memory_source.crew} crew report{" "}
          {detail.memory_source.crew === 1 ? "label" : "labels"} (verified or resolved) for this facility. {detail.memory_source.note} Memory gives context and never changes a rating.
        </p>
        <div className="memory-list">
          {detail.memory.map((m: any) => (
            <article key={m.record.id + m.record.hazard} className="memory-card">
              <header>
                <KindBadge kind="historical" />
                <strong>{m.record.event_type}</strong>
                <span>{m.record.begin.slice(0, 10)}</span>
                <span className="tiny-tag">{m.record.source}</span>
                <b className="memory-score">{fmt(m.score * 100, 0)}</b>
              </header>
              <p>{m.record.narrative || "No narrative in the source record."}</p>
              <FactorBars factors={m.factors} />
            </article>
          ))}
          {!detail.memory.length && <p className="empty">No historical records matched this facility.</p>}
        </div>

        <h3 className="section-subtitle">
          <GitCompare size={15} /> Other entities' public construction plans nearby
        </h3>
        <p className="body-note">
          {detail.planned_source?.inventory}. Compared with this mission's location and shift month; the facility operator ({site.operator}) is excluded. {detail.planned_source?.note}
        </p>
        <div className="planned-list">
          {detail.planned_nearby.map((p: any) => (
            <div key={p.project.plant_id + p.project.entity_id} className="planned-row">
              <MapPin size={13} />
              <span>
                <b>{p.project.entity_name}</b> · {p.project.plant_name} · {fmt(p.project.capacity_mw, 0)} MW {p.project.technologies.join(", ")}
                <small>
                  {p.distance_miles} mi away · planned {p.project.planned_month ?? "month unknown"} ({p.months_apart ?? "?"} months from this shift) ·{" "}
                  {p.project.statuses.join("; ")}
                </small>
              </span>
              <span className="tiny-tag">{p.kind.replaceAll("_", " ").toUpperCase()}</span>
            </div>
          ))}
          {!detail.planned_nearby.length && <p className="empty">No other entity's planned project is within the overlap thresholds.</p>}
        </div>
      </div>
    </>
  );
}

function Card({ icon, label, value, note }: { icon: React.ReactNode; label: string; value: string; note: string }) {
  return (
    <div className="condition">
      <span>
        {icon}
        {label}
      </span>
      <strong>{value}</strong>
      <small>{note}</small>
    </div>
  );
}

function DecisionForm({ entry, busy, onSubmit }: { entry: any; busy: boolean; onSubmit: (b: any) => Promise<void> }) {
  const [kind, setKind] = useState("accept");
  const locked = entry.rating === "stop_work";
  return (
    <form
      className="decision-form"
      onSubmit={(e) => {
        e.preventDefault();
        const f = new FormData(e.currentTarget);
        onSubmit({
          hazard: entry.hazard,
          decision: kind,
          rationale: String(f.get("rationale")),
          supervisor: String(f.get("supervisor")),
          rating: kind === "override" ? String(f.get("rating")) : null,
          mitigation: kind === "mitigate" ? String(f.get("mitigation")) : null,
        });
      }}
    >
      <div className="form-grid">
        <label className="field">
          Decision
          <select value={kind} onChange={(e) => setKind(e.target.value)}>
            <option value="accept">Accept rating and controls</option>
            <option value="mitigate">Mitigate with controls</option>
            <option value="override" disabled={locked}>
              Override rating{locked ? " (not allowed for stop work)" : ""}
            </option>
          </select>
        </label>
        <label className="field">
          Supervisor name
          <input name="supervisor" required minLength={2} maxLength={80} />
        </label>
      </div>
      {kind === "override" && (
        <label className="field">
          Supervisor rating
          <select name="rating">
            <option value="low">Low</option>
            <option value="elevated">Elevated</option>
            <option value="high">High</option>
          </select>
        </label>
      )}
      {kind === "mitigate" && (
        <label className="field">
          Controls applied
          <input name="mitigation" required maxLength={1000} />
        </label>
      )}
      <label className="field">
        Rationale
        <textarea name="rationale" required minLength={5} maxLength={1000} rows={2} />
      </label>
      {locked && (
        <p className="inline-note warning">
          <TriangleAlert size={14} /> Lightning stop work comes from the deterministic strike timer. Record a mitigation or wait for the timer review.
        </p>
      )}
      <button className="button secondary" disabled={busy}>
        Save decision
      </button>
    </form>
  );
}

function OverlapMap({ pairs }: { pairs: any[] }) {
  const container = useRef<HTMLDivElement>(null),
    map = useRef<L.Map | null>(null),
    layer = useRef<L.LayerGroup | null>(null);
  useEffect(() => {
    if (!container.current) return;
    map.current = L.map(container.current, { zoomControl: false }).setView([37.4, -96.5], 4);
    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap contributors</a>',
      maxZoom: 19,
    }).addTo(map.current);
    L.control.zoom({ position: "bottomright" }).addTo(map.current);
    layer.current = L.layerGroup().addTo(map.current);
    return () => {
      map.current?.remove();
      map.current = null;
    };
  }, []);
  useEffect(() => {
    if (!layer.current) return;
    layer.current.clearLayers();
    pairs.forEach((p) => {
      const a: [number, number] = [p.a.latitude, p.a.longitude],
        b: [number, number] = [p.b.latitude, p.b.longitude];
      const color = p.kind === "close_and_concurrent" ? "#c0613f" : p.kind === "close" ? "#c09b50" : "#6b8970";
      L.polyline([a, b], { color, weight: 2, opacity: 0.8 }).addTo(layer.current!);
      [p.a, p.b].forEach((x) => {
        const tip = document.createElement("span");
        tip.textContent = `${x.entity_name} · ${x.plant_name} · ${x.state} · planned ${x.planned_month ?? "?"}`;
        L.circleMarker([x.latitude, x.longitude], { radius: 4, color, fillColor: color, fillOpacity: 0.9, weight: 1 })
          .bindTooltip(tip)
          .addTo(layer.current!);
      });
    });
  }, [pairs]);
  return <div className="map overlap-map" ref={container} aria-label="Map of flagged planned-project overlaps" />;
}

export function OverlapsView({ api }: { api: Api }) {
  const [utilities, setUtilities] = useState(true),
    [crossState, setCrossState] = useState(false),
    [kind, setKind] = useState("");
  const [data, setData] = useState<any>(null),
    [error, setError] = useState("");
  useEffect(() => {
    setData(null);
    api(`/plan-overlaps?utilities_only=${utilities}&cross_state=${crossState}&limit=60${kind ? "&kind=" + kind : ""}`)
      .then(setData)
      .catch((e) => setError(String(e)));
  }, [utilities, crossState, kind]);
  return (
    <section className="panel overlap-page">
      <div className="panel-header">
        <div>
          <div className="eyebrow">PUBLIC PLANS · {data?.inventory?.toUpperCase() || "EIA-860M"}</div>
          <h2>Where planned construction overlaps</h2>
          <p>
            Pairs of planned plants owned by different entities: within {data?.thresholds.close_miles ?? 25} miles, or within{" "}
            {data?.thresholds.region_miles ?? 75} miles and planned to start within {data?.thresholds.concurrent_months ?? 6} months of each other.
          </p>
        </div>
      </div>
      <div className="overlap-filters">
        <label className="consent-option">
          <input type="checkbox" checked={utilities} onChange={(e) => setUtilities(e.target.checked)} /> Electric utilities only
        </label>
        <label className="consent-option">
          <input type="checkbox" checked={crossState} onChange={(e) => setCrossState(e.target.checked)} /> Different states only
        </label>
        <label className="field">
          Overlap type
          <select value={kind} onChange={(e) => setKind(e.target.value)}>
            <option value="">All types</option>
            <option value="close_and_concurrent">Close and concurrent</option>
            <option value="close">Close</option>
            <option value="concurrent_in_region">Concurrent in region</option>
          </select>
        </label>
      </div>
      {error && <p className="form-error">{error}</p>}
      {!data ? (
        <div className="loading">
          <RefreshCw className="spin" size={20} />
          Comparing planned projects…
        </div>
      ) : (
        <>
          <div className="method-grid overlap-stats">
            <div>
              <strong>{data.projects.toLocaleString()}</strong>
              <span>planned plant records</span>
            </div>
            <div>
              <strong>{data.entities.toLocaleString()}</strong>
              <span>owner entities</span>
            </div>
            <div>
              <strong>{data.total.toLocaleString()}</strong>
              <span>flagged pairs</span>
            </div>
            <div>
              <strong>{data.pairs.length}</strong>
              <span>shown, highest score first</span>
            </div>
          </div>
          <OverlapMap pairs={data.pairs} />
          <p className="body-note overlap-note">
            {data.note} Source: <a href={data.source.url}>{data.source.url}</a>, retrieved {data.source.retrieved_at.slice(0, 10)}, SHA-256{" "}
            {data.source.sha256.slice(0, 12)}…
          </p>
          {data.suspect_coordinates.length > 0 && (
            <p className="inline-note warning overlap-note">
              Left out of distance comparisons because the source gives them the same coordinates as a plant in another state:{" "}
              {data.suspect_coordinates.map((p: any) => `${p.plant_name} (${p.county}, ${p.state})`).join("; ")}.
            </p>
          )}
          <div className="overlap-list">
            {data.pairs.map((p: any) => (
              <article key={p.a.plant_id + p.a.entity_id + p.b.plant_id + p.b.entity_id} className="overlap-card">
                <header>
                  <span className={"tiny-tag kind-" + p.kind}>{p.kind.replaceAll("_", " ").toUpperCase()}</span>
                  {p.cross_state && <span className="tiny-tag">CROSS-STATE</span>}
                  {p.possible_affiliates && <span className="tiny-tag warn">POSSIBLE AFFILIATES · NAME MATCH</span>}
                  <b className="memory-score">{fmt(p.score * 100, 0)}</b>
                </header>
                <div className="overlap-pair">
                  {[p.a, p.b].map((x: any) => (
                    <div key={x.plant_id + x.entity_id}>
                      <strong>{x.entity_name}</strong>
                      <span>
                        {x.plant_name} · {x.county}, {x.state} · {fmt(x.capacity_mw, 0)} MW {x.technologies.join(", ")}
                      </span>
                      <small>
                        Planned {x.planned_month ?? "month unknown"} · {x.sector} · BA {x.balancing_authority ?? "unknown"} · {x.statuses.join("; ")}
                      </small>
                    </div>
                  ))}
                </div>
                <FactorBars factors={p.factors} />
              </article>
            ))}
          </div>
        </>
      )}
    </section>
  );
}
