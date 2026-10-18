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

import { observeMapResize } from "./map";

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
  const [loading, setLoading] = useState(true);
  const [detailError, setDetailError] = useState("");
  const currentRef = useRef(current);
  currentRef.current = current;
  async function loadList() {
    setLoading(true);
    setError("");
    try {
      const [missions, options] = await Promise.all([api("/missions"), api("/activities")]);
      setList(missions);
      setActivities(options);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }
  async function loadDetail(id: string) {
    try {
      const d = await api("/missions/" + id);
      if (currentRef.current === id) {
        setDetail(d);
        setList(items => items.map(m => m.id === id
          ? { ...m, ...d.mission, version: d.versions[d.versions.length - 1].version, readiness: d.jha.readiness }
          : m));
        setDetailError("");
      }
    } catch (e) {
      if (currentRef.current === id) setDetailError(e instanceof Error ? e.message : String(e));
    }
  }
  useEffect(() => {
    loadList();
    if (current) loadDetail(current);
  }, [revision]);
  useEffect(() => {
    setDetail(null);
    setDetailError("");
    if (!current) return;
    loadDetail(current);
    const timer = setInterval(() => loadDetail(current), 30000);
    return () => clearInterval(timer);
  }, [current]);
  useEffect(() => {
    if (!current && list.length) setCurrent(list[list.length - 1].id);
  }, [list]);
  if (error) return (
    <section className="panel">
      <div className="panel-header">
        <h2>Could not load missions</h2>
        <button className="button secondary" onClick={loadList}><RefreshCw size={15} />Try again</button>
      </div>
      <div className="message error panel-message" role="alert"><TriangleAlert size={18} /><span>{error}</span></div>
    </section>
  );
  if (loading && !list.length) return (
    <section className="panel loading" role="status"><RefreshCw className="spin" size={20} />Loading missions…</section>
  );
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
            <button key={m.id} className={"mission-row" + (m.id === current ? " selected" : "")} aria-pressed={m.id === current} onClick={() => setCurrent(m.id)}>
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
          {!list.length && <p className="empty">No missions planned.</p>}
        </div>
      </section>
      <section className="panel mission-detail">
        {detailError && <div className="message error panel-message" role="alert">
          <TriangleAlert size={18} /><span>Could not update this mission. {detailError}</span>
          <button className="text-button" onClick={() => current && loadDetail(current)}>Try again</button>
        </div>}
        {!current ? (
          <div className="empty mission-placeholder">
            <ClipboardList size={28} aria-hidden="true" />
            <h3>{list.length ? "Select a mission" : "Plan your first mission"}</h3>
            <p>Choose a facility, activity and shift to review its job hazard analysis.</p>
          </div>
        ) : !detail ? (
          !detailError &&
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
            onApplied={() => current && loadDetail(current)}
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

function MissionDetail({ detail, api, action, busy, onReport, onApplied }: {
  detail: any;
  api: Api;
  action: Action;
  busy: boolean;
  onReport: () => void;
  onApplied?: () => void;
}) {
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

        <h3 className="section-subtitle">Work windows</h3>
        <WorkWindows api={api} missionId={mission.id} revision={versions[versions.length - 1].version} tz={tz}
          action={action} supervisor={mission.supervisor} busy={busy} onApplied={onApplied} />

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
                <b className="memory-score"><small>Match score</small>{fmt(m.score * 100, 0)}</b>
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

function WorkWindows({ api, missionId, revision, tz, action, supervisor, busy, onApplied }: {
  api: Api;
  missionId: string;
  revision: number;
  tz: string;
  action: Action;
  supervisor: string;
  busy: boolean;
  onApplied?: () => void;
}) {
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState("");
  const [note, setNote] = useState("");
  const [tick, setTick] = useState(0);
  const [applied, setApplied] = useState<any>(null);
  useEffect(() => {
    let live = true;
    setData(null);
    setError("");
    api(`/api/missions/${missionId}/work-windows?hours=48&step_hours=1&flex_hours=2`)
      .then((d) => live && setData(d))
      .catch((e) => live && setError(String(e?.message || e)));
    return () => {
      live = false;
    };
  }, [missionId, revision, tick]);
  const record = async (choice: "move" | "keep", start?: string, apply = false) => {
    if (!data) return;
    let result: any = null;
    const ok = await action(
      async () => {
        result = await api(`/api/missions/${missionId}/window-decisions`, {
          method: "POST",
          body: JSON.stringify({ choice, start, apply, note: note.trim() || null, snapshot: data.snapshot, supervisor }),
        });
      },
      apply
        ? "Window decision recorded and the shift moved; the Job Hazard Analysis was rebuilt for the new window."
        : choice === "move"
          ? "Window decision recorded. Reschedule the crew in your own system; this record states what the comparison showed."
          : "Window decision recorded: the planned shift stands.",
    );
    if (ok) {
      setNote("");
      setApplied(result?.applied ?? null);
      setTick((value) => value + 1);
      if (apply) onApplied?.();
    }
  };
  if (error) return <p className="body-note">Work-window comparison unavailable. {error}</p>;
  if (!data) return <p className="body-note">Comparing candidate windows…</p>;
  const verdict = data.recommendation;
  const banner =
    verdict.choice === "move"
      ? "Move the shift to " + zoned(verdict.start, tz) + " (" + data.duration_hours + " h). " + verdict.reason
      : verdict.choice === "keep"
        ? "Keep the planned window. " + verdict.reason
        : "No change is advised on this evidence. " + verdict.reason;
  const footnote =
    data.candidate_count + " windows of " + data.duration_hours + " h compared across " + data.horizon_hours + " h; " +
    data.excluded.hour_band + " start(s) outside the allowed band, " + data.excluded.overlap + " overlapping another mission, " +
    data.excluded.coverage + " too thinly forecast. " +
    (data.ensemble_available ? "Weather-model members are shown; " : "No member sample was available, so this rests on the single-run forecast; ") +
    "starts stay within " + data.constraints.flex_hours + " h of the planned " + data.constraints.planned_local_hour +
    ":00 site-time start. " + data.notes[4];
  const recorded = data.decision
    ? "Recorded " + (data.decision.choice === "move" ? "move to " + zoned(data.decision.start, tz) : "keep of the planned window") +
      " by " + data.decision.actor + " on " + zoned(data.decision.created_at, tz) +
      (data.decision.note ? " · " + data.decision.note : "") +
      (data.decision.stale ? ". The forecast has changed since; record again if the decision still holds." : "")
    : "";
  return (
    <>
      <div className={"window-verdict verdict-" + verdict.choice} role="status">
        <TriangleAlert size={16} />
        <span>{banner}</span>
      </div>
      {data.decision && <p className={"body-note" + (data.decision.stale ? " warning-text" : "")}>{recorded}</p>}
      <div className="window-actions">
        <input
          value={note}
          onChange={(event) => setNote(event.target.value)}
          placeholder="Optional note for the audit record"
          maxLength={280}
          disabled={busy}
        />
        <button type="button" className="primary" disabled={busy} onClick={() => record("keep")}>
          Record: keep the plan
        </button>
        {verdict.choice === "move" && (
          <button type="button" className="primary" disabled={busy} onClick={() => record("move", verdict.start)}>
            Record: move to the recommended window
          </button>
        )}
        {verdict.choice === "move" && (
          <button type="button" className="primary" disabled={busy} onClick={() => record("move", verdict.start, true)}>
            Record and reschedule to {zoned(verdict.start, tz)}
          </button>
        )}
        {verdict.choice === "move" &&
          data.windows
            .filter((w: any) => !w.restricted && w.start !== verdict.start && w.start !== data.planned_start)
            .slice(0, 2)
            .map((w: any) => (
              <button type="button" key={w.start} disabled={busy} onClick={() => record("move", w.start)}>
                Move to {zoned(w.start, tz)}
              </button>
            ))}
      </div>
      {applied && (
        <p className="body-note">
          Shift moved: {zoned(applied.start, tz)} to {zoned(applied.end, tz)}.{" "}
          {applied.hazards_unchanged
            ? "Hazards were unchanged by the move, so the current JHA version stands and its check was refreshed."
            : "The Job Hazard Analysis was rebuilt for the new window as version " + applied.jha_version + "."}
        </p>
      )}
      {data.candidate_count > 0 && (
      <div className="work-windows">
        {data.windows.map((w: any) => (
          <div
            key={w.start}
            className={
              "work-window" +
              (verdict.choice === "move" && w.start === verdict.start ? " recommended" : "") +
              (w.start === data.planned_start ? " planned" : "")
            }
          >
            <div className="window-head">
              <span className={"rating-dot rating-" + (w.label === "restricted" ? "high" : w.label)} />
              <strong>
                {zoned(w.start, tz)} → {new Date(w.end).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", timeZone: tz })}
              </strong>
              <span className="window-class">
                {w.label}
                {verdict.choice === "move" && w.start === verdict.start ? " · recommended" : ""}
                {w.start === data.planned_start ? " · planned" : ""}
              </span>
              {w.uncertainty?.restrict_share != null && (
                <span className="window-prob" title={w.uncertainty.members + " weather-model members; share crossing a screening limit"}>
                  {Math.round(w.uncertainty.restrict_share * 100)}% of members restrict
                </span>
              )}
            </div>
            <small>
              apparent {fmt(w.apparent_max_c)}°C · gust {fmt(w.gust_max_kmh, 0)} km/h · rain {fmt(w.rain_mm)} mm
              {w.soil_deep_max != null ? " · deep soil " + fmt(w.soil_deep_max, 2) + " m³/m³" : ""}
              {w.coverage < 1 ? " · " + Math.round(w.coverage * 100) + "% of hours forecast" : ""}
              {w.unknown.length ? " · unknown " + w.unknown.join(", ") : ""}
            </small>
            {w.drivers.length > 0 && <small className="window-drivers">Drivers: {w.drivers.join(", ")}</small>}
          </div>
        ))}
      </div>
      )}
      <p className="body-note">{footnote}</p>
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
    map.current = L.map(container.current, {
      zoomControl: false,
      trackResize: false,
      minZoom: 2,
      maxBounds: [[-85, -180], [85, 180]],
    }).setView([37.4, -96.5], 4);
    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap contributors</a>',
      maxZoom: 19,
      noWrap: true,
    }).addTo(map.current);
    L.control.zoom({ position: "bottomright" }).addTo(map.current);
    layer.current = L.layerGroup().addTo(map.current);
    const disconnectResize = observeMapResize(map.current, container.current);
    return () => {
      disconnectResize();
      map.current?.remove();
      map.current = null;
      layer.current = null;
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
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    let active = true;
    setData(null);
    setError("");
    api(`/plan-overlaps?utilities_only=${utilities}&cross_state=${crossState}&limit=60${kind ? "&kind=" + kind : ""}`)
      .then((result) => { if (active) setData(result); })
      .catch((e) => { if (active) setError(e instanceof Error ? e.message : String(e)); });
    return () => { active = false; };
  }, [utilities, crossState, kind, retry]);
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
      {error && <div className="message error panel-message" role="alert">
        <TriangleAlert size={18} /><span>Could not load plan overlaps. {error}</span>
        <button className="text-button" onClick={() => setRetry(retry + 1)}>Try again</button>
      </div>}
      {!data ? !error && (
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
                  <b className="memory-score"><small>Overlap score</small>{fmt(p.score * 100, 0)}</b>
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
                <details className="overlap-factors">
                  <summary>Score factors</summary>
                  <FactorBars factors={p.factors} />
                </details>
              </article>
            ))}
          </div>
        </>
      )}
    </section>
  );
}
