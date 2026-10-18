import { useEffect, useRef, useState } from "react";

type Gauge = {
  station: string;
  name: string;
  relationship: string;
  distance_miles: number;
  observation: {
    available: boolean; reason?: string; latest_at?: string; quality?: string;
    value_cfs?: number; age_minutes?: number; change_6h_cfs?: number | null;
  };
  receipt: { received_at: string; url: string };
  experimental: { status: string; reasons: string[]; probability: number | null; baseline_probability: number | null };
};
type RiverContext = {
  summary: string; note: string; checked_at: string; stale: boolean;
  network: { available: boolean; candidate_count?: number; note?: string };
  gauges: Gauge[];
};
const reasons: Record<string, string> = {
  model_not_promoted: "The model has not passed independent validation.",
  site_flood_applicability_unverified: "Gauge high flow has not been validated as flooding at this work location.",
  gauge_has_no_frozen_reference_and_attributes: "This gauge has no reviewed historical reference and model attributes.",
  no_response_received_by_issue: "No saved response was received before the scheduled forecast issue.",
  receipt_outside_issue_availability_window: "The saved response is outside the forecast issue's availability window.",
  approved_history_missing: "Current observations are not approved model inputs.",
  approved_169_hour_history_incomplete: "The model needs 169 consecutive hours of approved discharge history.",
  not_quiet_for_24_hours: "Recent high flow means a new-onset forecast is not applicable.",
};

export function HydrologyPanel({ api, path, onChecked }: {
  api: (path: string, options?: RequestInit) => Promise<any>;
  path: string;
  onChecked?: () => void;
}) {
  const [data, setData] = useState<RiverContext | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const mounted = useRef(true);
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);
  async function check() {
    setBusy(true);
    setError("");
    try {
      const result = await api(path);
      if (mounted.current) { setData(result); onChecked?.(); }
    } catch (e) {
      if (mounted.current) { setData(null); setError(String(e instanceof Error ? e.message : e)); }
    } finally {
      if (mounted.current) setBusy(false);
    }
  }
  return <section aria-label="River intelligence">
    <h3 className="section-subtitle">River intelligence</h3>
    <p className="body-note">USGS river observations and watershed connections. Experimental ML never changes work clearance.</p>
    <button className="text-button" onClick={check} disabled={busy}>
      {busy ? "Checking river network and discharge…" : "Check live river data"}
    </button>
    {error ? <p role="alert">River data unavailable. {error}</p> : null}
    {data ? <div aria-live="polite">
      <p className="body-note">{data.summary} Checked {new Date(data.checked_at).toLocaleString()}.</p>
      {!data.network.available ? <p className="body-note">River network coverage is incomplete or unavailable.</p> : null}
      <ul className="evidence-list">
        {data.gauges.map(g => <li key={g.station}><span>
          <b>{g.name}</b> · {g.relationship.replaceAll("_", " ")} · {g.distance_miles} miles straight-line.<br />
          {g.observation.available
            ? `${g.observation.value_cfs?.toLocaleString()} ft³/s, ${g.observation.quality?.toLowerCase()}. Observed ${new Date(g.observation.latest_at!).toLocaleString()}. ${g.observation.age_minutes} minutes old.`
            : `Discharge unavailable or stale. ${g.observation.reason || "Check the source before using this reading."}`}
          {g.observation.available && g.observation.change_6h_cfs != null
            ? <span> Six-hour discharge change: {g.observation.change_6h_cfs > 0 ? "+" : ""}{g.observation.change_6h_cfs.toLocaleString()} ft³/s.</span> : null}
          <br /><a href={`https://waterdata.usgs.gov/monitoring-location/USGS-${g.station}/`} target="_blank" rel="noopener noreferrer">USGS gauge {g.station}</a>
          <details><summary>Experimental forecast withheld from operational decisions</summary>
            <ul>{g.experimental.reasons.map(reason => <li key={reason}>{reasons[reason] || "An input-contract check failed. Review the source data."}</li>)}</ul>
            {g.experimental.status === "experimental_gauge_score" ? <p>
              Unvalidated gauge-onset research score: {((g.experimental.probability ?? 0) * 100).toFixed(1)}%. Physical comparator: {((g.experimental.baseline_probability ?? 0) * 100).toFixed(1)}%. These are not site-flood probabilities.
            </p> : null}
            <small>Response received {new Date(g.receipt.received_at).toLocaleString()}. Receipt time does not establish first publication.</small>
          </details>
        </span></li>)}
      </ul>
      <p className="body-note">{data.network.note} {data.note}</p>
    </div> : null}
  </section>;
}
