import React, { useEffect, useRef, useState, useCallback } from "react";
import { createRoot } from "react-dom/client";
import L from "leaflet";
import {
  Activity,
  ArrowDown,
  ArrowLeft,
  ArrowRight,
  Bell,
  Building2,
  Check,
  CheckCircle2,
  ChevronRight,
  CloudRain,
  Compass,
  Download,
  ExternalLink,
  FileText,
  Globe2,
  Layers,
  MapPin,
  Plus,
  Radio,
  RefreshCw,
  Search,
  Settings2,
  Shield,
  Thermometer,
  TriangleAlert,
  Wind,
  X,
  Zap,
} from "lucide-react";
import { ClipboardList, GitCompare } from "lucide-react";
import "leaflet/dist/leaflet.css";
import "./styles.css";
import { KindBadge, MissionsView, OverlapsView } from "./missions";

type Site = {
  id: string;
  name: string;
  state: string;
  county: string;
  latitude: number;
  longitude: number;
  capacity_mw: number;
  technology: string;
  operator: string;
  timezone: string;
  source: string;
  index: Index | null;
  assessed_at: string | null;
  stale: boolean;
  lightning: any;
};
type Index = {
  score: number | null;
  coverage: number;
  score_max?: number;
  missing?: string[];
  complete: boolean;
  label: string;
  components: {
    name: string;
    severity: number | null;
    weight: number;
    points: number | null;
    detail: string;
  }[];
};
type Detail = {
  site: Site;
  index: Index;
  weather: any;
  alerts: any;
  model: any;
  lightning: any;
  reports: any[];
  floodplain: any;
  nearby?: { flood: any[]; wildfire: any[] };
  history: any[];
  assessed_at: string;
  stale?: boolean;
};
const CATEGORIES = ["heat", "storm", "flood", "lightning", "wildfire", "wildlife", "unstable_terrain", "equipment_obstruction", "other"];
const hazardName = (v: string) => v.replaceAll("_", " ");
const fmt = (v: number | null | undefined, digits = 1) =>
  v == null ? "—" : v.toFixed(digits);
const time = (v: string | null) =>
  v
    ? new Date(v).toLocaleString([], {
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      })
    : "Not fetched";
const color = (s: Site) =>
  s.lightning?.live?.active
    ? "#ee786b"
    : s.stale || !s.index || s.index.score == null
      ? "#9da7a4"
      : s.index.score >= 6
        ? "#ee786b"
        : s.index.score >= 3
          ? "#eac26b"
          : "#4a7958";
const riskLabel = (s: Site) =>
  s.stale
    ? "Stale"
    : !s.index
      ? "Unassessed"
      : s.index.score == null
        ? "Unknown"
        : s.index.score >= 6
          ? "High"
          : s.index.score >= 3
            ? "Elevated"
            : "Lower";
let supervisorToken = sessionStorage.getItem("planit-token") || "";
async function api(path: string, options: RequestInit = {}) {
  const response = await fetch("/api" + path, {
    ...options,
    headers: {
      ...(options.body instanceof FormData
        ? {}
        : { "Content-Type": "application/json" }),
      ...(supervisorToken
        ? { Authorization: "Bearer " + supervisorToken }
        : {}),
      ...options.headers,
    },
  });
  if (!response.ok) {
    const error = await response
      .json()
      .catch(() => ({ detail: response.statusText }));
    throw new Error(
      typeof error.detail === "string"
        ? error.detail
        : JSON.stringify(error.detail),
    );
  }
  return response.json();
}
function SiteMap({
  sites,
  selected,
  onSelect,
  reports,
}: {
  sites: Site[];
  selected: string | null;
  onSelect: (s: Site) => void;
  reports: any[];
}) {
  const container = useRef<HTMLDivElement>(null),
    map = useRef<L.Map | null>(null),
    markers = useRef<L.LayerGroup | null>(null),
    fitted = useRef(false);
  const [tileError, setTileError] = useState(false);
  useEffect(() => {
    if (!container.current) return;
    map.current = L.map(container.current, {
      zoomControl: false,
      attributionControl: true,
    }).setView([37.4, -96.5], 4);
    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution:
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap contributors</a>',
      maxZoom: 19,
    })
      .on("tileerror", () => setTileError(true))
      .addTo(map.current);
    L.control.zoom({ position: "bottomright" }).addTo(map.current);
    markers.current = L.layerGroup().addTo(map.current);
    const observer = new ResizeObserver(() => map.current?.invalidateSize());
    observer.observe(container.current);
    return () => {
      observer.disconnect();
      map.current?.remove();
      map.current = null;
    };
  }, []);
  useEffect(() => {
    if (!markers.current) return;
    if (sites.length && !fitted.current) {
      map.current?.fitBounds(
        L.latLngBounds(
          sites.map((s) => [s.latitude, s.longitude] as [number, number]),
        ),
        { padding: [30, 35] },
      );
      fitted.current = true;
    }
    markers.current.clearLayers();
    sites.forEach((s) => {
      const marker = L.circleMarker([s.latitude, s.longitude], {
        radius: s.id === selected ? 10 : 5.5,
        fillColor: color(s),
        color: s.id === selected ? "#fff" : color(s),
        weight: s.id === selected ? 2 : 1,
        fillOpacity: 0.95,
      });
      const tip = document.createElement("span");
      tip.textContent =
        s.name +
        " · " +
        (s.stale
          ? "stale"
          : s.index
            ? fmt(s.index.score) +
              "/10 · " +
              Math.round(s.index.coverage * 100) +
              "% inputs"
            : "unassessed");
      marker
        .bindTooltip(tip)
        .on("click", () => onSelect(s))
        .addTo(markers.current!);
    });
    reports
      .filter((r) => r.status !== "rejected")
      .forEach((r) => {
        const marker = L.marker([r.latitude, r.longitude], {
          icon: L.divIcon({
            className: "report-pin",
            html: "!",
            iconSize: [20, 20],
          }),
        });
        const tip = document.createElement("span");
        tip.textContent = `Crew report · ${(r.triage?.manual_categories || [r.classification]).map(hazardName).join(", ")} · ${r.status}`;
        marker.bindTooltip(tip).addTo(markers.current!);
      });
  }, [sites, selected, reports, onSelect]);
  useEffect(() => {
    const s = sites.find((s) => s.id === selected);
    if (s) map.current?.flyTo([s.latitude, s.longitude], 7, { duration: 1 });
  }, [selected]);
  return (
    <>
      <div
        className="map"
        ref={container}
        aria-label="Interactive map of EIA power plants"
      />
      {tileError && (
        <div className="tile-error">
          Basemap unavailable. Facility markers and the facility list remain
          available.
        </div>
      )}
    </>
  );
}
function App() {
  const [sites, setSites] = useState<Site[]>([]),
    [selected, setSelected] = useState<string | null>(null),
    [detail, setDetail] = useState<Detail | null>(null);
  const [view, setView] = useState("overview"),
    [search, setSearch] = useState(""),
    [loading, setLoading] = useState(false),
    [error, setError] = useState(""),
    [notice, setNotice] = useState("");
  const [modal, setModal] = useState<"report" | "settings" | null>(null),
    [token, setToken] = useState(supervisorToken),
    [evaluation, setEvaluation] = useState<any>(null),
    [briefings, setBriefings] = useState<any[]>([]),
    [refresh, setRefresh] = useState<any>({});
  const [busy, setBusy] = useState(false),
    [clock, setClock] = useState(Date.now()),
    [revision, setRevision] = useState(0);
  const [reportHazard, setReportHazard] = useState("all"),
    [reportQueue, setReportQueue] = useState("all");
  const selectedRef = useRef(selected);
  selectedRef.current = selected;
  async function loadSites() {
    try {
      const data = await api("/sites");
      setSites(data.sites);
      setRefresh(data.refresh);
    } catch (e) {
      setError(String(e));
    }
  }
  async function loadDetail(id: string) {
    setLoading(true);
    setError("");
    try {
      const d = await api("/sites/" + id);
      if (selectedRef.current === id) setDetail(d);
    } catch (e) {
      setError(String(e));
    } finally {
      if (selectedRef.current === id) setLoading(false);
    }
  }
  useEffect(() => {
    loadSites();
    const interval = setInterval(loadSites, 15000);
    const tick = setInterval(() => setClock(Date.now()), 1000);
    return () => {
      clearInterval(interval);
      clearInterval(tick);
    };
  }, []);
  useEffect(() => {
    setDetail(null);
    setBriefings([]);
    if (selected) {
      loadDetail(selected);
      api(`/sites/${selected}/briefings`)
        .then(setBriefings)
        .catch((e) => setError(String(e)));
    }
    const timer = setInterval(() => {
      if (selected) loadDetail(selected);
    }, 60000);
    return () => clearInterval(timer);
  }, [selected]);
  useEffect(() => {
    if (view === "model" && !evaluation)
      api("/evaluation")
        .then(setEvaluation)
        .catch((e) => setError(String(e)));
  }, [view]);
  useEffect(() => {
    if (!modal) return;
    const opener = document.activeElement as HTMLElement;
    const dialog = document.querySelector<HTMLElement>("[role=dialog]");
    const controls = () =>
      Array.from(
        dialog?.querySelectorAll<HTMLElement>(
          "button:not(:disabled),input,select,textarea,[href]",
        ) || [],
      );
    controls()[0]?.focus();
    const handler = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !busy) setModal(null);
      if (event.key === "Tab") {
        const items = controls();
        const first = items[0],
          last = items[items.length - 1];
        if (event.shiftKey && document.activeElement === first) {
          event.preventDefault();
          last?.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault();
          first?.focus();
        }
      }
    };
    document.addEventListener("keydown", handler);
    return () => {
      document.removeEventListener("keydown", handler);
      opener?.focus();
    };
  }, [modal, busy]);
  const choose = useCallback((s: Site) => {
    setSelected(s.id);
    setView("overview");
  }, []);
  async function action(fn: () => Promise<any>, message: string) {
    setBusy(true);
    setError("");
    try {
      await fn();
      setNotice(message);
      setRevision((r) => r + 1);
      if (selected) await loadDetail(selected);
      await loadSites();
      return true;
    } catch (e) {
      setError(String(e));
      return false;
    } finally {
      setBusy(false);
    }
  }
  const filtered = sites.filter((s) =>
    (s.name + " " + s.state + " " + s.county)
      .toLowerCase()
      .includes(search.toLowerCase()),
  );
  const assessed = sites.filter((s) => s.index && !s.stale).length;
  const elevated = sites.filter(
    (s) => !s.stale && s.index && s.index.score != null && s.index.score >= 3,
  ).length;
  const liveStops = sites.filter((s) => s.lightning?.live?.active).length;
  const unknown = sites.filter(
    (s) => s.stale || !s.index || s.index.coverage < 1,
  ).length;
  const selectedSite = sites.find((s) => s.id === selected);
  const visibleReports = (detail?.reports || []).filter(r => {
    const provider = r.triage?.provider;
    const categories = [...(r.triage?.manual_categories || [r.classification]), ...(provider?.categories || [])];
    return (reportHazard === "all" || categories.includes(reportHazard)) &&
      (reportQueue === "all" || r.status === "pending" &&
        (reportQueue === "manual" ? provider?.status !== "classified" : provider?.review_queue === reportQueue));
  }).sort((a, b) => {
    // Pending first: urgent, then clarification or unscreened, then standard; oldest waits least. Closed reports newest first.
    const rank = (r: any) => {
      if (r.status !== "pending") return 9;
      const p = r.triage?.provider;
      return p?.status !== "classified" ? 1 : ({ urgent: 0, clarification: 1, standard: 2 } as any)[p.review_queue] ?? 1;
    };
    return rank(a) - rank(b) || (rank(a) < 9 ? a.created_at.localeCompare(b.created_at) : b.created_at.localeCompare(a.created_at));
  });
  const remaining = (state: any) =>
    Math.max(
      0,
      Math.ceil((new Date(state?.review_after || 0).getTime() - clock) / 1000),
    );
  const countdown = (state: any) => {
    const sec = remaining(state);
    return `${Math.floor(sec / 60)
      .toString()
      .padStart(2, "0")}:${(sec % 60).toString().padStart(2, "0")}`;
  };
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a
          className="brand"
          href="#"
          onClick={(e) => {
            e.preventDefault();
            setView("overview");
          }}
        >
          <span className="brand-symbol">
            <Compass size={25} />
          </span>
          <span>
            planit<span className="brand-control">CONTROL</span>
          </span>
        </a>
        <div className="workspace">
          <span className="workspace-logo">P</span>
          <div>
            Field operations<small>United States portfolio</small>
          </div>
          <span className="workspace-dot" />
        </div>
        <div className="nav-label">WORKSPACE</div>
        <nav aria-label="Main navigation">
          <button
            className={view === "overview" ? "nav-item active" : "nav-item"}
            onClick={() => setView("overview")}
          >
            <Layers size={18} />
            Overview<span className="nav-count">60</span>
          </button>
          <button
            className={view === "briefings" ? "nav-item active" : "nav-item"}
            onClick={() => setView("briefings")}
          >
            <FileText size={18} />
            Safety briefings
          </button>
          <button
            className={view === "missions" ? "nav-item active" : "nav-item"}
            onClick={() => setView("missions")}
          >
            <ClipboardList size={18} />
            Missions & JHA
          </button>
          <button
            className={view === "overlaps" ? "nav-item active" : "nav-item"}
            onClick={() => setView("overlaps")}
          >
            <GitCompare size={18} />
            Plan overlaps
          </button>
          <button
            className={view === "model" ? "nav-item active" : "nav-item"}
            onClick={() => setView("model")}
          >
            <Activity size={18} />
            Model intelligence<span className="tiny-tag">ML</span>
          </button>
        </nav>
        <div className="sidebar-bottom">
          <div className="system-status">
            <span className="status-dot" />
            <div>
              Source-aware monitoring
              <small>
                {refresh.running
                  ? `Refreshing ${refresh.completed} / ${refresh.total}`
                  : "Refresh every 30 minutes"}
              </small>
            </div>
          </div>
          <button className="nav-item" onClick={() => setModal("settings")}>
            <Settings2 size={18} />
            Supervisor access
          </button>
          <div className="profile">
            <div className="avatar">FS</div>
            <div>
              Field supervisor<small>Local workspace</small>
            </div>
            <Shield size={16} />
          </div>
        </div>
      </aside>
      <main>
        <header className="topbar">
          <div className="breadcrumb">
            Workspace <ChevronRight size={13} />
            <span>
              {view === "overview"
                ? "Operations overview"
                : view === "model"
                  ? "Model intelligence"
                  : view === "missions"
                    ? "Missions & JHA"
                    : view === "overlaps"
                      ? "Plan overlaps"
                      : "Safety briefings"}
            </span>
          </div>
          <div className="topbar-right">
            <span className="date-label">
              {new Date(clock).toLocaleDateString([], {
                month: "short",
                day: "numeric",
                year: "numeric",
              })}
            </span>
            <span className="vertical-line" />
            <span className="research-badge">
              <span />
              RESEARCH MVP
            </span>
            <button
              className="icon-button"
              aria-label="Open supervisor access settings"
              onClick={() => setModal("settings")}
            >
              <Settings2 size={18} />
            </button>
          </div>
        </header>
        <div className="page-content">
          <div className="page-heading">
            <div>
              <div className="eyebrow">
                <span />
                ENVIRONMENTAL INTELLIGENCE
              </div>
              <h1>
                {view === "overview"
                  ? "Operations overview"
                  : view === "model"
                    ? "Model evaluation"
                    : view === "missions"
                      ? "Missions and job hazard analysis"
                      : view === "overlaps"
                        ? "Plan overlaps"
                        : "Safety briefings"}
              </h1>
              <p>
                {view === "overview"
                  ? "Environmental screening for each monitored EIA power plant. Select a site for its conditions, alerts and crew reports."
                  : view === "model"
                    ? "Real historical outcomes, measured performance, and visible limitations."
                    : view === "missions"
                      ? "Each mission's hazard analysis updates when conditions or crew reports change, and keeps every version."
                      : view === "overlaps"
                        ? "Compare utilities' public construction plans and flag work that is close in space or time."
                        : "Site-specific safety briefings, generated at 6 PM in each facility’s local time."}
              </p>
            </div>
            <button
              className="button primary"
              disabled={busy}
              onClick={() =>
                selected
                  ? setModal("report")
                  : setError(
                      "Select a facility on the map or in the list to submit a report.",
                    )
              }
            >
              <Plus size={16} />
              New crew report
            </button>
          </div>
          {error && (
            <div className="message error" role="alert">
              <TriangleAlert size={18} />
              <span>{error}</span>
              <button aria-label="Dismiss error" onClick={() => setError("")}>
                <X size={15} />
              </button>
            </div>
          )}
          {notice && (
            <div className="message success" role="status">
              <CheckCircle2 size={18} />
              <span>{notice}</span>
              <button
                aria-label="Dismiss notification"
                onClick={() => setNotice("")}
              >
                <X size={15} />
              </button>
            </div>
          )}
          {liveStops > 0 && (
            <div className="stop-banner">
              <Zap size={20} />
              <strong>
                STOP WORK · {liveStops} facilities with qualifying recorded
                lightning
              </strong>
              <span>
                Follow shelter procedures. Timer expiry requires supervisor
                review.
              </span>
            </div>
          )}
          {view === "overview" && (
            <>
              <div className="metrics-grid">
                <Metric
                  title="Monitored facilities"
                  value={String(sites.length || "—")}
                  label="EIA-860M · 33 states"
                  icon={<Building2 size={20} />}
                  accent="green"
                />
                <Metric
                  title="Elevated screening index"
                  value={String(elevated)}
                  label="Score ≥ 3 · partial inputs possible"
                  icon={<TriangleAlert size={20} />}
                  accent="amber"
                />
                <Metric
                  title="Recorded lightning holds"
                  value={String(liveStops)}
                  label="Live records only · feed may be unavailable"
                  icon={<Zap size={20} />}
                  accent="red"
                />
                <Metric
                  title="Incomplete assessments"
                  value={String(unknown)}
                  label={`${assessed} assessed · missing inputs stay unknown`}
                  icon={<Radio size={20} />}
                  accent="gray"
                />
              </div>
              <div className="operations-grid">
                <section className="panel map-panel">
                  <div className="panel-header">
                    <div>
                      <h2>
                        Portfolio map{" "}
                        <span className="small-count">{sites.length}</span>
                      </h2>
                      <p>Operational environmental screening · United States</p>
                    </div>
                    <button
                      className="text-button"
                      disabled={busy || refresh.running}
                      onClick={() =>
                        action(
                          () => api("/refresh", { method: "POST" }),
                          "Environmental refresh requested.",
                        )
                      }
                    >
                      <RefreshCw
                        size={14}
                        className={refresh.running ? "spin" : ""}
                      />
                      {refresh.running ? "Refreshing" : "Refresh feeds"}
                    </button>
                  </div>
                  <div className="map-container">
                    <SiteMap
                      sites={sites}
                      selected={selected}
                      onSelect={choose}
                      reports={visibleReports}
                    />
                    <div className="map-caption">
                      <span className="status-dot" />
                      60 real EIA facilities
                      <span className="caption-separator" />
                      Select a marker to inspect
                    </div>
                    <div className="map-scale">
                      <span>
                        <i style={{ background: "#4a7958" }} />
                        Lower
                      </span>
                      <span>
                        <i style={{ background: "#eac26b" }} />
                        Elevated
                      </span>
                      <span>
                        <i style={{ background: "#ee786b" }} />
                        High / hold
                      </span>
                      <span>
                        <i style={{ background: "#9da7a4" }} />
                        Unknown
                      </span>
                    </div>
                  </div>
                  <div className="map-footer">
                    <Globe2 size={14} />
                    <span>Inventory: December 2024 · coordinates from EIA</span>
                    <span className="ml-note">
                      Partial scores retain fixed weights
                    </span>
                  </div>
                </section>
                <section className="panel facility-panel">
                  <div className="panel-header">
                    <div>
                      <h2>Facility watchlist</h2>
                      <p>Ranked by the highest score unknown inputs could reach</p>
                    </div>
                    <ArrowDown size={17} />
                  </div>
                  <div className="search-box">
                    <Search size={15} />
                    <input
                      aria-label="Search facilities"
                      placeholder="Search name, county, state…"
                      value={search}
                      onChange={(e) => setSearch(e.target.value)}
                    />
                    <span>{filtered.length}</span>
                  </div>
                  <div className="list-label">
                    <span>FACILITY / LOCATION</span>
                    <span>INDEX</span>
                  </div>
                  <div className="facility-list">
                    {filtered.map((s) => (
                      <button
                        key={s.id}
                        className={`facility-row ${selected === s.id ? "selected" : ""}`}
                        onClick={() => choose(s)}
                      >
                        <span
                          className="facility-dot"
                          style={{ background: color(s) }}
                        />
                        <span className="facility-name">
                          {s.name}
                          <small>
                            {s.county}, {s.state}{" "}
                            <span>
                              ·{" "}
                              {s.stale
                                ? "stale"
                                : s.index
                                  ? Math.round(s.index.coverage * 100) +
                                    "% inputs" +
                                    (s.index.coverage < 1 && s.index.score_max != null
                                      ? " · up to " + fmt(s.index.score_max)
                                      : "")
                                  : "unassessed"}
                            </span>
                          </small>
                        </span>
                        <span
                          className="score-pill"
                          style={{ color: color(s) }}
                        >
                          {s.stale ? "—" : fmt(s.index?.score)}
                          <small>{riskLabel(s)}</small>
                        </span>
                        <ChevronRight size={14} />
                      </button>
                    ))}
                    {!filtered.length && (
                      <p className="empty">No facilities match your search.</p>
                    )}
                  </div>
                  <div className="watchlist-footer">
                    <Shield size={13} />A lower partial score does not establish
                    safety.
                  </div>
                </section>
              </div>
              {!selected && (
                <div className="below-map">
                  <div className="intro-card">
                    <div className="intro-icon">
                      <MapPin size={24} />
                    </div>
                    <div>
                      <h3>Select a facility</h3>
                      <p>
                        Pick a map marker or a watchlist row to see forecast
                        conditions, index contributions, alerts and crew reports.
                      </p>
                    </div>
                    <ArrowRight size={22} />
                  </div>
                  <div className="source-card">
                    <span className="eyebrow">BUILT ON PUBLIC DATA</span>
                    <div>
                      EIA <span>·</span> NOAA / NWS <span>·</span> Open-Meteo{" "}
                      <span>·</span> FEMA
                    </div>
                    <p>
                      Each assessment shows its source timestamps and which
                      feeds were unavailable.
                    </p>
                  </div>
                </div>
              )}
              {selected && (
                <section className="panel detail-panel">
                  <div className="panel-header">
                    <div>
                      <div className="eyebrow">
                        FACILITY INTELLIGENCE · EIA {selected}
                      </div>
                      <h2 className="site-heading">{selectedSite?.name}</h2>
                      <p>
                        {selectedSite?.county}, {selectedSite?.state} ·{" "}
                        {selectedSite?.technology} ·{" "}
                        {fmt(selectedSite?.capacity_mw, 0)} MW ·{" "}
                        {selectedSite?.timezone}
                      </p>
                    </div>
                    <button
                      className="icon-button"
                      aria-label="Close site details"
                      onClick={() => setSelected(null)}
                    >
                      <X size={19} />
                    </button>
                  </div>
                  {loading && !detail ? (
                    <div className="loading">
                      <RefreshCw className="spin" size={22} />
                      Fetching source conditions, alerts, and archived forecast…
                    </div>
                  ) : detail ? (
                    <>
                      <div className="detail-safety">
                        <Shield size={16} />
                        <span>
                          Supervisor decision support. Forecast estimates are
                          not site measurements. No model or timer certifies
                          safe work.
                        </span>
                        <small>Assessed {time(detail.assessed_at)}</small>
                      </div>
                      {remaining(detail.lightning.live) > 0 && (
                        <div className="stop-banner">
                          <Zap />
                          <strong>
                            LIVE RECORD · STOP WORK{" "}
                            {countdown(detail.lightning.live)}
                          </strong>
                          <span>{detail.lightning.live.instruction}</span>
                        </div>
                      )}
                      {remaining(detail.lightning.replay) > 0 && (
                        <div className="replay-banner">
                          <Zap size={18} />
                          <strong>
                            REPLAY SCENARIO ·{" "}
                            {countdown(detail.lightning.replay)}
                          </strong>
                          <span>
                            Simulated strike. This timer is separate from live
                            records.
                          </span>
                        </div>
                      )}
                      <div className="detail-grid">
                        <div className="conditions-column">
                          <h3>
                            Environmental outlook <span>Next 24 hours</span>
                          </h3>
                          <div className="condition-cards">
                            <Condition
                              icon={<Thermometer />}
                              label="Apparent high"
                              value={
                                fmt(detail.weather.summary.apparent_max_c) +
                                "°C"
                              }
                              note="Heat screening · not WBGT"
                            />
                            <Condition
                              icon={<CloudRain />}
                              label="Forecast rainfall"
                              value={
                                fmt(detail.weather.summary.rain_24h_mm) + " mm"
                              }
                              note={`Prior 72h: ${fmt(detail.weather.summary.rain_72h_mm)} mm (modeled)`}
                            />
                            <Condition
                              icon={<Wind />}
                              label="Maximum gust"
                              value={
                                fmt(detail.weather.summary.gust_max_kmh) +
                                " km/h"
                              }
                              note={`Soil: ${fmt(detail.weather.summary.soil_moisture, 3)} m³/m³`}
                            />
                          </div>
                          <div className="feed-status">
                            <span
                              className={
                                detail.weather.feed.available
                                  ? "available"
                                  : "unavailable"
                              }
                            />
                            <div>
                              <strong>
                                {detail.weather.feed.available
                                  ? "Forecast feed available"
                                  : "Forecast feed unavailable"}
                              </strong>
                              <small>
                                Open-Meteo · fetched{" "}
                                {time(detail.weather.feed.fetched_at)}
                                {detail.weather.current
                                  ? " · modeled current " +
                                    time(detail.weather.current.time)
                                  : ""}
                              </small>
                              {detail.weather.feed.error && (
                                <small>{detail.weather.feed.error}</small>
                              )}
                            </div>
                          </div>
                          <h3 className="section-subtitle">
                            Weather advisories
                          </h3>
                          {detail.alerts.items === null ? (
                            <div className="inline-note warning">
                              NWS feed unavailable. Advisory status is unknown.
                            </div>
                          ) : detail.alerts.items.length ? (
                            detail.alerts.items.map((a: any) => (
                              <div className="alert-card" key={a.id}>
                                <TriangleAlert size={17} />
                                <div>
                                  <strong>{a.event}</strong>
                                  <p>{a.headline}</p>
                                  <details>
                                    <summary>Read NWS instructions</summary>
                                    <p>{a.instruction || a.description}</p>
                                  </details>
                                  <small>
                                    Issued {time(a.sent)} · expires{" "}
                                    {time(a.expires)}
                                  </small>
                                </div>
                              </div>
                            ))
                          ) : (
                            <div className="inline-note">
                              <CheckCircle2 size={15} />
                              No active NWS advisories returned for this point.
                              This is not clearance.
                            </div>
                          )}
                          <small className="source-line">
                            NWS checked {time(detail.alerts.feed.fetched_at)}
                          </small>
                          <h3 className="section-subtitle">
                            Flood exposure & other hazards
                          </h3>
                          <p className="body-note">
                            {detail.floodplain.feed.available
                              ? detail.floodplain.zones.length
                                ? `FEMA mapped zones: ${detail.floodplain.zones.map((z: any) => z.FLD_ZONE).join(", ")}. ${detail.floodplain.note}`
                                : detail.floodplain.note
                              : "FEMA floodplain lookup unavailable. Static flood exposure is unknown."}
                          </p>
                          {detail.nearby && (
                            <ul className="evidence-list nearby-evidence">
                              {[...detail.nearby.flood, ...detail.nearby.wildfire].map((x: any) => (
                                <li key={x.id}>
                                  <KindBadge kind={x.kind} />
                                  <span>
                                    <b>{x.source}</b> {x.summary}
                                    {x.rating && <em> → {x.rating}</em>}
                                    {!x.counts && <em> · not counted</em>}
                                  </span>
                                </li>
                              ))}
                            </ul>
                          )}
                          <p className="body-note">
                            River gauges and wildfire incidents are measured from the
                            facility point and rate mission JHAs; they are not part of
                            the weighted index. Wildlife: crew sightings only.
                          </p>
                        </div>
                        <div className="contribution-column">
                          <div className="risk-summary">
                            <div>
                              <span className="eyebrow">OPERATIONAL INDEX</span>
                              <strong>
                                {fmt(detail.index.score)}
                                <small>/ 10</small>
                              </strong>
                            </div>
                            <div className="coverage-badge">
                              {Math.round(detail.index.coverage * 100)}%
                              <small>input coverage</small>
                            </div>
                          </div>
                          <p className="body-note">
                            {detail.index.label}. Missing inputs are not zero
                            risk.
                            {detail.index.missing?.length
                              ? ` Unknown: ${detail.index.missing.join(", ")}. With those at full severity the score could reach ${fmt(detail.index.score_max)}; the portfolio ranks sites by that upper bound.`
                              : ""}
                          </p>
                          <div className="contributions">
                            {detail.index.components.map((c) => (
                              <div className="contribution" key={c.name}>
                                <div>
                                  <span>{c.name}</span>
                                  <strong>
                                    {fmt(c.points, 2)}{" "}
                                    <small>
                                      / {(c.weight * 10).toFixed(1)}
                                    </small>
                                  </strong>
                                </div>
                                <div className="bar-track">
                                  <span
                                    style={{
                                      width: `${(c.severity || 0) * 100}%`,
                                      background:
                                        c.severity == null
                                          ? "#bbb"
                                          : c.severity > 0.6
                                            ? "#ce8064"
                                            : "#6b8970",
                                    }}
                                  />
                                </div>
                                <small>
                                  {c.severity == null ? "Unavailable · " : ""}
                                  {c.detail}
                                </small>
                              </div>
                            ))}
                          </div>
                          <button
                            className="button secondary full"
                            disabled={busy}
                            onClick={() =>
                              action(
                                () =>
                                  api(`/sites/${selected}/lightning-replay`, {
                                    method: "POST",
                                  }),
                                "Labeled replay strike recorded. A second click resets the 30-minute timer.",
                              )
                            }
                          >
                            <Zap size={15} />
                            Replay lightning / reset timer
                          </button>
                          <small className="source-line">
                            Live strike feed requires an authenticated provider.
                            No recorded strikes does not mean no lightning.
                          </small>
                        </div>
                      </div>
                      <div className="model-strip">
                        <div className="model-strip-title">
                          <Activity size={18} />
                          <h3>Explainable county-event forecast</h3>
                          <span className="tiny-tag">EXPERIMENTAL</span>
                        </div>
                        {detail.model.available ? (
                          <>
                            <p>
                              {detail.model.target_window} · ≥48h lead ·{" "}
                              {detail.model.county} · county representative forecast
                            </p>
                            <div className="prediction-grid">
                              {Object.entries(detail.model.hazards).map(
                                ([hazard, value]) => {
                                  const p: any = value;
                                  return (
                                    <div key={hazard} className="prediction">
                                      <span>
                                        {hazard === "flood"
                                          ? "Flood / flash flood"
                                          : "Severe storm / wind"}
                                      </span>
                                      <strong>
                                        {fmt(p.probability * 100)}
                                        <small>
                                          % county-event probability
                                        </small>
                                      </strong>
                                      <small>
                                        {p.model} · threshold{" "}
                                        {fmt(p.threshold * 100)}%
                                      </small>
                                      <small>SHAP baseline {fmt(p.explanation_base * 100, 2)}% · {!p.warning_supported ? "warning threshold unsupported" : p.elevated ? "above warning threshold" : "below warning threshold"}</small>
                                      {!p.warning_supported && <p className="inline-note warning">No validation threshold met the warning budget and recall floor; automatic warning withheld.</p>}
                                      <div className="explanation-list">
                                        {p.explanations
                                          .slice(0, 3)
                                          .map((e: any) => (
                                            <div key={e.feature}>
                                              <span>
                                                {e.feature.replaceAll("_", " ")}
                                              </span>
                                              <b>
                                                {e.probability_delta >= 0
                                                  ? "+"
                                                  : ""}
                                                {fmt(
                                                  e.probability_delta * 100,
                                                  2,
                                                )}{" "}
                                                pp
                                              </b>
                                            </div>
                                          ))}
                                      </div>
                                    </div>
                                  );
                                },
                              )}
                            </div>
                            {Object.keys(detail.model.hazards).length < 2 && <p className="inline-note warning">The other hazard model is unsupported in this county. Its probability and the combined ML index component remain unknown.</p>}
                            <p className="body-note">
                              {detail.model.explanation_method} County reporting
                              is incomplete. Estimates are not site-incident or
                              injury probabilities.
                            </p>
                            <div className="inline-note warning">
                              Warnings remain experimental. Review false alerts, missed events, calibration and geographic tests in Model intelligence.
                            </div>
                          </>
                        ) : (
                          <div className="inline-note warning">
                            {detail.model.reason}
                          </div>
                        )}
                      </div>
                      <div className="reports-section">
                        <div className="section-title">
                          <div>
                            <h3>Crew incident timeline</h3>
                            <p>
                              Geolocated reports · pending reports do not affect
                              the index
                            </p>
                          </div>
                          <button
                            className="button secondary"
                            onClick={() => setModal("report")}
                          >
                            <Plus size={15} />
                            Submit report
                          </button>
                        </div>
                        <div className="report-filters">
                          <label className="field">Hazard label or Jev suggestion
                            <select value={reportHazard} onChange={e => setReportHazard(e.target.value)}>
                              <option value="all">All hazards</option>
                              {CATEGORIES.map(c => <option key={c} value={c}>{hazardName(c)}</option>)}
                            </select>
                          </label>
                          <label className="field">Pending review queue
                            <select value={reportQueue} onChange={e => setReportQueue(e.target.value)}>
                              <option value="all">All reports and statuses</option>
                              <option value="urgent">Jev suggests urgent review</option>
                              <option value="clarification">Jev requests clarification / uncertain</option>
                              <option value="manual">Manual review, Jev unavailable</option>
                            </select>
                          </label>
                        </div>
                        <p className="body-note" aria-live="polite">{visibleReports.length} of {detail.reports.length} reports match. Filters also apply to report map pins; rejected reports have no pin. Model suggestions remain unverified.</p>
                        {visibleReports.length ? (
                          visibleReports.map((r) => (
                            <article className="report-card" key={r.id}>
                              {r.image && (
                                <a
                                  href={"/uploads/" + r.image}
                                  target="_blank"
                                  rel="noreferrer"
                                >
                                  <img
                                    src={"/uploads/" + r.image}
                                    alt={`Crew photo: ${r.classification} hazard`}
                                  />
                                </a>
                              )}
                              <div>
                                <div className="report-meta">
                                  <span className="tiny-tag">
                                    {(r.triage?.manual_categories || [r.classification]).map(hazardName).join(" · ")}
                                  </span>
                                  <span className={"review-status " + r.status}>
                                    {r.status}
                                  </span>
                                  <small>{time(r.created_at)}</small>
                                </div>
                                <p>{r.description}</p>
                                <ReportTriage report={r} busy={busy} onSave={(categories) => action(() => api(`/reports/${r.id}/triage`, {method: "PATCH", body: JSON.stringify({categories})}), "Manual classifications saved. Report review status is unchanged.")} />
                                <small>
                                  <MapPin size={11} />
                                  {r.latitude.toFixed(4)},{" "}
                                  {r.longitude.toFixed(4)}
                                </small>
                              </div>
                              <div className="review-buttons">
                                {r.status === "pending" && (
                                  <>
                                    <button
                                      disabled={busy}
                                      onClick={() =>
                                        action(
                                          () =>
                                            api("/reports/" + r.id, {
                                              method: "PATCH",
                                              body: JSON.stringify({
                                                status: "verified",
                                              }),
                                            }),
                                          "Report verified. Screening index updated.",
                                        )
                                      }
                                    >
                                      <Check size={13} />
                                      Verify
                                    </button>
                                    <button
                                      disabled={busy}
                                      onClick={() =>
                                        action(
                                          () =>
                                            api("/reports/" + r.id, {
                                              method: "PATCH",
                                              body: JSON.stringify({
                                                status: "rejected",
                                              }),
                                            }),
                                          "Report rejected.",
                                        )
                                      }
                                    >
                                      <X size={13} />
                                      Reject
                                    </button>
                                  </>
                                )}
                                {r.status === "verified" && (
                                  <button
                                    disabled={busy}
                                    onClick={() =>
                                      action(
                                        () =>
                                          api("/reports/" + r.id, {
                                            method: "PATCH",
                                            body: JSON.stringify({
                                              status: "resolved",
                                            }),
                                          }),
                                        "Report marked resolved.",
                                      )
                                    }
                                  >
                                    Resolve
                                  </button>
                                )}
                              </div>
                            </article>
                          ))
                        ) : (
                          <div className="empty-report">
                            <MapPin size={22} />
                            <p>{detail.reports.length ? "No reports match these filters." : "No crew reports at this facility yet."}</p>
                          </div>
                        )}
                      </div>
                      <details className="history-section">
                        <summary>
                          Assessment history ({detail.history.length} snapshots)
                        </summary>
                        {detail.history.length < 2 ? (
                          <p className="body-note">
                            More refreshes are needed before historical trends
                            can be assessed.
                          </p>
                        ) : (
                          <div className="history-table">
                            {detail.history.map((h) => (
                              <div key={h.recorded_at}>
                                <span>{time(h.recorded_at)}</span>
                                <b>{fmt(h.score)} / 10</b>
                                <span>
                                  {Math.round(h.coverage * 100)}% inputs
                                </span>
                              </div>
                            ))}
                          </div>
                        )}
                      </details>
                    </>
                  ) : (
                    <div className="empty">
                      Assessment unavailable. Check the error message and retry
                      selecting this facility.
                    </div>
                  )}
                </section>
              )}
            </>
          )}
          {view === "briefings" && (
            <section className="panel briefing-page">
              <div className="briefing-hero">
                <div className="intro-icon">
                  <FileText size={30} />
                </div>
                <div>
                  <span className="eyebrow">THE NEXT-DAY BRIEF</span>
                  <h2>Next-day briefing for one facility</h2>
                  <p>
                    The scheduler creates a persistent briefing after 18:00
                    local time. It catches up after restart while the
                    application is running.
                  </p>
                </div>
              </div>
              <label className="field">
                Facility
                <select
                  value={selected || ""}
                  onChange={(e) => setSelected(e.target.value || null)}
                >
                  <option value="">Choose a facility</option>
                  {sites.map((s) => (
                    <option value={s.id} key={s.id}>
                      {s.name} · {s.state}
                    </option>
                  ))}
                </select>
              </label>
              <button
                className="button primary"
                disabled={!selected || busy}
                onClick={() =>
                  action(async () => {
                    const b = await api(`/sites/${selected}/briefings`, {
                      method: "POST",
                    });
                    setBriefings(await api(`/sites/${selected}/briefings`));
                    return b;
                  }, "Next-day briefing saved. Existing briefings are kept as immutable snapshots.")
                }
              >
                <Plus size={16} />
                Generate next-day briefing now
              </button>
              <p className="body-note">
                Delivery is local history by default. Configure an HTTPS webhook
                and explicitly enable delivery to send externally. Runs require
                the server to remain on.
              </p>
              {briefings.map((b) => (
                <article className="briefing-card" key={b.id}>
                  <div className="section-title">
                    <div>
                      <span className="eyebrow">
                        {b.payload.timezone} · NEXT LOCAL DAY
                      </span>
                      <h2>{b.payload.site_name}</h2>
                      <p>
                        {b.local_date} · generated {time(b.generated_at)} ·
                        delivery: {b.delivery}
                      </p>
                    </div>
                    <span className="tiny-tag">
                      {b.payload.forecast_hours} FORECAST HOURS
                    </span>
                  </div>
                  <div className="condition-cards">
                    <Condition
                      icon={<Thermometer />}
                      label="Apparent high"
                      value={fmt(b.payload.apparent_max_c) + "°C"}
                      note="Not occupational WBGT"
                    />
                    <Condition
                      icon={<CloudRain />}
                      label="Rainfall"
                      value={fmt(b.payload.rainfall_mm) + " mm"}
                      note="Local calendar day"
                    />
                    <Condition
                      icon={<Wind />}
                      label="Peak gust"
                      value={fmt(b.payload.gust_max_kmh) + " km/h"}
                      note="Forecast estimate"
                    />
                  </div>
                  <ul>
                    {b.payload.actions.map((a: string) => (
                      <li key={a}>{a}</li>
                    ))}
                  </ul>
                  <p className="body-note">{b.payload.note}</p>
                </article>
              ))}
              {!briefings.length && (
                <p className="empty">
                  {selected
                    ? "No saved briefings for this facility. Generate one to inspect tomorrow’s local forecast."
                    : "Select a facility to view and generate its briefings."}
                </p>
              )}
            </section>
          )}
          {view === "missions" && (
            <MissionsView
              api={api}
              action={action}
              busy={busy}
              sites={sites}
              selectedSite={selected}
              revision={revision}
              onReport={(siteId) => {
                setSelected(siteId);
                setModal("report");
              }}
            />
          )}
          {view === "overlaps" && <OverlapsView api={api} />}
          {view === "model" && (
            <section className="panel evaluation-page">
              <div className="section-title">
                <div>
                  <span className="eyebrow">REPRODUCIBLE BENCHMARK</span>
                  <h2>Trained on past days, tested on later ones</h2>
                  <p>
                    NOAA county-event labels · archived GFS forecasts ·
                    chronological held-out evaluation
                  </p>
                </div>
                <Activity size={28} />
              </div>
              {!evaluation ? (
                <div className="loading">
                  <RefreshCw className="spin" />
                  Loading measured evaluation…
                </div>
              ) : (
                <>
                  <div className="method-grid">
                    <div>
                      <strong>{evaluation.rows.toLocaleString()}</strong>
                      <span>Complete county-days</span>
                    </div>
                    <div>
                      <strong>{evaluation.counties} counties</strong>
                      <span>2024–June 2026 convenience sample</span>
                    </div>
                    <div>
                      <strong>≥48 hours</strong>
                      <span>Archived forecast lead per hour</span>
                    </div>
                    <div>
                      <strong>Jan–Jun 2026</strong>
                      <span>Final chronological test</span>
                    </div>
                  </div>
                  <div className="table-scroll">
                    <table>
                      <thead>
                        <tr>
                          <th>Hazard / model</th>
                          <th>Precision</th>
                          <th>Recall</th>
                          <th>PR-AUC</th>
                          <th>ROC-AUC</th>
                          <th>Brier</th>
                          <th>False alerts /100 county-days</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(evaluation.hazards).flatMap(
                          ([h, info]) =>
                            Object.entries((info as any).candidates).map(
                              ([name, r]) => {
                                const m = (r as any).test;
                                return (
                                  <tr key={h + name}>
                                    <td>
                                      <strong>
                                        {h} · {name}
                                      </strong>
                                      {name === (info as any).selected && (
                                        <span className="winner">
                                          selected on validation
                                        </span>
                                      )}
                                    </td>
                                    <td>{fmt(m.precision, 3)}</td>
                                    <td>{fmt(m.recall, 3)}</td>
                                    <td>{fmt(m.pr_auc, 3)}</td>
                                    <td>{fmt(m.roc_auc, 3)}</td>
                                    <td>{fmt(m.brier, 4)}</td>
                                    <td>
                                      {fmt(
                                        m.false_alerts_per_100_county_days,
                                        2,
                                      )}
                                    </td>
                                  </tr>
                                );
                              },
                            ),
                        )}
                      </tbody>
                    </table>
                  </div>
                  <div className="methodology">
                    <h3>The evaluation contract</h3>
                    <p>
                      Fit: 2024. Sigmoid calibration: January–June 2025. Model and threshold selection: July–December 2025, including independent Georgia and Colorado counties. Final temporal test: January–June 2026. California, Arizona, Washington and Utah form the new-model geographic test; Arizona was used by the frozen baseline and is not independent for that candidate.
                    </p>
                    <p>
                      Seven newly fitted logistic/XGBoost/LightGBM candidates compete with the unchanged original estimators. Selection uses mean temporal and geographic validation average precision, with temporal Brier as tie-break. Warning thresholds target at most 20% flagged days and at least 30% recall on validation; these are provisional cost hypotheses. F1/F2 alternatives remain inspectable. No final-test tuning or nationwide validity is claimed.
                    </p>
                    <p className="inline-note warning">
                      Final results are mixed: the original flood estimator was retained. The selected storm model did not improve ranking on the same ten original counties in 2026, and missed all three independent geographic-test events. Lower false-alert rates also sacrifice recall. Geographic validation contains only one flood and three storm positives.
                    </p>
                    <details>
                      <summary>Storm regression investigation, saved models unchanged</summary>
                      <p>Matched average precision fell from 0.150 to 0.130. The paired calendar-week interval for the change is −0.091 to +0.029, so an improvement is not demonstrated. Recall fell from 81.3% to 66.7% as false alerts fell from 40.55 to 19.45 per 100 county-days. Changing the warning threshold cannot repair ranking.</p>
                      <p>The three missed geographic positives are county-days in Los Angeles (February 16 and 18, 2026) and Emery County, Utah (May 13, 2026), representing two NOAA episodes. Both counties lie west of all selected-model fitting coordinates. Geographic validation had just three positive days in one county; geographic transfer and selection stability remain uncertain.</p>
                      <p>Positive-slope calibration preserves the saved rankings; the selected model also overpredicts the matched event rate. Training geography, seasons, features and estimator changed together, so these diagnostics cannot isolate a cause. The predeclared selection and county restrictions remain intact; these known tests cannot independently validate future changes.</p>
                    </details>
                    <h3>Same-test comparison and uncertainty</h3>
                    <p>Both versions below use identical original ten county locations, labels and January–June 2026 dates. The original threshold stays unchanged; updated thresholds use only 2025 validation. AP is average precision; the paired intervals resample calendar weeks with counties kept together.</p>
                    <div className="table-scroll"><table>
                      <thead><tr><th>Hazard / version</th><th>Precision</th><th>Recall</th><th>AP</th><th>Brier</th><th>False alerts /100 county-days</th></tr></thead>
                      <tbody>{Object.entries(evaluation.hazards).flatMap(([h, info]) => ["original_threshold_baseline", "updated"].map(version => {
                        const m = (info as any).same_final_rows[version];
                        return <tr key={h+version}><td>{h} · {version.replaceAll("_", " ")}</td><td>{fmt(m.precision,3)}</td><td>{fmt(m.recall,3)}</td><td>{fmt(m.pr_auc,3)}</td><td>{fmt(m.brier,4)}</td><td>{fmt(m.false_alerts_per_100_county_days,2)}</td></tr>;
                      }))}</tbody>
                    </table></div>
                    <details><summary>Reliability bins, warning policies, prevalence, regional variation and paired uncertainty</summary>
                      <pre>{JSON.stringify(Object.fromEntries(Object.entries(evaluation.hazards).map(([h,i]) => {
                        const info = i as any; const m = info.candidates[info.selected];
                        return [h, {temporal: m.test, geographic: m.geographic_test, threshold_policies: m.threshold_policies, warning_supported: m.warning_supported, quarters: m.quarters, counties: m.counties, paired_uncertainty: info.same_final_rows.paired_uncertainty}];
                      })),null,2)}</pre>
                    </details>
                    <h3>Limitations that travel with the model</h3>
                    <ul>
                      {evaluation.limitations.map((l: string) => (
                        <li key={l}>{l}</li>
                      ))}
                    </ul>
                    <details>
                      <summary>
                        Partition sizes, class balance & geographic holdout
                      </summary>
                      <pre>
                        {JSON.stringify(
                          {
                            partitions: evaluation.partitions,
                            geographic_tests: Object.fromEntries(
                              Object.entries(evaluation.hazards).map(
                                ([h, i]) => [
                                  h,
                                  (i as any).candidates[(i as any).selected]
                                    .geographic_test,
                                ],
                              ),
                            ),
                          },
                          null,
                          2,
                        )}
                      </pre>
                    </details>
                    <p className="body-note">
                      Measured {time(evaluation.created_at)}. Full results and
                      confusion matrices: models/evaluation.json. Reproduce with
                      python -m scripts.train.
                    </p>
                  </div>
                </>
              )}
            </section>
          )}
          <footer className="page-footer">
            <span>
              <Shield size={13} />
              Decision support for supervisor review.
            </span>
            <span>
              PlanIT Control <span className="footer-divider">/</span>{" "}
              Environmental safety intelligence
            </span>
          </footer>
        </div>
      </main>
      {modal && (
        <div
          className="modal-backdrop"
          onMouseDown={(e) => {
            if (e.target === e.currentTarget && !busy) setModal(null);
          }}
        >
          <section
            className="modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="modal-title"
          >
            <div className="section-title">
              <div>
                <div className="eyebrow">
                  {modal === "settings"
                    ? "LOCAL WORKSPACE"
                    : "FIELD OBSERVATION"}
                </div>
                <h2 id="modal-title">
                  {modal === "settings"
                    ? "Supervisor access"
                    : "Submit a crew report"}
                </h2>
              </div>
              <button
                className="icon-button"
                disabled={busy}
                onClick={() => setModal(null)}
                aria-label="Close dialog"
              >
                <X />
              </button>
            </div>
            {modal === "settings" ? (
              <form
                onSubmit={(e) => {
                  e.preventDefault();
                  supervisorToken = token;
                  sessionStorage.setItem("planit-token", token);
                  setModal(null);
                  setNotice("Supervisor token saved for this browser session.");
                }}
              >
                <p className="body-note">
                  Use PLANIT_ADMIN_TOKEN from your local server environment.
                  This enables report review, replay scenarios, manual
                  refreshes, and briefings.
                </p>
                <label className="field">
                  Supervisor token
                  <input
                    autoFocus
                    type="password"
                    value={token}
                    onChange={(e) => setToken(e.target.value)}
                    autoComplete="off"
                    required
                  />
                </label>
                <button className="button primary">
                  Save session access
                  <ArrowRight size={16} />
                </button>
              </form>
            ) : (
              <ReportForm
                site={selectedSite!}
                busy={busy}
                onSubmit={async (form) => {
                  await action(async () => {
                    await api(`/sites/${selected}/reports`, {
                      method: "POST",
                      body: form,
                    });
                    setModal(null);
                  }, "Crew report saved and placed on the map. Awaiting supervisor review.");
                }}
              />
            )}
          </section>
        </div>
      )}
    </div>
  );
}
function Metric({
  title,
  value,
  label,
  icon,
  accent,
}: {
  title: string;
  value: string;
  label: string;
  icon: React.ReactNode;
  accent: string;
}) {
  return (
    <div className="metric">
      <div className="metric-top">
        <span>{title}</span>
        <span className={"metric-icon " + accent}>{icon}</span>
      </div>
      <strong>{value}</strong>
      <p>{label}</p>
    </div>
  );
}
function Condition({
  icon,
  label,
  value,
  note,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  note: string;
}) {
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
function ReportTriage({report, busy, onSave}: {report: any; busy: boolean; onSave: (categories: string[]) => Promise<boolean>}) {
  const [editing, setEditing] = useState(false);
  const [error, setError] = useState("");
  const provider = report.triage?.provider;
  const categories: string[] = report.triage?.manual_categories || [report.classification];
  return <div className="report-triage">
    <small>Classification source: {report.triage?.source || "manual"}. Human review status: {report.status}.</small>
    {provider?.status === "classified" ? <>
      <p>Jev suggestions ({provider.model}): {provider.categories.map(hazardName).join(", ") || (provider.abstained ? "abstained" : "no hazard identified")}. {provider.abstained && (provider.abstention ? `Abstained on hazard labels: ${provider.abstention.reasons.join("; ").replaceAll("_", " ")}.` : "Uncertain. Needs human clarification.")} {provider.abstention?.possible_other && "A further unlisted hazard is possible."}</p>
      <p>{provider.immediate_review === true ? "Suggests immediate supervisor review." : provider.immediate_review === null ? "Priority uncertain; supervisor review required." : "No immediate-priority suggestion; this does not clear work."} {provider.needs_clarification && "Clarify the report."}</p>
      <p>Suggested inbox: {provider.routing.choice.replaceAll("_", " ")}. Review queue: {provider.review_queue}. Urgency: {provider.urgency.score.toFixed(2)} / 2 (0 routine follow-up, 1 prompt review, 2 immediate review of current exposure).</p>
      <details><summary>Unverified Jev judgments and usage</summary><pre>{JSON.stringify({answers: provider.answers, uncertain: provider.uncertain, location_sufficient: provider.location_sufficient, usage: provider.usage},null,2)}</pre></details>
      <small>Provider probabilities and 0.3/0.7 cutoffs are unvalidated for this domain. Suggestions never verify or dismiss reports.</small>
    </> : <p className="body-note">{provider?.reason || "No AI enrichment requested."}</p>}
    <button className="text-button" type="button" onClick={() => setEditing(!editing)}>Correct classifications</button>
    {editing && <form onSubmit={async e => {
      e.preventDefault(); const values = new FormData(e.currentTarget).getAll("category").map(String);
      if (!values.length) {setError("Select at least one classification."); return;}
      setError("");
      if (await onSave(values)) setEditing(false);
    }}>
      <fieldset className="hazard-options"><legend>Supervisor classifications. Select at least one.</legend>{CATEGORIES.map(c => <label key={c}><input name="category" type="checkbox" value={c} defaultChecked={categories.includes(c)} /> {hazardName(c)}</label>)}</fieldset>
      {error && <p role="alert" className="form-error">{error}</p>}
      <button disabled={busy} className="button secondary">Save classifications</button>
    </form>}
  </div>;
}
function ReportForm({
  site,
  busy,
  onSubmit,
}: {
  site: Site;
  busy: boolean;
  onSubmit: (f: FormData) => Promise<void>;
}) {
  const [lat, setLat] = useState(String(site.latitude)),
    [lon, setLon] = useState(String(site.longitude)),
    [geoError, setGeoError] = useState("");
  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        const form = new FormData(e.currentTarget);
        form.set("categories", JSON.stringify(form.getAll("extra_category")));
        form.delete("extra_category");
        const photo = form.get("image");
        if (photo instanceof File && !photo.size) form.delete("image");
        onSubmit(form);
      }}
    >
      <p className="body-note">
        {site.name} · Reports start as pending. Only verified, recent reports
        contribute to the index.
      </p>
      <label className="field">
        Hazard classification
        <select name="classification" required>
          {CATEGORIES.map((v) => (
            <option key={v} value={v}>
              {hazardName(v)}
            </option>
          ))}
        </select>
      </label>
      <fieldset className="hazard-options"><legend>Additional hazards (optional)</legend>
        {CATEGORIES.map(c => <label key={c}><input type="checkbox" name="extra_category" value={c} /> {hazardName(c)}</label>)}
      </fieldset>
      <label className="field">
        What did the crew observe?
        <textarea
          autoFocus
          name="description"
          minLength={10}
          maxLength={2000}
          required
          rows={3}
          placeholder="Describe the hazard, location, and immediate precautions…"
        />
      </label>
      <div className="form-grid">
        <label className="field">
          Latitude
          <input
            type="number"
            step="any"
            min={-90}
            max={90}
            name="latitude"
            value={lat}
            onChange={(e) => setLat(e.target.value)}
            required
          />
        </label>
        <label className="field">
          Longitude
          <input
            type="number"
            step="any"
            min={-180}
            max={180}
            name="longitude"
            value={lon}
            onChange={(e) => setLon(e.target.value)}
            required
          />
        </label>
      </div>
      <button
        type="button"
        className="text-button"
        onClick={() => {
          if (!navigator.geolocation) {
            setGeoError("Browser geolocation unavailable; enter coordinates.");
            return;
          }
          navigator.geolocation.getCurrentPosition(
            (p) => {
              setLat(String(p.coords.latitude));
              setLon(String(p.coords.longitude));
              setGeoError("");
            },
            (e) => setGeoError(e.message),
          );
        }}
      >
        <MapPin size={14} />
        Use my current location
      </button>
      <p className="body-note">
        Coordinates start at the facility. Confirm the actual observation
        location. Maximum 50 miles from this facility.
      </p>
      {geoError && <p className="form-error">{geoError}</p>}
      <label className="field file-field">
        Photo (optional, up to 5 MB)
        <input
          type="file"
          name="image"
          accept="image/jpeg,image/png,image/webp"
        />
      </label>
      <p className="body-note">
        Images are resized and re-encoded to remove embedded metadata. Reports
        and photos remain stored locally unless you explicitly consent to sharing report text below.
      </p>
      <label className="consent-option"><input type="checkbox" name="jev_consent" value="true" />
        I authorize sending this observation text to TypeSafe AI for optional Jev suggestions. Remove personal or sensitive details first. Photos and coordinates are not sent. Server configuration must separately authorize API costs; unavailable Jev leaves manual labels intact.
      </label>
      <button disabled={busy} className="button primary full">
        {busy ? <RefreshCw size={16} className="spin" /> : <Plus size={16} />}
        Save crew report
      </button>
    </form>
  );
}
createRoot(document.getElementById("root")!).render(<App />);
