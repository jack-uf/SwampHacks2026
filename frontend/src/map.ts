import type L from "leaflet";

// A resize must not reset Leaflet's pixel origin while a zoom or flyTo is
// projecting layers. Use one observer instead of Leaflet's window listener.
export function observeMapResize(map: L.Map, container: HTMLElement) {
  let moving = false, pending = false, frame = 0;
  const flush = () => {
    frame = 0;
    if (moving || !pending) return;
    pending = false;
    map.invalidateSize({ animate: false, pan: false });
  };
  const schedule = () => {
    pending = true;
    if (!moving && !frame) frame = requestAnimationFrame(flush);
  };
  const start = () => { moving = true; };
  const end = () => {
    moving = false;
    if (pending && !frame) frame = requestAnimationFrame(flush);
  };
  map.on("movestart", start).on("moveend", end);
  const observer = new ResizeObserver(schedule);
  observer.observe(container);
  return () => {
    observer.disconnect();
    cancelAnimationFrame(frame);
    map.off("movestart", start).off("moveend", end);
  };
}
