"""Leaflet.js + OpenStreetMap Interactive Geospatial Intelligence Map for CyberTrace Phase 8.

Renders an interactive, zero-cost, open-source geospatial workbench visualizing:
1. Complaint Location (Red pin with pulsating indicator)
2. Predicted Withdrawal Zone (Cyan bounding box with probability-scaled fill)
3. Cross-Boundary Candidate Zones (Secondary candidate sectors with active search banner)
4. Ranked Candidate ATMs (Emerald green for #1 candidate, amber for alternative candidates, purple for selected)
5. Comprehensive Dark-Mode Popups, Interactive Map Controls, and Semantic Legend
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import json
import streamlit.components.v1 as components

from src.geographic.map_data import prepare_map_data
from src.utils.logging import get_logger

logger = get_logger("LeafletMap")


def render_investigation_map(
    case_analysis: Dict[str, Any],
    visible_candidate_count: int = 10,
    bank_filter: Optional[str] = None,
    zone_filter: Optional[str] = None,
    selected_atm_id: Optional[str] = None,
    height: int = 560,
) -> None:
    """Render the full interactive geospatial intelligence map from Phase 7 case analysis result.

    Parameters:
        case_analysis: Standardized dictionary from src.intelligence.case_analysis:analyze_case()
        visible_candidate_count: Top N candidates to display (default 10)
        bank_filter: Optional bank name to filter markers
        zone_filter: Optional zone ID to filter markers
        selected_atm_id: Optional ATM ID to highlight and auto-open popup
        height: Map container height in pixels (default 560)
    """
    # 1. Convert analysis result into standardized map data contract
    map_data = prepare_map_data(
        case_analysis=case_analysis,
        max_candidates=visible_candidate_count,
        bank_filter=bank_filter,
        zone_filter=zone_filter,
        selected_atm_id=selected_atm_id,
    )

    complaint_json = json.dumps(map_data.get("complaint_point"))
    pred_zone_json = json.dumps(map_data.get("predicted_zone"))
    cand_zones_json = json.dumps(map_data.get("candidate_zones", []))
    atms_json = json.dumps(map_data.get("ranked_atms", []))
    bounds_json = json.dumps(map_data.get("map_bounds"))
    center_json = json.dumps(map_data.get("map_center", [28.6139, 77.2090]))
    zoom_val = int(map_data.get("map_zoom", 11))
    cross_zone_bool = "true" if map_data.get("cross_zone_search") else "false"
    selected_id_json = json.dumps(selected_atm_id)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CyberTrace Operational Map</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin="" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
  <style>
    body {{
      margin: 0;
      padding: 0;
      background: #080d1a;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      color: #f1f5f9;
      overflow: hidden;
    }}
    #map {{
      width: 100%;
      height: {height}px;
      border-radius: 8px;
      border: 1px solid #1e293b;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.6);
    }}
    
    /* Dark Intelligence Popup Styles */
    .leaflet-popup-content-wrapper {{
      background: #0f172a !important;
      color: #e2e8f0 !important;
      border: 1px solid #38bdf8 !important;
      border-radius: 8px !important;
      box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.8), 0 0 15px rgba(56, 189, 248, 0.2) !important;
      padding: 4px 6px !important;
    }}
    .leaflet-popup-tip {{
      background: #0f172a !important;
      border: 1px solid #38bdf8 !important;
    }}
    .popup-title {{
      font-size: 13px;
      font-weight: 700;
      color: #38bdf8;
      border-bottom: 1px solid #334155;
      padding-bottom: 4px;
      margin-bottom: 6px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .popup-badge {{
      font-size: 10px;
      padding: 2px 6px;
      border-radius: 4px;
      font-weight: 600;
      text-transform: uppercase;
    }}
    .badge-top {{ background: #065f46; color: #34d399; border: 1px solid #10b981; }}
    .badge-alt {{ background: #78350f; color: #fbbf24; border: 1px solid #f59e0b; }}
    .badge-complaint {{ background: #7f1d1d; color: #f87171; border: 1px solid #ef4444; }}
    .badge-zone {{ background: #0c4a6e; color: #38bdf8; border: 1px solid #0284c7; }}

    .popup-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 4px 12px;
      font-size: 11px;
      margin-bottom: 6px;
    }}
    .popup-label {{ color: #94a3b8; font-weight: 500; }}
    .popup-val {{ color: #f8fafc; font-weight: 600; text-align: right; }}

    .popup-scores {{
      background: #1e293b;
      border-radius: 4px;
      padding: 6px 8px;
      margin-top: 6px;
      font-size: 10px;
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 4px;
      text-align: center;
    }}
    .score-box {{ background: #0f172a; padding: 3px; border-radius: 3px; border: 1px solid #334155; }}
    .score-val {{ font-weight: 700; color: #38bdf8; }}
    .score-lbl {{ font-size: 9px; color: #64748b; }}

    .popup-disclaimer {{
      font-size: 9px;
      color: #94a3b8;
      font-style: italic;
      border-top: 1px solid #1e293b;
      padding-top: 4px;
      margin-top: 6px;
      line-height: 1.3;
    }}

    /* Map Controls and Legends */
    .map-legend {{
      background: rgba(15, 23, 42, 0.92) !important;
      border: 1px solid #334155 !important;
      color: #f1f5f9 !important;
      padding: 10px 14px !important;
      border-radius: 8px !important;
      font-size: 11px !important;
      line-height: 1.7 !important;
      box-shadow: 0 4px 14px rgba(0, 0, 0, 0.5) !important;
      backdrop-filter: blur(4px);
    }}
    .legend-item {{ display: flex; align-items: center; margin-bottom: 2px; }}
    .legend-bullet {{
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 50%;
      margin-right: 8px;
      border: 1px solid #ffffff;
    }}
    .legend-rect {{
      display: inline-block;
      width: 12px;
      height: 8px;
      border-radius: 2px;
      margin-right: 8px;
    }}

    .map-btn-bar {{
      background: rgba(15, 23, 42, 0.9) !important;
      border: 1px solid #334155 !important;
      border-radius: 6px !important;
      padding: 4px !important;
      box-shadow: 0 2px 10px rgba(0,0,0,0.5) !important;
    }}
    .map-btn {{
      background: #1e293b;
      color: #e2e8f0;
      border: 1px solid #475569;
      border-radius: 4px;
      padding: 4px 8px;
      font-size: 10px;
      font-weight: 600;
      cursor: pointer;
      margin-right: 4px;
      transition: all 0.15s ease;
    }}
    .map-btn:hover {{
      background: #0284c7;
      color: #ffffff;
      border-color: #38bdf8;
    }}

    .crosszone-banner {{
      background: linear-gradient(90deg, rgba(245, 158, 11, 0.95), rgba(217, 119, 6, 0.95)) !important;
      color: #0f172a !important;
      padding: 6px 12px !important;
      border-radius: 6px !important;
      font-size: 11px !important;
      font-weight: 800 !important;
      box-shadow: 0 4px 12px rgba(245, 158, 11, 0.4) !important;
      letter-spacing: 0.3px;
      border: 1px solid #fbbf24;
      display: flex;
      align-items: center;
      gap: 6px;
    }}

    /* Pulsing Complaint Marker */
    @keyframes pulse {{
      0% {{ r: 10; opacity: 0.9; }}
      50% {{ r: 18; opacity: 0.2; }}
      100% {{ r: 10; opacity: 0.9; }}
    }}
  </style>
</head>
<body>
  <div id="map"></div>

  <script>
    // 1. Initialize Map
    const initialCenter = {center_json};
    const initialZoom = {zoom_val};
    const mapBounds = {bounds_json};
    const isCrossZone = {cross_zone_bool};
    const selectedAtmId = {selected_id_json};

    const map = L.map('map', {{
      center: initialCenter,
      zoom: initialZoom,
      zoomControl: true,
      fadeAnimation: true,
      zoomAnimation: true
    }});

    // 2. OpenStreetMap Standard Carto Tiles (Strictly Free, Open-Source & Attribution Compliant)
    L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      maxZoom: 19,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank">OpenStreetMap</a> contributors'
    }}).addTo(map);

    // Fit Initial Bounds with smooth padding
    if (mapBounds && mapBounds.length === 2) {{
      map.fitBounds(mapBounds, {{ padding: [35, 35], maxZoom: 14 }});
    }}

    // Layer groups for toggling
    const complaintLayer = L.layerGroup().addTo(map);
    const zonesLayer = L.layerGroup().addTo(map);
    const atmsLayer = L.layerGroup().addTo(map);

    let complaintMarker = null;
    let topAtmMarker = null;
    const atmMarkersMap = {{}};

    // 3. Render Complaint Origin Location (Red)
    const complaint = {complaint_json};
    if (complaint && complaint.latitude && complaint.longitude) {{
      // Outer subtle halo
      L.circleMarker([complaint.latitude, complaint.longitude], {{
        radius: 16,
        fillColor: "#ef4444",
        color: "#f87171",
        weight: 1,
        opacity: 0.4,
        fillOpacity: 0.15
      }}).addTo(complaintLayer);

      // Core complaint pin
      complaintMarker = L.circleMarker([complaint.latitude, complaint.longitude], {{
        radius: 9,
        fillColor: "#ef4444",
        color: "#ffffff",
        weight: 2,
        opacity: 1,
        fillOpacity: 0.95
      }}).addTo(complaintLayer);

      const complaintPopup = `
        <div class="popup-title">
          <span>🚨 Complaint Location</span>
          <span class="popup-badge badge-complaint">Origin</span>
        </div>
        <div class="popup-grid">
          <div class="popup-label">Case ID:</div>
          <div class="popup-val">${{complaint.case_id}}</div>
          <div class="popup-label">Reported City:</div>
          <div class="popup-val">${{complaint.city}}</div>
          <div class="popup-label">Disputed Amount:</div>
          <div class="popup-val">₹${{Number(complaint.amount).toLocaleString()}}</div>
          <div class="popup-label">Victim Bank:</div>
          <div class="popup-val">${{complaint.bank}}</div>
          <div class="popup-label">Reported Time:</div>
          <div class="popup-val">${{complaint.time || 'N/A'}} (${{complaint.date || 'N/A'}})</div>
        </div>
        <div class="popup-disclaimer">
          ⚠️ <strong>Investigative Notice:</strong> Victim incident origin reported to authorities. This is NOT the physical cash withdrawal location.
        </div>
      `;
      complaintMarker.bindPopup(complaintPopup);
    }}

    // 4. Render Candidate Zones (Bounding Polygons with Probability Styling)
    const candidateZones = {cand_zones_json};
    if (candidateZones && candidateZones.length > 0) {{
      candidateZones.forEach(zone => {{
        const b = zone.bbox;
        const bounds = [[b.min_lat, b.min_lon], [b.max_lat, b.max_lon]];
        const isPrimary = zone.is_primary;
        const color = zone.color || (isPrimary ? "#00e5ff" : "#f59e0b");
        const fillOp = zone.fill_opacity || (isPrimary ? 0.20 : 0.12);

        const rect = L.rectangle(bounds, {{
          color: color,
          weight: isPrimary ? 2.5 : 2.0,
          dashArray: isPrimary ? "6, 6" : "4, 8",
          fillColor: color,
          fillOpacity: fillOp
        }}).addTo(zonesLayer);

        const zonePopup = `
          <div class="popup-title">
            <span>🎯 ${{zone.zone_id}}</span>
            <span class="popup-badge ${{isPrimary ? 'badge-top' : 'badge-alt'}}">${{isPrimary ? 'Primary Sector' : 'Cross-Zone Sector'}}</span>
          </div>
          <div class="popup-grid">
            <div class="popup-label">Model Probability:</div>
            <div class="popup-val" style="color:${{color}};">${{zone.probability}}%</div>
            <div class="popup-label">Search Status:</div>
            <div class="popup-val">${{isPrimary ? 'Active Primary' : 'Active Cross-Zone'}}</div>
            <div class="popup-label">Role:</div>
            <div class="popup-val">${{isPrimary ? 'Highest Model Probability' : 'Contested Boundary Area'}}</div>
          </div>
          <div class="popup-disclaimer">
            Predicted withdrawal sector derived by supervised machine learning model. Represents statistical area of interest.
          </div>
        `;
        rect.bindPopup(zonePopup);
      }});
    }}

    // 5. Render Ranked ATM Candidate Markers
    const atms = {atms_json};
    if (atms && atms.length > 0) {{
      atms.forEach((atm, index) => {{
        if (atm.latitude && atm.longitude) {{
          const isTop = atm.is_top || (index === 0);
          const isSelected = atm.is_selected;
          const color = atm.marker_color || (isTop ? "#10b981" : "#f59e0b");
          const radius = atm.marker_radius || (isTop ? 10 : 7);

          const marker = L.circleMarker([atm.latitude, atm.longitude], {{
            radius: radius,
            fillColor: color,
            color: isSelected ? "#ffffff" : (isTop ? "#ffffff" : "#1e293b"),
            weight: isTop ? 2.5 : 1.8,
            opacity: 1,
            fillOpacity: 0.90
          }}).addTo(atmsLayer);

          atmMarkersMap[atm.atm_id] = marker;
          if (isTop && !topAtmMarker) {{
            topAtmMarker = marker;
          }}

          const flagTags = (atm.evidence_flags || []).map(f => `<span style="background:#1e293b; color:#94a3b8; padding:1px 4px; border-radius:3px; font-size:9px; border:1px solid #334155; margin-right:2px;">${{f}}</span>`).join(' ');

          const popupContent = `
            <div class="popup-title">
              <span>${{isTop ? '⭐ ATM CANDIDATE #1' : '📍 Candidate #' + atm.rank}}</span>
              <span class="popup-badge ${{isTop ? 'badge-top' : 'badge-alt'}}">${{isTop ? 'Highest-Ranked' : 'Alternative'}}</span>
            </div>
            <div class="popup-grid">
              <div class="popup-label">ATM Node ID:</div>
              <div class="popup-val">${{atm.atm_id}}</div>
              <div class="popup-label">Bank:</div>
              <div class="popup-val" style="color:#38bdf8;">${{atm.bank}}</div>
              <div class="popup-label">Operator:</div>
              <div class="popup-val">${{atm.operator}}</div>
              <div class="popup-label">Location:</div>
              <div class="popup-val">${{atm.city}} (${{atm.zone}})</div>
              <div class="popup-label">Distance:</div>
              <div class="popup-val">${{Number(atm.distance_km).toFixed(1)}} km</div>
              <div class="popup-label">Overall Score:</div>
              <div class="popup-val" style="color:#10b981; font-size:12px;"><strong>${{Number(atm.overall_score).toFixed(1)}} / 100</strong></div>
            </div>

            <div class="popup-scores">
              <div class="score-box">
                <div class="score-val">${{Number(atm.zone_probability_score || 0).toFixed(0)}}</div>
                <div class="score-lbl">Zone Prob</div>
              </div>
              <div class="score-box">
                <div class="score-val">${{Number(atm.spatial_score || 0).toFixed(0)}}</div>
                <div class="score-lbl">Spatial</div>
              </div>
              <div class="score-box">
                <div class="score-val">${{Number(atm.bank_score || 0).toFixed(0)}}</div>
                <div class="score-lbl">Bank</div>
              </div>
              <div class="score-box">
                <div class="score-val">${{Number(atm.time_score || 0).toFixed(0)}}</div>
                <div class="score-lbl">Time</div>
              </div>
              <div class="score-box">
                <div class="score-val">${{Number(atm.activity_score || 0).toFixed(0)}}</div>
                <div class="score-lbl">Activity</div>
              </div>
              <div class="score-box">
                <div class="score-val">${{Number(atm.amount_compatibility_score || 0).toFixed(0)}}</div>
                <div class="score-lbl">Amount</div>
              </div>
            </div>

            <div style="margin-top:6px; font-size:10px; color:#cbd5e1;">
              <strong>Signals:</strong> ${{flagTags || 'Standard candidate'}}
            </div>
            
            <div class="popup-disclaimer">
              ℹ️ Candidate ranking generated by multi-criteria heuristic scoring. Does NOT establish confirmed physical withdrawal without banking audit records.
            </div>
          `;

          marker.bindPopup(popupContent);

          if (isSelected) {{
            marker.openPopup();
          }}
        }}
      }});
    }}

    // 6. Cross-Zone Search Banner
    if (isCrossZone) {{
      const crossBanner = L.control({{ position: 'topright' }});
      crossBanner.onAdd = function() {{
        const div = L.DomUtil.create('div', 'crosszone-banner');
        div.innerHTML = `🔄 <strong>CROSS-ZONE CANDIDATE SEARCH ACTIVE</strong>`;
        return div;
      }};
      crossBanner.addTo(map);
    }}

    // 7. Map Navigation Controls (Top-Left under zoom)
    const navControl = L.control({{ position: 'topleft' }});
    navControl.onAdd = function() {{
      const div = L.DomUtil.create('div', 'map-btn-bar');
      div.innerHTML = `
        <button class="map-btn" onclick="resetMapView()">⟲ Fit All</button>
        <button class="map-btn" onclick="focusComplaint()">🚨 Complaint</button>
        <button class="map-btn" onclick="focusTopAtm()">⭐ Top ATM</button>
      `;
      L.DomEvent.disableClickPropagation(div);
      return div;
    }};
    navControl.addTo(map);

    // Global navigation handlers
    window.resetMapView = function() {{
      if (mapBounds && mapBounds.length === 2) {{
        map.flyToBounds(mapBounds, {{ padding: [35, 35], duration: 1.2 }});
      }}
    }};

    window.focusComplaint = function() {{
      if (complaintMarker) {{
        map.flyTo(complaintMarker.getLatLng(), 13, {{ duration: 1.0 }});
        complaintMarker.openPopup();
      }}
    }};

    window.focusTopAtm = function() {{
      if (topAtmMarker) {{
        map.flyTo(topAtmMarker.getLatLng(), 14, {{ duration: 1.0 }});
        topAtmMarker.openPopup();
      }}
    }};

    // 8. Map Semantic Legend (Bottom-Right)
    const legend = L.control({{ position: 'bottomright' }});
    legend.onAdd = function() {{
      const div = L.DomUtil.create('div', 'map-legend');
      div.innerHTML = `
        <div style="font-weight:700; color:#38bdf8; margin-bottom:4px; font-size:11px; border-bottom:1px solid #334155; padding-bottom:2px;">
          GEOSPATIAL INTELLIGENCE LEGEND
        </div>
        <div class="legend-item">
          <span class="legend-bullet" style="background:#ef4444;"></span> Complaint Location
        </div>
        <div class="legend-item">
          <span class="legend-rect" style="border:2px dashed #00e5ff; background:rgba(0,229,255,0.2);"></span> Primary Predicted Zone
        </div>
        ${{isCrossZone ? `
        <div class="legend-item">
          <span class="legend-rect" style="border:2px dashed #f59e0b; background:rgba(245,158,11,0.15);"></span> Secondary Candidate Zone
        </div>` : ''}}
        <div class="legend-item">
          <span class="legend-bullet" style="background:#10b981; border:2px solid #ffffff; width:11px; height:11px;"></span> Highest-Ranked ATM (#1)
        </div>
        <div class="legend-item">
          <span class="legend-bullet" style="background:#f59e0b;"></span> Alternative ATM Candidates
        </div>
      `;
      return div;
    }};
    legend.addTo(map);

  </script>
</body>
</html>
"""
    components.html(html_content, height=height + 20)


def render_neutral_map(height: int = 540) -> None:
    """Render a neutral empty-state map before case analysis is submitted."""
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CyberTrace Neutral Map</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin="" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
  <style>
    body {{
      margin: 0;
      padding: 0;
      background: #f8fafc;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }}
    #map {{
      width: 100%;
      height: {height}px;
      border-radius: 12px;
      border: 1px solid #e2e8f0;
      box-shadow: 0 4px 20px rgba(15, 23, 42, 0.05);
    }}
    .empty-banner {{
      position: absolute;
      top: 20px;
      left: 50%;
      transform: translateX(-50%);
      z-index: 1000;
      background: rgba(255, 255, 255, 0.95);
      backdrop-filter: blur(8px);
      border: 1px solid #e2e8f0;
      border-radius: 9999px;
      padding: 8px 20px;
      font-size: 13px;
      font-weight: 600;
      color: #0f172a;
      box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
      display: flex;
      align-items: center;
      gap: 8px;
      pointer-events: none;
    }}
  </style>
</head>
<body>
  <div class="empty-banner">
    <span>📍</span> Geographic intelligence will appear here after case analysis
  </div>
  <div id="map"></div>
  <script>
    var map = L.map('map', {{
      center: [29.5, 76.5],
      zoom: 7,
      zoomControl: true
    }});
    L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank">OpenStreetMap</a> contributors',
      maxZoom: 18
    }}).addTo(map);
  </script>
</body>
</html>
"""
    components.html(html_content, height=height + 15)

