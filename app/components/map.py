"""Leaflet.js + OpenStreetMap interactive mapping component for CyberTrace."""

from typing import List, Dict, Any, Optional, Tuple
import json
import streamlit.components.v1 as components

def render_leaflet_map(
    center_lat: float = 31.3260,
    center_lon: float = 75.5762,
    zoom: int = 12,
    complaint_point: Optional[Tuple[float, float, str]] = None,
    predicted_zone_bbox: Optional[Tuple[float, float, float, float]] = None,
    candidate_atms: Optional[List[Dict[str, Any]]] = None,
    height: int = 520,
) -> None:
    """Render an interactive Leaflet map embedded directly into Streamlit via OpenStreetMap tiles."""
    candidate_atms = candidate_atms or []

    # Prepare JSON payloads
    complaint_json = (
        json.dumps({
            "lat": complaint_point[0],
            "lon": complaint_point[1],
            "label": complaint_point[2],
        })
        if complaint_point
        else "null"
    )

    bbox_json = (
        json.dumps({
            "min_lat": predicted_zone_bbox[0],
            "min_lon": predicted_zone_bbox[1],
            "max_lat": predicted_zone_bbox[2],
            "max_lon": predicted_zone_bbox[3],
        })
        if predicted_zone_bbox
        else "null"
    )

    atms_json = json.dumps(candidate_atms)

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin="" />
      <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
      <style>
        body {{ margin: 0; padding: 0; background: #0a0e17; font-family: 'Segoe UI', Arial, sans-serif; }}
        #map {{ width: 100%; height: {height}px; border-radius: 8px; border: 1px solid #1f2e48; }}
        .leaflet-popup-content-wrapper {{
          background: #111827;
          color: #f1f5f9;
          border: 1px solid #00e5ff;
          border-radius: 6px;
          box-shadow: 0 4px 14px rgba(0,0,0,0.5);
        }}
        .leaflet-popup-tip {{ background: #111827; }}
        .map-legend {{
          background: rgba(17, 24, 39, 0.9);
          border: 1px solid #1f2e48;
          color: #f1f5f9;
          padding: 8px 12px;
          border-radius: 6px;
          font-size: 11px;
          line-height: 1.6;
        }}
        .legend-bullet {{
          display: inline-block;
          width: 10px;
          height: 10px;
          border-radius: 50%;
          margin-right: 6px;
        }}
      </style>
    </head>
    <body>
      <div id="map"></div>
      <script>
        const map = L.map('map').setView([{center_lat}, {center_lon}], {zoom});

        // OpenStreetMap Carto tiles (Free & Open Source, No Google Maps API)
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
          maxZoom: 19,
          attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank">OpenStreetMap</a> contributors'
        }}).addTo(map);

        const complaint = {complaint_json};
        const bbox = {bbox_json};
        const atms = {atms_json};

        // 1. Complaint Location Marker (Red)
        if (complaint) {{
          const redIcon = L.circleMarker([complaint.lat, complaint.lon], {{
            radius: 9,
            fillColor: "#ef4444",
            color: "#ffffff",
            weight: 2,
            opacity: 1,
            fillOpacity: 0.9
          }}).addTo(map);
          redIcon.bindPopup("<strong>🚨 Complaint Origin</strong><br>" + complaint.label);
        }}

        // 2. Predicted Withdrawal Zone Bounding Box (Cyan/Blue boundary)
        if (bbox) {{
          const bounds = [[bbox.min_lat, bbox.min_lon], [bbox.max_lat, bbox.max_lon]];
          const rect = L.rectangle(bounds, {{
            color: "#00e5ff",
            weight: 2,
            dashArray: "4, 6",
            fillColor: "#0284c7",
            fillOpacity: 0.15
          }}).addTo(map);
          rect.bindPopup("<strong>🎯 Predicted Cash-Out Zone</strong><br>Statistical geographic bounding box");
          map.fitBounds(bounds, {{ padding: [30, 30] }});
        }}

        // 3. Candidate ATMs
        atms.forEach((atm, index) => {{
          if (atm.latitude && atm.longitude) {{
            const isTop = index === 0;
            const color = isTop ? "#10b981" : "#f59e0b";
            const radius = isTop ? 8 : 6;

            const marker = L.circleMarker([atm.latitude, atm.longitude], {{
              radius: radius,
              fillColor: color,
              color: "#ffffff",
              weight: 1.5,
              opacity: 1,
              fillOpacity: 0.85
            }}).addTo(map);

            const popupContent = `
              <strong>${{isTop ? '⭐ HIGHEST-RANKED CANDIDATE' : '📍 Candidate ATM'}}</strong><br>
              <strong>ID:</strong> ${{atm.atm_id}}<br>
              <strong>Bank:</strong> ${{atm.bank}}<br>
              <strong>Distance:</strong> ${{atm.distance_km}} km<br>
              <strong>Candidate Score:</strong> ${{atm.candidate_score}}<br>
              <small>Source: ${{atm.source || 'OpenStreetMap'}}</small>
            `;
            marker.bindPopup(popupContent);
          }}
        }});

        // Add Legend Control
        const legend = L.control({{ position: 'bottomright' }});
        legend.onAdd = function() {{
          const div = L.DomUtil.create('div', 'map-legend');
          div.innerHTML = `
            <strong>Legend</strong><br>
            <span class="legend-bullet" style="background:#ef4444;"></span> Complaint Location<br>
            <span class="legend-bullet" style="background:#00e5ff;"></span> Predicted Zone Area<br>
            <span class="legend-bullet" style="background:#10b981;"></span> Top ATM Candidate<br>
            <span class="legend-bullet" style="background:#f59e0b;"></span> Alternate Candidate
          `;
          return div;
        }};
        legend.addTo(map);
      </script>
    </body>
    </html>
    """
    components.html(html_content, height=height + 20)
