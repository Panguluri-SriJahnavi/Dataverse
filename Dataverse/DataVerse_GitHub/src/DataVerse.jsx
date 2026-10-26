import React, { useState, useEffect, useRef, useCallback, useMemo } from "react";
import * as d3 from "d3";
import { Upload, X } from "lucide-react";

// ---------- palette ----------
const CLUSTER_COLORS = ["#5EEAD4", "#A78BFA", "#FBBF24", "#FB7185", "#86EFAC", "#60A5FA"];
const OUTLIER_COLOR = "#F87171";
const MAX_ROWS = 300; // keeps the animation smooth; bigger files are randomly sampled

// ---------- helpers start ----------
const MISSING = new Set(["", "nan", "null", "na", "n/a", "none", "undefined"]);
const isMissing = (v) => v === undefined || v === null || MISSING.has(String(v).trim().toLowerCase());
const num = (v) => {
  if (isMissing(v)) return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
};
const pretty = (s) => String(s).replace(/[_-]+/g, " ").trim();
const fmt = (n) => (Math.abs(n) >= 100 ? n.toFixed(0) : n.toFixed(1));

// A column counts as numeric if every filled value is a number and at least 60% of rows are filled.
// (Cancelled rides, for example, have blank fare columns - those columns should still be usable.)
function numericColumns(rows) {
  if (!rows.length) return [];
  const keys = Object.keys(rows[0]).filter((k) => k !== "_key");
  return keys.filter((k) => {
    if (/(^|[_\s-])id$/i.test(k)) return false; // ID columns are labels, not features
    let filled = 0;
    const seen = new Set();
    let allInt = true;
    for (const r of rows) {
      if (isMissing(r[k])) continue;
      const n = num(r[k]);
      if (n === null) return false; // contains text, so not numeric
      filled++;
      seen.add(n);
      if (!Number.isInteger(n)) allInt = false;
    }
    if (filled < rows.length * 0.6) return false;
    if (allInt && filled > 20 && seen.size === filled) return false; // unique running numbers look like IDs
    return true;
  });
}

function columnStats(rows, cols) {
  const stats = {};
  cols.forEach((c) => {
    const vals = rows.map((r) => num(r[c])).filter((v) => v !== null);
    stats[c] = {
      mean: d3.mean(vals),
      std: d3.deviation(vals) || 1,
      min: d3.min(vals),
      max: d3.max(vals),
    };
  });
  return stats;
}

// min-max scale each column to 0..1 so no column dominates; blanks are filled with the column average
function normalize(rows, cols, stats) {
  return rows.map((r) =>
    cols.map((c) => {
      const { mean, min, max } = stats[c];
      const v = num(r[c]) ?? mean;
      return max > min ? (v - min) / (max - min) : 0;
    })
  );
}

function kmeansOnce(points, k, iterations = 30) {
  const n = points.length;
  const dim = points[0].length;
  const dist = (a, b) => {
    let s = 0;
    for (let i = 0; i < dim; i++) {
      const d = a[i] - b[i];
      s += d * d;
    }
    return s;
  };

  // k-means++ start: spread the first centroids apart
  const centroids = [[...points[Math.floor(Math.random() * n)]]];
  while (centroids.length < k) {
    const d2 = points.map((p) => Math.min(...centroids.map((c) => dist(p, c))));
    const total = d2.reduce((a, b) => a + b, 0);
    let pick = Math.floor(Math.random() * n);
    if (total > 0) {
      let r = Math.random() * total;
      for (let i = 0; i < n; i++) {
        r -= d2[i];
        if (r <= 0) {
          pick = i;
          break;
        }
      }
    }
    centroids.push([...points[pick]]);
  }

  let assignments = new Array(n).fill(0);
  for (let iter = 0; iter < iterations; iter++) {
    for (let i = 0; i < n; i++) {
      let best = 0,
        bestD = Infinity;
      for (let c = 0; c < k; c++) {
        const d = dist(points[i], centroids[c]);
        if (d < bestD) {
          bestD = d;
          best = c;
        }
      }
      assignments[i] = best;
    }
    const sums = centroids.map(() => new Array(dim).fill(0));
    const counts = new Array(k).fill(0);
    for (let i = 0; i < n; i++) {
      const c = assignments[i];
      counts[c]++;
      for (let d = 0; d < dim; d++) sums[c][d] += points[i][d];
    }
    for (let c = 0; c < k; c++) {
      if (counts[c] > 0) centroids[c] = sums[c].map((s) => s / counts[c]);
    }
  }
  let inertia = 0;
  for (let i = 0; i < n; i++) inertia += dist(points[i], centroids[assignments[i]]);
  return { assignments, inertia };
}

// run k-means a few times and keep the tightest result (a single random start can land on a bad grouping)
function kmeans(points, k, restarts = 6) {
  let best = null;
  for (let r = 0; r < restarts; r++) {
    const res = kmeansOnce(points, k);
    if (!best || res.inertia < best.inertia) best = res;
  }
  return best.assignments;
}

function detectOutliers(rows, cols, stats, threshold = 2.5) {
  return rows.map((r) => {
    for (const c of cols) {
      const v = num(r[c]);
      if (v === null) continue;
      const { mean, std } = stats[c];
      if (Math.abs((v - mean) / std) > threshold) return true;
    }
    return false;
  });
}

function pearson(a, b) {
  const ma = d3.mean(a),
    mb = d3.mean(b);
  let numr = 0,
    da = 0,
    db = 0;
  for (let i = 0; i < a.length; i++) {
    numr += (a[i] - ma) * (b[i] - mb);
    da += (a[i] - ma) ** 2;
    db += (b[i] - mb) ** 2;
  }
  const denom = Math.sqrt(da * db);
  return denom ? numr / denom : 0;
}

// correlation between two columns, using only rows where both have a value
function colCorr(rows, colA, colB) {
  const a = [],
    b = [];
  rows.forEach((r) => {
    const x = num(r[colA]),
      y = num(r[colB]);
    if (x !== null && y !== null) {
      a.push(x);
      b.push(y);
    }
  });
  return a.length < 10 ? 0 : pearson(a, b);
}

function topCorrelation(rows, cols) {
  if (cols.length < 2) return null;
  let best = null;
  for (let i = 0; i < cols.length; i++) {
    for (let j = i + 1; j < cols.length; j++) {
      const r = colCorr(rows, cols[i], cols[j]);
      if (!best || Math.abs(r) > Math.abs(best.r)) best = { a: cols[i], b: cols[j], r };
    }
  }
  return best;
}

const levelWord = (z) => (z >= 1 ? "Very high" : z >= 0.4 ? "High" : z <= -1 ? "Very low" : "Low");

// Give every cluster a readable name by checking which columns make it stand out
// from the dataset as a whole (how many standard deviations its average is above/below overall).
function describeClusters(rows, cols, stats, assignments, outliers, k) {
  const usedNames = {};
  const result = [];
  for (let c = 0; c < k; c++) {
    let idx = [];
    for (let i = 0; i < rows.length; i++) if (assignments[i] === c && !outliers[i]) idx.push(i);
    if (!idx.length) for (let i = 0; i < rows.length; i++) if (assignments[i] === c) idx.push(i);

    const feats = cols.map((col) => {
      const vals = idx.map((i) => num(rows[i][col])).filter((v) => v !== null);
      const mean = vals.length ? d3.mean(vals) : stats[col].mean;
      return { col, mean, overall: stats[col].mean, z: (mean - stats[col].mean) / stats[col].std };
    });
    // pick the 2 most distinctive columns, skipping a column that is almost a copy of one already picked
    // (e.g. "total fare" vs "ride charge"), so the name doesn't say the same thing twice
    const top = [];
    [...feats]
      .sort((a, b) => Math.abs(b.z) - Math.abs(a.z))
      .filter((f) => Math.abs(f.z) >= 0.4)
      .forEach((f) => {
        if (top.length >= 2) return;
        if (top.some((t) => Math.abs(colCorr(rows, t.col, f.col)) > 0.9)) return;
        top.push(f);
      });

    const parts = top.length ? top.map((f) => `${levelWord(f.z)} ${pretty(f.col)}`) : ["Average / mixed"];
    let name = parts.join(" · ");
    usedNames[name] = (usedNames[name] || 0) + 1;
    if (usedNames[name] > 1) name += ` (${usedNames[name]})`;

    const detail = top.length
      ? top.map((f) => `${pretty(f.col)} avg ${fmt(f.mean)} (overall ${fmt(f.overall)})`).join(" · ")
      : "Close to the overall average on every column";

    result.push({ name, parts, detail });
  }
  return result;
}

function splitLine(line) {
  const out = [];
  let cur = "";
  let inQ = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') {
      if (inQ && line[i + 1] === '"') {
        cur += '"';
        i++;
      } else inQ = !inQ;
    } else if (ch === "," && !inQ) {
      out.push(cur);
      cur = "";
    } else cur += ch;
  }
  out.push(cur);
  return out;
}

// Reads the CSV; if there are more than maxRows rows, a random sample is used so the picture stays readable
function parseCSV(text, maxRows = MAX_ROWS) {
  const lines = text
    .replace(/^﻿/, "")
    .split(/\r?\n/)
    .filter((l) => l.trim() !== "");
  if (lines.length < 2) return { rows: [], total: 0 };
  const headers = splitLine(lines[0]).map((h) => h.trim());
  const body = lines.slice(1);
  const total = body.length;
  let picks = body.map((_, i) => i);
  if (total > maxRows) {
    d3.shuffle(picks);
    picks = picks.slice(0, maxRows).sort((a, b) => a - b);
  }
  const rows = picks.map((i) => {
    const cells = splitLine(body[i]);
    const obj = { _key: i };
    headers.forEach((h, idx) => (obj[h] = (cells[idx] || "").trim()));
    return obj;
  });
  return { rows, total };
}
// ---------- helpers end ----------

// ---------- synthetic default dataset ----------
function makeSampleData() {
  const rows = [];
  const groups = [
    { name: "A", cx: 20, cy: 80, n: 40 },
    { name: "B", cx: 70, cy: 30, n: 40 },
    { name: "C", cx: 50, cy: 60, n: 35 },
  ];
  let id = 1;
  groups.forEach((g) => {
    for (let i = 0; i < g.n; i++) {
      rows.push({
        _key: id,
        id: id++,
        score: +(g.cx + (Math.random() - 0.5) * 18).toFixed(1),
        engagement: +(g.cy + (Math.random() - 0.5) * 18).toFixed(1),
        category: g.name,
      });
    }
  });
  for (let i = 0; i < 6; i++) {
    rows.push({
      _key: id,
      id: id++,
      score: +(Math.random() * 100).toFixed(1),
      engagement: +(Math.random() < 0.5 ? -20 - Math.random() * 20 : 120 + Math.random() * 20).toFixed(1),
      category: "?",
    });
  }
  return rows;
}

export default function DataVerse() {
  const [rawRows, setRawRows] = useState(() => makeSampleData());
  const [totalInFile, setTotalInFile] = useState(null);
  const [k, setK] = useState(3);
  const [selectedKey, setSelectedKey] = useState(null);
  const [fileName, setFileName] = useState(null);
  const [selectedCols, setSelectedCols] = useState(null); // null = use every numeric column
  const svgRef = useRef(null);
  const simRef = useRef(null);
  const [nodes, setNodes] = useState([]);
  const [dims, setDims] = useState({ w: 900, h: 560 });
  const containerRef = useRef(null);

  const numCols = useMemo(() => numericColumns(rawRows), [rawRows]);
  const cols = useMemo(
    () => (selectedCols ? numCols.filter((c) => selectedCols.includes(c)) : numCols),
    [numCols, selectedCols]
  );

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const cr = entries[0].contentRect;
      setDims((prev) =>
        Math.abs(prev.w - cr.width) < 2 && Math.abs(prev.h - cr.height) < 2 ? prev : { w: cr.width, h: cr.height }
      );
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const analysis = useMemo(() => {
    if (!rawRows.length || cols.length === 0) return null;
    const stats = columnStats(rawRows, cols);
    const normed = normalize(rawRows, cols, stats);
    const effectiveK = Math.min(k, rawRows.length);
    const assignments = kmeans(normed, effectiveK);
    const outliers = detectOutliers(rawRows, cols, stats);
    const corr = topCorrelation(rawRows, cols);
    const clusterCounts = {};
    assignments.forEach((a, i) => {
      if (outliers[i]) return;
      clusterCounts[a] = (clusterCounts[a] || 0) + 1;
    });
    const clusters = describeClusters(rawRows, cols, stats, assignments, outliers, effectiveK);
    return { assignments, outliers, corr, clusterCounts, effectiveK, clusters };
  }, [rawRows, cols, k]);

  // build + run force simulation whenever analysis or size changes
  useEffect(() => {
    if (!analysis) {
      setNodes([]);
      return;
    }
    const { w, h } = dims;
    const cx = w / 2,
      cy = h / 2;
    const R = Math.min(w, h) * 0.3;
    const clusterAngle = (c) => (c / analysis.effectiveK) * Math.PI * 2;

    const simNodes = rawRows.map((r, i) => {
      const isOutlier = analysis.outliers[i];
      const cluster = analysis.assignments[i];
      let tx, ty;
      if (isOutlier) {
        const a = (i * 137.5 * Math.PI) / 180; // golden-angle scatter
        tx = cx + Math.cos(a) * R * 1.7;
        ty = cy + Math.sin(a) * R * 1.7;
      } else {
        const a = clusterAngle(cluster);
        tx = cx + Math.cos(a) * R;
        ty = cy + Math.sin(a) * R;
      }
      return {
        ...r,
        __cluster: cluster,
        __outlier: isOutlier,
        targetX: tx,
        targetY: ty,
        x: cx + (Math.random() - 0.5) * 40,
        y: cy + (Math.random() - 0.5) * 40,
      };
    });

    if (simRef.current) simRef.current.stop();

    const sim = d3
      .forceSimulation(simNodes)
      .force("x", d3.forceX((d) => d.targetX).strength(0.12))
      .force("y", d3.forceY((d) => d.targetY).strength(0.12))
      .force("charge", d3.forceManyBody().strength(-6))
      .force("collide", d3.forceCollide(7))
      .alpha(1)
      .alphaDecay(0.02)
      .on("tick", () => setNodes([...sim.nodes()]));

    simRef.current = sim;
    return () => sim.stop();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [analysis, dims.w, dims.h]);

  // names that float above each group of dots
  const clusterLabels = useMemo(() => {
    if (!analysis || !nodes.length) return [];
    const groups = {};
    nodes.forEach((n) => {
      if (n.__outlier) return;
      if (!groups[n.__cluster]) groups[n.__cluster] = [];
      groups[n.__cluster].push(n);
    });
    return Object.entries(groups)
      .map(([c, ns]) => {
        const info = analysis.clusters[c];
        if (!info) return null;
        const x = Math.min(Math.max(d3.mean(ns, (d) => d.x), 100), dims.w - 100);
        const y = Math.max(18, d3.min(ns, (d) => d.y) - 16 - (info.parts.length - 1) * 14);
        return { c: +c, x, y, parts: info.parts };
      })
      .filter(Boolean);
  }, [nodes, analysis, dims.w]);

  const handleFile = useCallback((file) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const { rows, total } = parseCSV(e.target.result);
      setRawRows(rows);
      setTotalInFile(total);
      setFileName(file.name);
      setSelectedCols(null);
      setSelectedKey(null);
    };
    reader.readAsText(file);
  }, []);

  const toggleCol = (c) => {
    const current = cols;
    if (current.includes(c)) {
      if (current.length === 1) return; // keep at least one column
      setSelectedCols(current.filter((x) => x !== c));
    } else {
      setSelectedCols([...current, c]);
    }
  };

  const outlierCount = analysis ? analysis.outliers.filter(Boolean).length : 0;
  const shownRows = rawRows.length;

  return (
    <div
      style={{
        fontFamily: "'Inter', system-ui, sans-serif",
        background: "#0B0E1A",
        color: "#E7EAF3",
        width: "100%",
        height: "100vh",
        display: "flex",
        overflow: "hidden",
      }}
    >
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');
        .dl-scroll::-webkit-scrollbar { width: 6px; }
        .dl-scroll::-webkit-scrollbar-thumb { background: #2A3348; border-radius: 4px; }
        @keyframes dl-pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.45; } }
      `}</style>

      {/* Sidebar */}
      <div
        className="dl-scroll"
        style={{
          width: 320,
          flexShrink: 0,
          borderRight: "1px solid #1B2333",
          padding: "28px 22px",
          overflowY: "auto",
          background: "#0D1120",
        }}
      >
        <div style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: 22, fontWeight: 700, letterSpacing: "-0.02em" }}>
          DataVerse
        </div>
        <div style={{ fontSize: 13, color: "#8891A7", marginTop: 6, lineHeight: 1.5 }}>
          Upload a dataset to see its groups and unusual records laid out visually.
        </div>

        <label
          style={{
            marginTop: 22,
            display: "flex",
            alignItems: "center",
            gap: 8,
            border: "1px dashed #2A3348",
            borderRadius: 8,
            padding: "12px 14px",
            cursor: "pointer",
            fontSize: 13,
            color: "#B9C0D4",
          }}
        >
          <Upload size={15} />
          {fileName ? fileName : "Upload CSV"}
          <input
            type="file"
            accept=".csv"
            style={{ display: "none" }}
            onChange={(e) => e.target.files[0] && handleFile(e.target.files[0])}
          />
        </label>
        {totalInFile !== null && totalInFile > shownRows && (
          <div style={{ fontSize: 11.5, color: "#8891A7", marginTop: 8, lineHeight: 1.5 }}>
            Showing a random sample of {shownRows.toLocaleString()} of {totalInFile.toLocaleString()} rows.
          </div>
        )}

        <div style={{ marginTop: 22 }}>
          <div style={{ fontSize: 12, color: "#8891A7", marginBottom: 8 }}>Group records using these columns</div>
          {numCols.length === 0 && (
            <div style={{ fontSize: 12, color: "#F87171" }}>No numeric columns found in this file.</div>
          )}
          {numCols.map((c) => (
            <label
              key={c}
              style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12.5, color: "#C7CCDC", marginBottom: 5, cursor: "pointer" }}
            >
              <input type="checkbox" checked={cols.includes(c)} onChange={() => toggleCol(c)} />
              {pretty(c)}
            </label>
          ))}
        </div>

        <div style={{ marginTop: 22 }}>
          <div style={{ fontSize: 12, color: "#8891A7", marginBottom: 8 }}>Number of groups (k = {k})</div>
          <input type="range" min={2} max={6} value={k} onChange={(e) => setK(parseInt(e.target.value))} style={{ width: "100%" }} />
        </div>

        <div style={{ marginTop: 24, display: "flex", gap: 10 }}>
          <StatBlock label="Records shown" value={shownRows} />
          <StatBlock label="Outliers" value={outlierCount} accent={OUTLIER_COLOR} />
        </div>

        <div style={{ marginTop: 24 }}>
          <div style={{ fontSize: 12, color: "#8891A7", marginBottom: 10 }}>Groups</div>
          {analysis &&
            analysis.clusters.map((info, c) => (
              <div key={c} style={{ display: "flex", alignItems: "flex-start", gap: 8, marginBottom: 14 }}>
                <span
                  style={{
                    width: 9,
                    height: 9,
                    borderRadius: "50%",
                    background: CLUSTER_COLORS[c % CLUSTER_COLORS.length],
                    display: "inline-block",
                    marginTop: 4,
                    flexShrink: 0,
                  }}
                />
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 13, color: "#E7EAF3", fontWeight: 500 }}>{info.name}</div>
                  <div style={{ fontSize: 11.5, color: "#8891A7", marginTop: 3, lineHeight: 1.5 }}>{info.detail}</div>
                </div>
                <span style={{ color: "#8891A7", fontSize: 13 }}>{analysis.clusterCounts[c] || 0}</span>
              </div>
            ))}
          <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 13, marginTop: 4 }}>
            <span
              style={{
                width: 9,
                height: 9,
                borderRadius: "50%",
                background: OUTLIER_COLOR,
                display: "inline-block",
                animation: "dl-pulse 1.6s ease-in-out infinite",
              }}
            />
            <span style={{ color: "#C7CCDC" }}>Outliers (unusual records)</span>
            <span style={{ marginLeft: "auto", color: "#8891A7" }}>{outlierCount}</span>
          </div>
        </div>

        {analysis && analysis.corr && (
          <div
            style={{
              marginTop: 24,
              padding: "14px 14px",
              background: "#111628",
              border: "1px solid #1B2333",
              borderRadius: 8,
              fontSize: 12.5,
              color: "#B9C0D4",
              lineHeight: 1.6,
            }}
          >
            Strongest relationship: <strong style={{ color: "#E7EAF3" }}>{pretty(analysis.corr.a)}</strong> and{" "}
            <strong style={{ color: "#E7EAF3" }}>{pretty(analysis.corr.b)}</strong> (r = {analysis.corr.r.toFixed(2)})
          </div>
        )}

        <div style={{ fontSize: 11.5, color: "#5B6480", marginTop: 22, lineHeight: 1.6 }}>
          Grouping: k-means on the selected columns (scaled 0 to 1). Group names compare each group's average to the whole
          dataset. Outliers: any value more than 2.5 standard deviations from its column average.
        </div>
      </div>

      {/* Canvas */}
      <div ref={containerRef} style={{ flex: 1, position: "relative" }}>
        <svg ref={svgRef} width={dims.w} height={dims.h} style={{ display: "block" }}>
          {nodes.map((n, i) => {
            const isSel = selectedKey !== null && selectedKey === n._key;
            return (
              <circle
                key={n._key ?? i}
                cx={n.x}
                cy={n.y}
                r={isSel ? 8 : 6}
                fill={n.__outlier ? OUTLIER_COLOR : CLUSTER_COLORS[n.__cluster % CLUSTER_COLORS.length]}
                stroke={isSel ? "#fff" : "none"}
                strokeWidth={2}
                opacity={n.__outlier ? 0.9 : 0.85}
                style={{ cursor: "pointer" }}
                onMouseDown={() => setSelectedKey(n._key)}
              />
            );
          })}
          {clusterLabels.map((l) => (
            <text
              key={l.c}
              x={l.x}
              y={l.y}
              textAnchor="middle"
              fill={CLUSTER_COLORS[l.c % CLUSTER_COLORS.length]}
              fontSize={12}
              fontWeight={600}
              style={{ paintOrder: "stroke", stroke: "#0B0E1A", strokeWidth: 4, strokeLinejoin: "round", pointerEvents: "none" }}
            >
              {l.parts.map((p, idx) => (
                <tspan key={idx} x={l.x} dy={idx === 0 ? 0 : 14}>
                  {p}
                </tspan>
              ))}
            </text>
          ))}
        </svg>
      </div>

      {/* Detail panel */}
      {selectedKey !== null &&
        (() => {
          const rec = nodes.find((n) => n._key === selectedKey) || rawRows.find((r) => r._key === selectedKey);
          if (!rec) return null;
          const clusterName =
            analysis && rec.__cluster !== undefined && analysis.clusters[rec.__cluster]
              ? analysis.clusters[rec.__cluster].name
              : "";
          return (
            <div
              style={{
                width: 280,
                flexShrink: 0,
                borderLeft: "1px solid #1B2333",
                background: "#0D1120",
                padding: "22px 20px",
                overflowY: "auto",
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: 15, fontWeight: 600 }}>Record</div>
                <X size={16} style={{ cursor: "pointer", color: "#8891A7" }} onClick={() => setSelectedKey(null)} />
              </div>
              <div
                style={{
                  marginTop: 6,
                  marginBottom: 16,
                  fontSize: 12.5,
                  lineHeight: 1.5,
                  color: rec.__outlier ? OUTLIER_COLOR : CLUSTER_COLORS[(rec.__cluster ?? 0) % CLUSTER_COLORS.length],
                }}
              >
                {rec.__outlier ? "Outlier (unusual record)" : clusterName}
              </div>
              {Object.keys(rec)
                .filter((key) => !key.startsWith("__") && !["_key", "x", "y", "targetX", "targetY", "vx", "vy", "index"].includes(key))
                .map((key) => (
                  <div
                    key={key}
                    style={{ display: "flex", justifyContent: "space-between", gap: 10, fontSize: 13, padding: "6px 0", borderBottom: "1px solid #1B2333" }}
                  >
                    <span style={{ color: "#8891A7" }}>{key}</span>
                    <span style={{ color: "#E7EAF3", textAlign: "right", wordBreak: "break-word" }}>{String(rec[key])}</span>
                  </div>
                ))}
            </div>
          );
        })()}
    </div>
  );
}

function StatBlock({ label, value, accent }) {
  return (
    <div style={{ flex: 1, background: "#111628", border: "1px solid #1B2333", borderRadius: 8, padding: "10px 12px" }}>
      <div style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: 20, fontWeight: 700, color: accent || "#E7EAF3" }}>
        {value}
      </div>
      <div style={{ fontSize: 11, color: "#8891A7", marginTop: 2 }}>{label}</div>
    </div>
  );
}
