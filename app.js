import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ====================================================================
// Parameters (live-bound to UI)
// ====================================================================
const params = {
  N: 12,
  alpha: 0.0,
  numGeodesics: 24,
  geoLength: Math.PI,
  geoSteps: 160,
  eps: 0.05,
  showCells: true,
  showGeo: true,
  showMarkers: true,
};

// ====================================================================
// Scene
// ====================================================================
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0d10);
const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.01, 200);
camera.position.set(0, 2.4, 7.8);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
scene.add(new THREE.AmbientLight(0xffffff, 0.7));
const sun = new THREE.DirectionalLight(0xffffff, 0.4);
sun.position.set(4, 6, 5);
scene.add(sun);

// ====================================================================
// Two stereographic balls
// ====================================================================
const BALL_R = 1.0;
const BALL_OFFSET = 1.7;
const CENTER_POS = new THREE.Vector3(+BALL_OFFSET, 0, 0); // w >= 0 hemisphere
const CENTER_NEG = new THREE.Vector3(-BALL_OFFSET, 0, 0); // w <  0 hemisphere

function addRefBall(center) {
  const sphere = new THREE.Mesh(
    new THREE.SphereGeometry(BALL_R, 48, 32),
    new THREE.MeshBasicMaterial({ color: 0x4a90c4, wireframe: true, transparent: true, opacity: 0.09 })
  );
  sphere.position.copy(center);
  scene.add(sphere);
  const ringMat = new THREE.LineBasicMaterial({ color: 0x4a90c4, transparent: true, opacity: 0.30 });
  for (const axis of ['x','y','z']) {
    const pts = [];
    for (let i = 0; i <= 96; i++) {
      const t = (i / 96) * Math.PI * 2; const c = Math.cos(t), s = Math.sin(t);
      let p;
      if (axis === 'x') p = new THREE.Vector3(0, c, s);
      else if (axis === 'y') p = new THREE.Vector3(c, 0, s);
      else p = new THREE.Vector3(c, s, 0);
      pts.push(p.multiplyScalar(BALL_R).add(center));
    }
    scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), ringMat));
  }
}
addRefBall(CENTER_POS);
addRefBall(CENTER_NEG);

// ====================================================================
// Tesseract cells (8 cubes)
// ====================================================================
function buildCells() {
  const palette = [
    0xff6b6b, 0xffb84d,  // x = +1, x = -1
    0xfff066, 0xb6f06b,  // y = +1, y = -1
    0x66e0ff, 0x8a8bff,  // z = +1, z = -1
    0xff7bd1, 0xe6e6e6,  // w = +1, w = -1
  ];
  const out = [];
  let idx = 0;
  for (let axis = 0; axis < 4; axis++) {
    for (const sign of [+1, -1]) {
      const free = [0,1,2,3].filter(i => i !== axis);
      const vertices = [];
      for (let mask = 0; mask < 8; mask++) {
        const v = [0,0,0,0];
        v[axis] = sign;
        for (let k = 0; k < 3; k++) v[free[k]] = (mask & (1 << k)) ? 1 : -1;
        vertices.push(v);
      }
      const edges = [];
      for (let i = 0; i < 8; i++)
        for (let j = i + 1; j < 8; j++) {
          let diff = 0;
          for (let k = 0; k < 3; k++) if (((i >> k) & 1) !== ((j >> k) & 1)) diff++;
          if (diff === 1) edges.push([i, j]);
        }
      out.push({ axis, sign, free, vertices, edges, color: palette[idx++] });
    }
  }
  return out;
}
const cells = buildCells();

// ====================================================================
// 4D rotation
// ====================================================================
function ident4() { return [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]; }
function matMul(A, B) {
  const C = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]];
  for (let i = 0; i < 4; i++)
    for (let j = 0; j < 4; j++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += A[i][k] * B[k][j];
      C[i][j] = s;
    }
  return C;
}
function rotPlane(i, j, theta) {
  const M = ident4();
  const c = Math.cos(theta), s = Math.sin(theta);
  M[i][i] = c; M[j][j] = c;
  M[i][j] = -s; M[j][i] = s;
  return M;
}
function apply4(M, v) {
  return [
    M[0][0]*v[0] + M[0][1]*v[1] + M[0][2]*v[2] + M[0][3]*v[3],
    M[1][0]*v[0] + M[1][1]*v[1] + M[1][2]*v[2] + M[1][3]*v[3],
    M[2][0]*v[0] + M[2][1]*v[1] + M[2][2]*v[2] + M[2][3]*v[3],
    M[3][0]*v[0] + M[3][1]*v[1] + M[3][2]*v[2] + M[3][3]*v[3],
  ];
}
let R = ident4();

// ====================================================================
// Projection: rotate by R, then stereographic to one of two unit balls
// ====================================================================
function projectS3(v4) {
  const r = apply4(R, v4);
  const x = r[0], y = r[1], z = r[2], w = r[3];
  if (w >= 0) {
    const k = 1 / (1 + w + 1e-12);
    return { x: x*k + CENTER_POS.x, y: y*k + CENTER_POS.y, z: z*k + CENTER_POS.z, ball: 1 };
  } else {
    const k = 1 / (1 - w + 1e-12);
    return { x: x*k + CENTER_NEG.x, y: y*k + CENTER_NEG.y, z: z*k + CENTER_NEG.z, ball: -1 };
  }
}
function normalize4(v) {
  const l = Math.hypot(v[0], v[1], v[2], v[3]) || 1;
  return [v[0]/l, v[1]/l, v[2]/l, v[3]/l];
}
function projectAndNormalize(p4) { return projectS3(normalize4(p4)); }

// ====================================================================
// Cell line geometry: dense polylines along each edge, broken at w = 0
// ====================================================================
const cellGroup = new THREE.Group();
scene.add(cellGroup);
const SAMPLES_PER_EDGE = 72;
const cellEntries = cells.map((cell) => {
  const maxSegs = cell.edges.length * SAMPLES_PER_EDGE;
  const positions = new Float32Array(maxSegs * 2 * 3);
  const geom = new THREE.BufferGeometry();
  const attr = new THREE.BufferAttribute(positions, 3);
  attr.setUsage(THREE.DynamicDrawUsage);
  geom.setAttribute('position', attr);
  geom.setDrawRange(0, 0);
  const mat = new THREE.LineBasicMaterial({ color: cell.color, transparent: true, opacity: 0.85 });
  const line = new THREE.LineSegments(geom, mat);
  cellGroup.add(line);
  return { cell, line, positions, attr };
});
function rebuildCellGeometry() {
  for (const entry of cellEntries) {
    const { cell, positions, attr } = entry;
    let segIdx = 0;
    for (const [a, b] of cell.edges) {
      const va = cell.vertices[a], vb = cell.vertices[b];
      let prev = null;
      for (let s = 0; s <= SAMPLES_PER_EDGE; s++) {
        const t = s / SAMPLES_PER_EDGE, it = 1 - t;
        const p4 = [va[0]*it + vb[0]*t, va[1]*it + vb[1]*t, va[2]*it + vb[2]*t, va[3]*it + vb[3]*t];
        const cur = projectAndNormalize(p4);
        if (prev !== null && prev.ball === cur.ball) {
          const o = segIdx * 6;
          positions[o + 0] = prev.x; positions[o + 1] = prev.y; positions[o + 2] = prev.z;
          positions[o + 3] = cur.x;  positions[o + 4] = cur.y;  positions[o + 5] = cur.z;
          segIdx++;
        }
        prev = cur;
      }
    }
    attr.needsUpdate = true;
    entry.line.geometry.setDrawRange(0, segIdx * 2);
    entry.line.geometry.computeBoundingSphere();
  }
}

// ====================================================================
// Random S^3 utilities
// ====================================================================
function randn() {
  let u1 = 0, u2 = 0;
  while (u1 === 0) u1 = Math.random();
  while (u2 === 0) u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}
function randomS3() { return normalize4([randn(), randn(), randn(), randn()]); }
function dot4(a, b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]; }

// ====================================================================
// Potential field on the 8 cells (per-voxel scalar; trilinear sampling)
// ====================================================================
let poles = [randomS3(), randomS3(), randomS3()];
let sourcePoint = randomS3();
let potential = null;

function buildPotential() {
  const N = params.N;
  potential = new Float32Array(8 * N * N * N);
  for (let cellId = 0; cellId < 8; cellId++) {
    const axis = cellId >> 1;
    const sign = (cellId & 1) ? -1 : +1;
    const free = [0,1,2,3].filter(a => a !== axis);
    const cellBase = cellId * N * N * N;
    for (let ix = 0; ix < N; ix++) {
      const ux = (ix + 0.5) * 2 / N - 1;
      for (let iy = 0; iy < N; iy++) {
        const uy = (iy + 0.5) * 2 / N - 1;
        for (let iz = 0; iz < N; iz++) {
          const uz = (iz + 0.5) * 2 / N - 1;
          const v = [0,0,0,0];
          v[axis] = sign;
          v[free[0]] = ux; v[free[1]] = uy; v[free[2]] = uz;
          const p = normalize4(v);
          let phi = 0;
          for (const pole of poles) {
            const c = Math.max(-1, Math.min(1, dot4(p, pole)));
            const d = Math.acos(c);
            phi += 1 / (d + params.eps);
          }
          potential[cellBase + ix * N * N + iy * N + iz] = phi;
        }
      }
    }
  }
}

// Sample interpolated phi at any 4D point (not necessarily unit).
// Implements phĩ(p) = phi(p/|p|) via the cube-face chart selected by argmax|p_a|.
function samplePhi(p) {
  let absMax = 0, axis = 0;
  for (let a = 0; a < 4; a++) {
    const av = Math.abs(p[a]);
    if (av > absMax) { absMax = av; axis = a; }
  }
  if (absMax === 0) return 0;
  const sign = p[axis] < 0 ? -1 : +1;
  const cellId = 2 * axis + (sign < 0 ? 1 : 0);
  const N = params.N;
  const inv = 1 / absMax;
  // free axes in canonical order (same as buildCells)
  const free0 = (axis === 0) ? 1 : 0;
  const free1 = (axis <= 1) ? 2 : 1;
  const free2 = (axis <= 2) ? 3 : 2;
  const u0 = p[free0] * inv;
  const u1 = p[free1] * inv;
  const u2 = p[free2] * inv;
  // continuous voxel coords (voxel center j sits at c = j)
  let c0 = (u0 + 1) * N * 0.5 - 0.5;
  let c1 = (u1 + 1) * N * 0.5 - 0.5;
  let c2 = (u2 + 1) * N * 0.5 - 0.5;
  if (c0 < 0) c0 = 0; else if (c0 > N - 1) c0 = N - 1;
  if (c1 < 0) c1 = 0; else if (c1 > N - 1) c1 = N - 1;
  if (c2 < 0) c2 = 0; else if (c2 > N - 1) c2 = N - 1;
  const i0 = Math.floor(c0), i1 = Math.floor(c1), i2 = Math.floor(c2);
  const t0 = c0 - i0, t1 = c1 - i1, t2 = c2 - i2;
  const i0a = Math.min(N - 1, i0 + 1), i1a = Math.min(N - 1, i1 + 1), i2a = Math.min(N - 1, i2 + 1);
  const base = cellId * N * N * N;
  const NN = N * N;
  const w000 = (1-t0)*(1-t1)*(1-t2);
  const w100 = t0*(1-t1)*(1-t2);
  const w010 = (1-t0)*t1*(1-t2);
  const w110 = t0*t1*(1-t2);
  const w001 = (1-t0)*(1-t1)*t2;
  const w101 = t0*(1-t1)*t2;
  const w011 = (1-t0)*t1*t2;
  const w111 = t0*t1*t2;
  return potential[base + i0*NN  + i1*N  + i2]  * w000
       + potential[base + i0a*NN + i1*N  + i2]  * w100
       + potential[base + i0*NN  + i1a*N + i2]  * w010
       + potential[base + i0a*NN + i1a*N + i2]  * w110
       + potential[base + i0*NN  + i1*N  + i2a] * w001
       + potential[base + i0a*NN + i1*N  + i2a] * w101
       + potential[base + i0*NN  + i1a*N + i2a] * w011
       + potential[base + i0a*NN + i1a*N + i2a] * w111;
}

// ====================================================================
// Geodesic ODE on S^3 with conformal factor n = exp(alpha * phi)
// gamma'' = -gamma + ((grad n)_T - gamma'(gamma' . grad n)) / n
// (gamma is unit, gamma' is unit and tangent; s = S^3 arc length)
// ====================================================================
const FD_EPS = 0.004;

function geodesicAccel(g, v, alpha) {
  const phi0 = samplePhi(g);
  const n = Math.exp(alpha * phi0);
  let gx = 0, gy = 0, gz = 0, gw = 0;
  if (alpha !== 0) {
    // central difference of phĩ in R^4 (samplePhi handles radial projection internally)
    const e = FD_EPS;
    const pp = g.slice(), pm = g.slice();
    pp[0] += e; pm[0] -= e; gx = alpha * n * (samplePhi(pp) - samplePhi(pm)) / (2*e); pp[0] = g[0]; pm[0] = g[0];
    pp[1] += e; pm[1] -= e; gy = alpha * n * (samplePhi(pp) - samplePhi(pm)) / (2*e); pp[1] = g[1]; pm[1] = g[1];
    pp[2] += e; pm[2] -= e; gz = alpha * n * (samplePhi(pp) - samplePhi(pm)) / (2*e); pp[2] = g[2]; pm[2] = g[2];
    pp[3] += e; pm[3] -= e; gw = alpha * n * (samplePhi(pp) - samplePhi(pm)) / (2*e);
  }
  // Tangential projection: subtract radial component (already ~0 from radial extension, but
  // numerical drift can leak a tiny bit, so we still subtract it).
  const grad_dot_g = gx*g[0] + gy*g[1] + gz*g[2] + gw*g[3];
  const gtx = gx - grad_dot_g * g[0];
  const gty = gy - grad_dot_g * g[1];
  const gtz = gz - grad_dot_g * g[2];
  const gtw = gw - grad_dot_g * g[3];
  const v_dot_grad = v[0]*gx + v[1]*gy + v[2]*gz + v[3]*gw;
  return [
    -g[0] + (gtx - v[0] * v_dot_grad) / n,
    -g[1] + (gty - v[1] * v_dot_grad) / n,
    -g[2] + (gtz - v[2] * v_dot_grad) / n,
    -g[3] + (gtw - v[3] * v_dot_grad) / n,
  ];
}

function rk4Step(g, v, alpha, ds) {
  const a1 = geodesicAccel(g, v, alpha);
  const g2 = [g[0]+0.5*ds*v[0], g[1]+0.5*ds*v[1], g[2]+0.5*ds*v[2], g[3]+0.5*ds*v[3]];
  const v2 = [v[0]+0.5*ds*a1[0], v[1]+0.5*ds*a1[1], v[2]+0.5*ds*a1[2], v[3]+0.5*ds*a1[3]];
  const a2 = geodesicAccel(g2, v2, alpha);
  const g3 = [g[0]+0.5*ds*v2[0], g[1]+0.5*ds*v2[1], g[2]+0.5*ds*v2[2], g[3]+0.5*ds*v2[3]];
  const v3 = [v[0]+0.5*ds*a2[0], v[1]+0.5*ds*a2[1], v[2]+0.5*ds*a2[2], v[3]+0.5*ds*a2[3]];
  const a3 = geodesicAccel(g3, v3, alpha);
  const g4 = [g[0]+ds*v3[0], g[1]+ds*v3[1], g[2]+ds*v3[2], g[3]+ds*v3[3]];
  const v4 = [v[0]+ds*a3[0], v[1]+ds*a3[1], v[2]+ds*a3[2], v[3]+ds*a3[3]];
  const a4 = geodesicAccel(g4, v4, alpha);
  const gn = [
    g[0] + (ds/6)*(v[0] + 2*v2[0] + 2*v3[0] + v4[0]),
    g[1] + (ds/6)*(v[1] + 2*v2[1] + 2*v3[1] + v4[1]),
    g[2] + (ds/6)*(v[2] + 2*v2[2] + 2*v3[2] + v4[2]),
    g[3] + (ds/6)*(v[3] + 2*v2[3] + 2*v3[3] + v4[3]),
  ];
  const vn = [
    v[0] + (ds/6)*(a1[0] + 2*a2[0] + 2*a3[0] + a4[0]),
    v[1] + (ds/6)*(a1[1] + 2*a2[1] + 2*a3[1] + a4[1]),
    v[2] + (ds/6)*(a1[2] + 2*a2[2] + 2*a3[2] + a4[2]),
    v[3] + (ds/6)*(a1[3] + 2*a2[3] + 2*a3[3] + a4[3]),
  ];
  // Project gn onto S^3 and vn onto T_{gn} S^3
  const gl = Math.hypot(gn[0], gn[1], gn[2], gn[3]) || 1;
  gn[0] /= gl; gn[1] /= gl; gn[2] /= gl; gn[3] /= gl;
  const vd = vn[0]*gn[0] + vn[1]*gn[1] + vn[2]*gn[2] + vn[3]*gn[3];
  vn[0] -= vd*gn[0]; vn[1] -= vd*gn[1]; vn[2] -= vd*gn[2]; vn[3] -= vd*gn[3];
  return [gn, vn];
}

function traceGeodesic(g0, v0, alpha, length, steps) {
  const ds = length / steps;
  const traj = new Array(steps + 1);
  traj[0] = g0.slice();
  let g = g0.slice(), v = v0.slice();
  for (let s = 0; s < steps; s++) {
    [g, v] = rk4Step(g, v, alpha, ds);
    traj[s + 1] = g;
  }
  return traj;
}

// ====================================================================
// Orthonormal frame on T_{γ₀} S^3, and a Fibonacci spread of directions
// ====================================================================
function buildFrame(g) {
  const basis = [g.slice()];
  const candidates = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
  candidates.sort((a, b) => Math.abs(dot4(a, g)) - Math.abs(dot4(b, g)));
  for (let k = 0; k < 3; k++) {
    let v = candidates[k].slice();
    for (const b of basis) {
      const c = dot4(v, b);
      for (let i = 0; i < 4; i++) v[i] -= c * b[i];
    }
    const len = Math.hypot(v[0], v[1], v[2], v[3]);
    if (len < 1e-9) continue;
    for (let i = 0; i < 4; i++) v[i] /= len;
    basis.push(v);
  }
  return basis.slice(1);
}

function fibSphere(K) {
  const dirs = [];
  const golden = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < K; i++) {
    const y = K === 1 ? 0 : 1 - (i / (K - 1)) * 2;
    const r = Math.sqrt(Math.max(0, 1 - y * y));
    const theta = golden * i;
    dirs.push([Math.cos(theta) * r, y, Math.sin(theta) * r]);
  }
  return dirs;
}

// ====================================================================
// Geodesic visualization
// ====================================================================
let trajectories = []; // array of arrays of 4D unit vectors
const geodesicGroup = new THREE.Group();
scene.add(geodesicGroup);
let geodesicLines = [];

function rebuildTrajectories() {
  const frame = buildFrame(sourcePoint);
  const dirs = fibSphere(params.numGeodesics);
  const out = [];
  for (let i = 0; i < dirs.length; i++) {
    const d = dirs[i];
    const v0 = [0,0,0,0];
    for (let a = 0; a < 4; a++) v0[a] = d[0]*frame[0][a] + d[1]*frame[1][a] + d[2]*frame[2][a];
    out.push(traceGeodesic(sourcePoint, v0, params.alpha, params.geoLength, params.geoSteps));
  }
  trajectories = out;
  rebuildGeodesicLineMeshes();
}

function rebuildGeodesicLineMeshes() {
  for (const e of geodesicLines) {
    geodesicGroup.remove(e.line);
    e.line.geometry.dispose();
    e.line.material.dispose();
  }
  geodesicLines = [];
  for (let i = 0; i < trajectories.length; i++) {
    const traj = trajectories[i];
    const maxSegs = traj.length;
    const positions = new Float32Array(maxSegs * 2 * 3);
    const geom = new THREE.BufferGeometry();
    const attr = new THREE.BufferAttribute(positions, 3);
    attr.setUsage(THREE.DynamicDrawUsage);
    geom.setAttribute('position', attr);
    geom.setDrawRange(0, 0);
    const color = new THREE.Color().setHSL(i / Math.max(1, trajectories.length), 0.65, 0.62);
    const mat = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.85 });
    const line = new THREE.LineSegments(geom, mat);
    geodesicGroup.add(line);
    geodesicLines.push({ line, positions, attr });
  }
}

function reprojectGeodesics() {
  for (let i = 0; i < geodesicLines.length; i++) {
    const traj = trajectories[i];
    const e = geodesicLines[i];
    let segIdx = 0;
    let prev = null;
    for (let k = 0; k < traj.length; k++) {
      const proj = projectS3(traj[k]);
      if (prev !== null && prev.ball === proj.ball) {
        const o = segIdx * 6;
        e.positions[o + 0] = prev.x; e.positions[o + 1] = prev.y; e.positions[o + 2] = prev.z;
        e.positions[o + 3] = proj.x; e.positions[o + 4] = proj.y; e.positions[o + 5] = proj.z;
        segIdx++;
      }
      prev = proj;
    }
    e.attr.needsUpdate = true;
    e.line.geometry.setDrawRange(0, segIdx * 2);
    e.line.geometry.computeBoundingSphere();
  }
}

// ====================================================================
// Pole and source markers
// ====================================================================
const markerGroup = new THREE.Group();
scene.add(markerGroup);
let poleMeshes = [];
let sourceMesh = null;

function buildMarkers() {
  while (markerGroup.children.length) {
    const c = markerGroup.children.pop();
    c.geometry?.dispose();
    c.material?.dispose();
  }
  poleMeshes = [];
  for (let i = 0; i < poles.length; i++) {
    const m = new THREE.Mesh(
      new THREE.SphereGeometry(0.045, 18, 14),
      new THREE.MeshBasicMaterial({ color: 0xff5566 })
    );
    markerGroup.add(m);
    poleMeshes.push(m);
  }
  sourceMesh = new THREE.Mesh(
    new THREE.SphereGeometry(0.055, 18, 14),
    new THREE.MeshBasicMaterial({ color: 0xffd166 })
  );
  markerGroup.add(sourceMesh);
}

function repositionMarkers() {
  for (let i = 0; i < poles.length; i++) {
    const proj = projectS3(poles[i]);
    poleMeshes[i].position.set(proj.x, proj.y, proj.z);
  }
  const sp = projectS3(sourcePoint);
  sourceMesh.position.set(sp.x, sp.y, sp.z);
}

// ====================================================================
// Dirty flags & main loop
// ====================================================================
let dirtyPotential = true;
let dirtyGeodesics = true;
let dirtyCells = true;
let dirtyReproject = true;

function rebuildAll() {
  if (dirtyPotential) { buildPotential(); dirtyPotential = false; dirtyGeodesics = true; }
  if (dirtyGeodesics) { rebuildTrajectories(); dirtyGeodesics = false; dirtyReproject = true; }
}

buildMarkers();
rebuildAll();
rebuildCellGeometry();
repositionMarkers();

// ====================================================================
// Keyboard: 4D rotations
// ====================================================================
const keys = Object.create(null);
window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT') return;
  const k = e.key.toLowerCase();
  keys[k] = true;
  if (k === 'r') { R = ident4(); dirtyCells = true; dirtyReproject = true; }
});
window.addEventListener('keyup', (e) => { keys[e.key.toLowerCase()] = false; });
window.addEventListener('blur', () => { for (const k in keys) keys[k] = false; });

const RATE = 0.9;
let last = performance.now();
function tick(now) {
  requestAnimationFrame(tick);
  const dt = Math.min(0.05, (now - last) / 1000);
  last = now;
  const dir = keys['shift'] ? -1 : 1;
  const dtheta = RATE * dt * dir;
  let rotated = false;
  if (keys['x']) { R = matMul(rotPlane(0, 3, dtheta), R); rotated = true; }
  if (keys['y']) { R = matMul(rotPlane(1, 3, dtheta), R); rotated = true; }
  if (keys['z']) { R = matMul(rotPlane(2, 3, dtheta), R); rotated = true; }
  if (keys['w']) { R = matMul(rotPlane(0, 1, dtheta), R); rotated = true; }
  if (rotated) { dirtyCells = true; dirtyReproject = true; }

  if (dirtyPotential || dirtyGeodesics) rebuildAll();
  if (dirtyCells) { rebuildCellGeometry(); dirtyCells = false; }
  if (dirtyReproject) { reprojectGeodesics(); repositionMarkers(); dirtyReproject = false; }

  cellGroup.visible = params.showCells;
  geodesicGroup.visible = params.showGeo;
  markerGroup.visible = params.showMarkers;

  controls.update();
  renderer.render(scene, camera);
}
requestAnimationFrame((t) => { last = t; tick(t); });

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// ====================================================================
// UI hookup
// ====================================================================
function bindRange(id, valId, fmt, onChange) {
  const el = document.getElementById(id);
  const lab = document.getElementById(valId);
  const update = () => {
    const x = parseFloat(el.value);
    lab.textContent = fmt(x);
    onChange(x);
  };
  el.addEventListener('input', update);
  update();
}

bindRange('alpha', 'alpha-v', x => x.toFixed(2), (x) => { params.alpha = x; dirtyGeodesics = true; });
bindRange('numGeo', 'numGeo-v', x => String(x|0), (x) => { params.numGeodesics = x|0; dirtyGeodesics = true; });
bindRange('geoLen', 'geoLen-v', x => x.toFixed(2), (x) => { params.geoLength = x * Math.PI; dirtyGeodesics = true; });
bindRange('geoSteps', 'geoSteps-v', x => String(x|0), (x) => { params.geoSteps = x|0; dirtyGeodesics = true; });
bindRange('voxN', 'voxN-v', x => String(x|0), (x) => { params.N = x|0; dirtyPotential = true; });
bindRange('eps', 'eps-v', x => x.toFixed(3), (x) => { params.eps = x; dirtyPotential = true; });

document.getElementById('reroll-poles').addEventListener('click', () => {
  poles = [randomS3(), randomS3(), randomS3()];
  buildMarkers();
  dirtyPotential = true;
});
document.getElementById('reroll-source').addEventListener('click', () => {
  sourcePoint = randomS3();
  dirtyGeodesics = true;
});
document.getElementById('show-cells').addEventListener('change', (e) => { params.showCells = e.target.checked; });
document.getElementById('show-geo').addEventListener('change', (e) => { params.showGeo = e.target.checked; });
document.getElementById('show-mark').addEventListener('change', (e) => { params.showMarkers = e.target.checked; });
