import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ====================================================================
// Parameters (live-bound to UI)
// ====================================================================
const params = {
  N: 32,
  alpha: 0.0,
  numGeodesics: 24,
  geoLength: Math.PI,
  geoSteps: 160,
  eps: 1e-4,
  kernel: 'inv',     // 'inv' = 1/(eps+d); 'lor' = d/(eps+d^2)
  sourceMode: 'current', // 'point' = phi = kernel(d_min); 'current' = ∫ kernel(d(p,q))·(1-cosθ)·|ds|
  currentSamples: 192, // number of curve samples used for the current integral
  showCells: false,
  showGeo: false,
  showMarkers: true,
};

const orbitParams = {
  enabled: true,
  distance: 0.30,
  orthVel: 0.30,
};

function kernelPhi(d, eps) {
  if (params.kernel === 'lor') return d / (eps + d * d);
  return 1 / (eps + d);
}

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
// Circle source on S^3 (fixed center, fixed 2-plane, UI radius)
//   gamma(theta) = cos(r) * center + sin(r) * (cos(theta) u1 + sin(theta) u2)
// Closed-form S^3 distance from p to gamma:
//   d(p, gamma) = arccos( cos(r) (p . c) + sin(r) sqrt((p . u1)^2 + (p . u2)^2) )
// ====================================================================
const circleSource = {
  center: [0, 0, 0, 1],
  u1:     [1, 0, 0, 0],
  u2:     [0, 1, 0, 0],
  radius: 0.50 * Math.PI,
};

let sourcePoint = randomS3();
let potential = null;

function distToCircle(p) {
  const c = circleSource;
  const a = p[0]*c.center[0] + p[1]*c.center[1] + p[2]*c.center[2] + p[3]*c.center[3];
  const b = p[0]*c.u1[0]     + p[1]*c.u1[1]     + p[2]*c.u1[2]     + p[3]*c.u1[3];
  const e = p[0]*c.u2[0]     + p[1]*c.u2[1]     + p[2]*c.u2[2]     + p[3]*c.u2[3];
  let m = Math.cos(c.radius) * a + Math.sin(c.radius) * Math.sqrt(b*b + e*e);
  if (m > 1) m = 1; else if (m < -1) m = -1;
  return Math.acos(m);
}

function buildPotentialPoint() {
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
          const d = distToCircle(p);
          potential[cellBase + ix * N * N + iy * N + iz] = kernelPhi(d, params.eps);
        }
      }
    }
  }
}

// Current-source potential: discretize the circle into M tangent elements
// and integrate phi(q) = sum_m kernel(d(p_m, q)) * (1 - cos(angle(ds_m, p_m->q))) * |ds|
// where p_m->q is the initial unit tangent at p_m of the great-circle geodesic to q.
function buildPotentialCurrent() {
  const N = params.N;
  const M = Math.max(8, params.currentSamples|0);
  potential = new Float32Array(8 * N * N * N);
  const cs = circleSource;
  const cosRc = Math.cos(cs.radius), sinRc = Math.sin(cs.radius);
  const elemLen = sinRc * (2 * Math.PI / M); // |dp/dtheta| = sinRc for unit u1, u2
  const ps = new Float32Array(M * 4);
  const dsd = new Float32Array(M * 4);
  for (let m = 0; m < M; m++) {
    const theta = (m / M) * 2 * Math.PI;
    const ct = Math.cos(theta), st = Math.sin(theta);
    for (let i = 0; i < 4; i++) {
      ps[m*4 + i]  = cosRc*cs.center[i] + sinRc*(ct*cs.u1[i] + st*cs.u2[i]);
      dsd[m*4 + i] = -st*cs.u1[i] + ct*cs.u2[i];
    }
  }
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
          const q = normalize4(v);
          let phi = 0;
          for (let m = 0; m < M; m++) {
            const i4 = m * 4;
            const px = ps[i4], py = ps[i4+1], pz = ps[i4+2], pw = ps[i4+3];
            const dx = dsd[i4], dy = dsd[i4+1], dz = dsd[i4+2], dw = dsd[i4+3];
            let cosD = px*q[0] + py*q[1] + pz*q[2] + pw*q[3];
            if (cosD > 1) cosD = 1; else if (cosD < -1) cosD = -1;
            const d = Math.acos(cosD);
            const sinD = Math.sqrt(Math.max(0, 1 - cosD*cosD));
            let cosA = 0;
            if (sinD > 1e-9) {
              const ex = (q[0] - cosD*px) / sinD;
              const ey = (q[1] - cosD*py) / sinD;
              const ez = (q[2] - cosD*pz) / sinD;
              const ew = (q[3] - cosD*pw) / sinD;
              cosA = dx*ex + dy*ey + dz*ez + dw*ew;
            }
            phi += kernelPhi(d, params.eps) * (1 - cosA) * elemLen;
          }
          potential[cellBase + ix * N * N + iy * N + iz] = phi;
        }
      }
    }
  }
}

function buildPotential() {
  if (params.sourceMode === 'current') buildPotentialCurrent();
  else buildPotentialPoint();
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
// Circle-source visualization, test-source marker
// ====================================================================
const markerGroup = new THREE.Group();
scene.add(markerGroup);
const CIRCLE_SAMPLES = 256;
let circleSamples = []; // 4D unit vectors around the circle
let circleLine = null;  // LineSegments, rebuilt when CIRCLE_SAMPLES changes
let circleCenterMesh = null;
let sourceMesh = null;

function rebuildCircleSamples() {
  const c = circleSource;
  const cosR = Math.cos(c.radius), sinR = Math.sin(c.radius);
  circleSamples = new Array(CIRCLE_SAMPLES);
  for (let i = 0; i < CIRCLE_SAMPLES; i++) {
    const theta = (i / CIRCLE_SAMPLES) * Math.PI * 2;
    const ct = Math.cos(theta), st = Math.sin(theta);
    circleSamples[i] = [
      cosR*c.center[0] + sinR*(ct*c.u1[0] + st*c.u2[0]),
      cosR*c.center[1] + sinR*(ct*c.u1[1] + st*c.u2[1]),
      cosR*c.center[2] + sinR*(ct*c.u1[2] + st*c.u2[2]),
      cosR*c.center[3] + sinR*(ct*c.u1[3] + st*c.u2[3]),
    ];
  }
}

function buildMarkers() {
  while (markerGroup.children.length) {
    const c = markerGroup.children.pop();
    c.geometry?.dispose();
    c.material?.dispose();
  }
  // fixed-center dot
  circleCenterMesh = new THREE.Mesh(
    new THREE.SphereGeometry(0.035, 16, 12),
    new THREE.MeshBasicMaterial({ color: 0xff5566, transparent: true, opacity: 0.7 })
  );
  markerGroup.add(circleCenterMesh);
  // circle line (LineSegments; breaks at w = 0)
  const maxSegs = CIRCLE_SAMPLES;
  const positions = new Float32Array(maxSegs * 2 * 3);
  const geom = new THREE.BufferGeometry();
  const attr = new THREE.BufferAttribute(positions, 3);
  attr.setUsage(THREE.DynamicDrawUsage);
  geom.setAttribute('position', attr);
  geom.setDrawRange(0, 0);
  const mat = new THREE.LineBasicMaterial({ color: 0xff5566, transparent: true, opacity: 0.95 });
  circleLine = new THREE.LineSegments(geom, mat);
  circleLine.userData = { positions, attr };
  markerGroup.add(circleLine);
  sourceMesh = null;
}

function reprojectCircle() {
  const { positions, attr } = circleLine.userData;
  const N = circleSamples.length;
  let segIdx = 0;
  let prev = projectS3(circleSamples[0]);
  for (let i = 1; i <= N; i++) {
    const cur = projectS3(circleSamples[i % N]);
    if (prev.ball === cur.ball) {
      const o = segIdx * 6;
      positions[o + 0] = prev.x; positions[o + 1] = prev.y; positions[o + 2] = prev.z;
      positions[o + 3] = cur.x;  positions[o + 4] = cur.y;  positions[o + 5] = cur.z;
      segIdx++;
    }
    prev = cur;
  }
  attr.needsUpdate = true;
  circleLine.geometry.setDrawRange(0, segIdx * 2);
  circleLine.geometry.computeBoundingSphere();
}

function repositionMarkers() {
  const cp = projectS3(circleSource.center);
  circleCenterMesh.position.set(cp.x, cp.y, cp.z);
  reprojectCircle();
}

// ====================================================================
// Spaceship simulation
// --------------------------------------------------------------------
// Interpret the refractive S^3 as a (1+2)-D positive-definite "spacetime":
// the ship has a frame (v_S, X_S, Y_S) orthonormal in T_{gamma_S} S^3, with
// v_S as its time direction. We propagate the ship along its refractive
// geodesic and parallel-transport the spatial axes along it (round-S^3
// rule du/ds = -(gamma' . u) gamma -- exact on round S^3, an approximation
// under refraction). Test objects are placed at grid points in the ship's
// initial (X_S, Y_S)-plane and given v_S parallel-transported along the
// spatial geodesic to their location. Each object then propagates along
// its own refractive geodesic by an amount dT = tau * dt of its own arc
// length, where tau = v_S . v_i (ambient R^4) is the velocity dot product.
// The view (phi, r) is read off by projecting gamma_i orthogonally to v_S
// (round-S^3 spatial slicing) and inverting the round-S^3 exp at gamma_S.
// This is a simplification of the spec's linear-estimate + grid-search:
// in the small-dt limit the two coincide.
// ====================================================================
const simParams = {
  extent: 0.10,
  N: 9,
  dt: 0.05,
  steps: 120,
  showSim: true,
  viewSize: 0.50,
  heliocentric: true,
};
let shipState = null;       // {gamma, v, X, Y}; null = need init
let simHistory = null;      // result of last runShipSim()

function randomTangent(g) {
  for (;;) {
    let v = [randn(), randn(), randn(), randn()];
    const c = dot4(v, g);
    for (let i = 0; i < 4; i++) v[i] -= c * g[i];
    const len = Math.hypot(v[0], v[1], v[2], v[3]);
    if (len > 1e-6) {
      for (let i = 0; i < 4; i++) v[i] /= len;
      return v;
    }
  }
}

function makeShip(gamma, vDir) {
  // Build orthonormal X, Y orthogonal to (gamma, v).
  const basis = [gamma, vDir];
  const candidates = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
  candidates.sort((a, b) => (Math.abs(dot4(a,gamma)) + Math.abs(dot4(a,vDir))) - (Math.abs(dot4(b,gamma)) + Math.abs(dot4(b,vDir))));
  const tangentBasis = [];
  for (let k = 0; k < 4 && tangentBasis.length < 2; k++) {
    let u = candidates[k].slice();
    for (const b of basis) { const c = dot4(u, b); for (let i = 0; i < 4; i++) u[i] -= c * b[i]; }
    for (const r of tangentBasis) { const c = dot4(u, r); for (let i = 0; i < 4; i++) u[i] -= c * r[i]; }
    const len = Math.hypot(u[0], u[1], u[2], u[3]);
    if (len < 1e-6) continue;
    for (let i = 0; i < 4; i++) u[i] /= len;
    tangentBasis.push(u);
  }
  return { gamma: gamma.slice(), v: vDir.slice(), X: tangentBasis[0], Y: tangentBasis[1] };
}

function randomShip() {
  const g = sourcePoint.slice();
  const v = randomTangent(g);
  return makeShip(g, v);
}

// Place the ship at distance d from the circle source, in a direction
// perpendicular to the source. At the ship location its velocity is
//   v_S = sqrt(1 - b^2) * e_para + b * e_pol
// where e_para is the parallel-transported source tangent and e_pol is
// the remaining orthonormal direction in T_{gamma_ship} S^3 (perpendicular
// to the geodesic that took the ship there). The radial component is 0.
// For round S^3, b = sin(d) gives a circular orbit at radius d.
function orbitShip(d, b) {
  const cs = circleSource;
  const theta0 = 0;
  const cosT0 = Math.cos(theta0), sinT0 = Math.sin(theta0);
  const cosRc = Math.cos(cs.radius), sinRc = Math.sin(cs.radius);
  const g_src = [
    cosRc*cs.center[0] + sinRc*(cosT0*cs.u1[0] + sinT0*cs.u2[0]),
    cosRc*cs.center[1] + sinRc*(cosT0*cs.u1[1] + sinT0*cs.u2[1]),
    cosRc*cs.center[2] + sinRc*(cosT0*cs.u1[2] + sinT0*cs.u2[2]),
    cosRc*cs.center[3] + sinRc*(cosT0*cs.u1[3] + sinT0*cs.u2[3]),
  ];
  const e_para = [
    -sinT0*cs.u1[0] + cosT0*cs.u2[0],
    -sinT0*cs.u1[1] + cosT0*cs.u2[1],
    -sinT0*cs.u1[2] + cosT0*cs.u2[2],
    -sinT0*cs.u1[3] + cosT0*cs.u2[3],
  ];
  // Gram-Schmidt of standard basis vectors against (g_src, e_para)
  // gives an out-of-plane direction u_perp at the source.
  const candidates = [[0,0,1,0], [0,0,0,1], [1,0,0,0], [0,1,0,0]];
  function orthAgainst(v, basis) {
    for (const b of basis) {
      const c = dot4(v, b);
      for (let i = 0; i < 4; i++) v[i] -= c * b[i];
    }
    const l = Math.hypot(v[0], v[1], v[2], v[3]);
    if (l < 1e-6) return null;
    return [v[0]/l, v[1]/l, v[2]/l, v[3]/l];
  }
  let u_perp = null;
  for (const cand of candidates) {
    const u = orthAgainst(cand.slice(), [g_src, e_para]);
    if (u) { u_perp = u; break; }
  }
  if (!u_perp) u_perp = [0,0,1,0];
  const cd = Math.cos(d), sd = Math.sin(d);
  const g_ship = [
    cd*g_src[0] + sd*u_perp[0],
    cd*g_src[1] + sd*u_perp[1],
    cd*g_src[2] + sd*u_perp[2],
    cd*g_src[3] + sd*u_perp[3],
  ];
  // e_para is perpendicular to g_src, u_perp and the great-circle tangent
  // throughout, so its parallel transport along the displacement geodesic
  // is itself.
  // Radial direction at the ship.
  const e_rad = [
    -sd*g_src[0] + cd*u_perp[0],
    -sd*g_src[1] + cd*u_perp[1],
    -sd*g_src[2] + cd*u_perp[2],
    -sd*g_src[3] + cd*u_perp[3],
  ];
  // e_pol = the remaining unit vector orthogonal to (g_ship, e_para, e_rad).
  let e_pol = null;
  for (const cand of candidates) {
    const u = orthAgainst(cand.slice(), [g_ship, e_para, e_rad]);
    if (u) { e_pol = u; break; }
  }
  if (!e_pol) e_pol = [0,0,0,1];
  const bc = Math.max(-1, Math.min(1, b));
  const a = Math.sqrt(Math.max(0, 1 - bc*bc));
  const v_S = [
    a*e_para[0] + bc*e_pol[0],
    a*e_para[1] + bc*e_pol[1],
    a*e_para[2] + bc*e_pol[2],
    a*e_para[3] + bc*e_pol[3],
  ];
  return makeShip(g_ship, v_S);
}

function initialShip() {
  if (orbitParams.enabled) return orbitShip(orbitParams.distance, orbitParams.orthVel);
  return randomShip();
}

// RK4 step that integrates one geodesic (gamma, v_geo) under the refractive
// metric AND parallel-transports two extra tangent vectors A, B along it.
// Returns [gamma', v_geo', A', B']. Pass dummy zero vectors to ignore A/B.
function rk4StepFrame(gamma, vgeo, A, B, alpha, ds) {
  const accel = (g, v) => geodesicAccel(g, v, alpha);
  const ptDer = (g, v, u) => {
    const c = v[0]*u[0] + v[1]*u[1] + v[2]*u[2] + v[3]*u[3];
    return [-c*g[0], -c*g[1], -c*g[2], -c*g[3]];
  };
  const deriv = (g, v, a, b) => [v.slice(), accel(g, v), ptDer(g, v, a), ptDer(g, v, b)];
  const step = (s, k) => [
    [gamma[0]+s*k[0][0], gamma[1]+s*k[0][1], gamma[2]+s*k[0][2], gamma[3]+s*k[0][3]],
    [vgeo[0]+s*k[1][0], vgeo[1]+s*k[1][1], vgeo[2]+s*k[1][2], vgeo[3]+s*k[1][3]],
    [A[0]+s*k[2][0], A[1]+s*k[2][1], A[2]+s*k[2][2], A[3]+s*k[2][3]],
    [B[0]+s*k[3][0], B[1]+s*k[3][1], B[2]+s*k[3][2], B[3]+s*k[3][3]],
  ];
  const k1 = deriv(gamma, vgeo, A, B);
  const s2 = step(0.5*ds, k1); const k2 = deriv(s2[0], s2[1], s2[2], s2[3]);
  const s3 = step(0.5*ds, k2); const k3 = deriv(s3[0], s3[1], s3[2], s3[3]);
  const s4 = step(ds, k3);     const k4 = deriv(s4[0], s4[1], s4[2], s4[3]);
  const out = [gamma.slice(), vgeo.slice(), A.slice(), B.slice()];
  for (let q = 0; q < 4; q++) for (let i = 0; i < 4; i++)
    out[q][i] += (ds/6) * (k1[q][i] + 2*k2[q][i] + 2*k3[q][i] + k4[q][i]);
  // Project onto S^3 and tangents.
  const gl = Math.hypot(out[0][0], out[0][1], out[0][2], out[0][3]) || 1;
  for (let i = 0; i < 4; i++) out[0][i] /= gl;
  const projTangent = (u) => {
    const d = u[0]*out[0][0] + u[1]*out[0][1] + u[2]*out[0][2] + u[3]*out[0][3];
    u[0] -= d*out[0][0]; u[1] -= d*out[0][1]; u[2] -= d*out[0][2]; u[3] -= d*out[0][3];
  };
  projTangent(out[1]); projTangent(out[2]); projTangent(out[3]);
  return out;
}

// Trace ship by ds_total, integrating frame transport, with substeps.
function traceShipFrame(ship, ds_total, subSteps = 4) {
  let g = ship.gamma, v = ship.v, X = ship.X, Y = ship.Y;
  const ds = ds_total / subSteps;
  for (let s = 0; s < subSteps; s++) {
    [g, v, X, Y] = rk4StepFrame(g, v, X, Y, params.alpha, ds);
  }
  // Reorthonormalize the frame against v.
  const vsq = v[0]*v[0] + v[1]*v[1] + v[2]*v[2] + v[3]*v[3];
  let d = (X[0]*v[0] + X[1]*v[1] + X[2]*v[2] + X[3]*v[3]) / vsq;
  X = [X[0]-d*v[0], X[1]-d*v[1], X[2]-d*v[2], X[3]-d*v[3]];
  const Xl = Math.hypot(X[0], X[1], X[2], X[3]) || 1;
  X = [X[0]/Xl, X[1]/Xl, X[2]/Xl, X[3]/Xl];
  d = (Y[0]*v[0] + Y[1]*v[1] + Y[2]*v[2] + Y[3]*v[3]) / vsq;
  Y = [Y[0]-d*v[0], Y[1]-d*v[1], Y[2]-d*v[2], Y[3]-d*v[3]];
  d = Y[0]*X[0] + Y[1]*X[1] + Y[2]*X[2] + Y[3]*X[3];
  Y = [Y[0]-d*X[0], Y[1]-d*X[1], Y[2]-d*X[2], Y[3]-d*X[3]];
  const Yl = Math.hypot(Y[0], Y[1], Y[2], Y[3]) || 1;
  Y = [Y[0]/Yl, Y[1]/Yl, Y[2]/Yl, Y[3]/Yl];
  return { gamma: g, v, X, Y };
}

// Place a test object: shoot a refractive spatial geodesic of length r in
// direction cos(phi) X + sin(phi) Y, parallel-transporting v_S along it.
function placeTestObject(ship, x, y) {
  const r = Math.hypot(x, y);
  if (r < 1e-9) return { gamma: ship.gamma.slice(), v: ship.v.slice(), T: 0, alive: true, x0: x, y0: y };
  const cph = x / r, sph = y / r;
  const dir = [
    cph*ship.X[0] + sph*ship.Y[0], cph*ship.X[1] + sph*ship.Y[1],
    cph*ship.X[2] + sph*ship.Y[2], cph*ship.X[3] + sph*ship.Y[3],
  ];
  const subSteps = Math.max(8, Math.ceil(r * 32));
  let g = ship.gamma.slice(), vgeo = dir, u = ship.v.slice(), dummy = [0,0,0,0];
  const ds = r / subSteps;
  for (let s = 0; s < subSteps; s++) {
    [g, vgeo, u, dummy] = rk4StepFrame(g, vgeo, u, dummy, params.alpha, ds);
  }
  // Renormalize the transported velocity.
  const ul = Math.hypot(u[0], u[1], u[2], u[3]) || 1;
  for (let i = 0; i < 4; i++) u[i] /= ul;
  return { gamma: g, v: u, T: 0, alive: true, x0: x, y0: y };
}

// (phi, r) of a point g on S^3 as seen by the ship: orthogonal projection
// onto v_S^perp in ambient R^4, then round-S^3 inverse-exp at gamma_S.
function viewPoint(g, ship) {
  const v = ship.v, gS = ship.gamma, X = ship.X, Y = ship.Y;
  const vsq = v[0]*v[0] + v[1]*v[1] + v[2]*v[2] + v[3]*v[3];
  const d = (g[0]*v[0] + g[1]*v[1] + g[2]*v[2] + g[3]*v[3]) / vsq;
  const p = [g[0]-d*v[0], g[1]-d*v[1], g[2]-d*v[2], g[3]-d*v[3]];
  const pl = Math.hypot(p[0], p[1], p[2], p[3]);
  if (pl < 1e-9) return { r: 0, phi: 0 };
  const pn = [p[0]/pl, p[1]/pl, p[2]/pl, p[3]/pl];
  const cosR = Math.max(-1, Math.min(1, pn[0]*gS[0] + pn[1]*gS[1] + pn[2]*gS[2] + pn[3]*gS[3]));
  const r = Math.acos(cosR);
  if (r < 1e-6) return { r: 0, phi: 0 };
  const sinR = Math.sin(r);
  const ux = (pn[0] - cosR*gS[0]) / sinR;
  const uy = (pn[1] - cosR*gS[1]) / sinR;
  const uz = (pn[2] - cosR*gS[2]) / sinR;
  const uw = (pn[3] - cosR*gS[3]) / sinR;
  const xc = ux*X[0] + uy*X[1] + uz*X[2] + uw*X[3];
  const yc = ux*Y[0] + uy*Y[1] + uz*Y[2] + uw*Y[3];
  return { r, phi: Math.atan2(yc, xc) };
}

function viewCoords(g_i, v_i, ship) {
  const { r, phi } = viewPoint(g_i, ship);
  const v = ship.v;
  const vsq = v[0]*v[0] + v[1]*v[1] + v[2]*v[2] + v[3]*v[3];
  const vi_len = Math.sqrt(v_i[0]*v_i[0] + v_i[1]*v_i[1] + v_i[2]*v_i[2] + v_i[3]*v_i[3]) || 1;
  const v_len = Math.sqrt(vsq) || 1;
  const tau = (v[0]*v_i[0] + v[1]*v_i[1] + v[2]*v_i[2] + v[3]*v_i[3]) / (v_len * vi_len);
  return { r, phi, tau };
}

// Build a heliocentric frame attached to the source point closest to g_ship.
// e_para is the source curve's unit tangent there; (e_perp_a, e_perp_b) span
// the 2D subspace of T_{gamma_src} S^3 perpendicular to the source curve.
function buildHelioFrame(g_ship) {
  const cs = circleSource;
  const px = dot4(g_ship, cs.u1);
  const py = dot4(g_ship, cs.u2);
  const theta = Math.atan2(py, px);
  const ct = Math.cos(theta), st = Math.sin(theta);
  const cR = Math.cos(cs.radius), sR = Math.sin(cs.radius);
  const g_src = [
    cR*cs.center[0] + sR*(ct*cs.u1[0] + st*cs.u2[0]),
    cR*cs.center[1] + sR*(ct*cs.u1[1] + st*cs.u2[1]),
    cR*cs.center[2] + sR*(ct*cs.u1[2] + st*cs.u2[2]),
    cR*cs.center[3] + sR*(ct*cs.u1[3] + st*cs.u2[3]),
  ];
  const e_para = [
    -st*cs.u1[0] + ct*cs.u2[0],
    -st*cs.u1[1] + ct*cs.u2[1],
    -st*cs.u1[2] + ct*cs.u2[2],
    -st*cs.u1[3] + ct*cs.u2[3],
  ];
  const cands = [[0,0,1,0], [0,0,0,1], [1,0,0,0], [0,1,0,0]];
  const gs = (v, basis) => {
    for (const b of basis) { const c = dot4(v, b); for (let i = 0; i < 4; i++) v[i] -= c * b[i]; }
    const l = Math.hypot(v[0], v[1], v[2], v[3]);
    if (l < 1e-6) return null;
    return [v[0]/l, v[1]/l, v[2]/l, v[3]/l];
  };
  let e_pa = null, e_pb = null;
  for (const c of cands) { e_pa = gs(c.slice(), [g_src, e_para]); if (e_pa) break; }
  for (const c of cands) { e_pb = gs(c.slice(), [g_src, e_para, e_pa]); if (e_pb) break; }
  return { g_src, e_para, e_pa, e_pb };
}

// Project a 4D point through the helio frame: round-S^3 inverse-exp at
// gamma_src, then 2D coordinates in (e_pa, e_pb). gamma_src itself lands at
// the origin; points along the source tangent line collapse there too.
function helioProject(p, frame) {
  const dot_pg = dot4(p, frame.g_src);
  const cosR = dot_pg > 1 ? 1 : (dot_pg < -1 ? -1 : dot_pg);
  const r = Math.acos(cosR);
  if (r < 1e-9) return { x: 0, y: 0, r: 0 };
  const sinR = Math.sin(r);
  const ux = (p[0] - cosR * frame.g_src[0]) / sinR;
  const uy = (p[1] - cosR * frame.g_src[1]) / sinR;
  const uz = (p[2] - cosR * frame.g_src[2]) / sinR;
  const uw = (p[3] - cosR * frame.g_src[3]) / sinR;
  const xa = ux*frame.e_pa[0] + uy*frame.e_pa[1] + uz*frame.e_pa[2] + uw*frame.e_pa[3];
  const xb = ux*frame.e_pb[0] + uy*frame.e_pb[1] + uz*frame.e_pb[2] + uw*frame.e_pb[3];
  return { x: r * xa, y: r * xb, r };
}

function runShipSim() {
  if (!shipState) shipState = initialShip();
  const dt = simParams.dt;
  const nSteps = simParams.steps;
  // Initial test object grid
  const N = simParams.N;
  const ext = simParams.extent;
  const objs = [];
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const u = N === 1 ? 0 : -1 + (2 * i) / (N - 1);
      const w = N === 1 ? 0 : -1 + (2 * j) / (N - 1);
      const x = u * ext, y = w * ext;
      objs.push(placeTestObject(shipState, x, y));
    }
  }
  // Snapshot initial states (objs is mutated in place during integration).
  const objs0 = objs.map(o => ({ x0: o.x0, y0: o.y0, gamma: o.gamma.slice(), v: o.v.slice() }));
  // Allocate per-object history.
  const objHistory = objs.map(o => ({ x0: o.x0, y0: o.y0, view: [], T: [], alive: [], pos: [] }));
  // Ship history.
  let ship = { gamma: shipState.gamma.slice(), v: shipState.v.slice(), X: shipState.X.slice(), Y: shipState.Y.slice() };
  const shipPositions = [ship.gamma.slice()];
  const shipHistory = [{ gamma: ship.gamma.slice(), v: ship.v.slice(), X: ship.X.slice(), Y: ship.Y.slice() }];
  // Record step 0 views and positions.
  for (let i = 0; i < objs.length; i++) {
    const v = viewCoords(objs[i].gamma, objs[i].v, ship);
    objHistory[i].view.push(v);
    objHistory[i].T.push(objs[i].T);
    objHistory[i].alive.push(objs[i].alive);
    objHistory[i].pos.push(objs[i].gamma.slice());
  }
  // Step forward.
  for (let k = 0; k < nSteps; k++) {
    // Snapshot old ship state (used to build a linear estimate of each
    // test object's dT before refining against the NEW ship slice).
    const old_gamma_S = ship.gamma.slice();
    const old_v_S = ship.v.slice();
    const a_S_old = geodesicAccel(old_gamma_S, old_v_S, params.alpha);
    // Advance ship by dt.
    ship = traceShipFrame(ship, dt, 4);
    shipPositions.push(ship.gamma.slice());
    shipHistory.push({ gamma: ship.gamma.slice(), v: ship.v.slice(), X: ship.X.slice(), Y: ship.Y.slice() });
    // Advance each test object so that gamma_i lies on the NEW spatial slice
    // {p : p . v_S(t+dt) = 0}. We use the linearised rate
    //   dT/dt = - gamma_i . a_S / (v_i . v_S)
    // for an initial Euler step (this reduces to gamma_i . gamma_S / (v_i . v_S)
    // for round S^3, giving cos(r) initially in the parallel-transport setup),
    // then run up to 3 Newton iterations on F(T) = gamma_i(T) . v_S(new) = 0,
    // F'(T) = v_i(T) . v_S(new).
    for (let i = 0; i < objs.length; i++) {
      const o = objs[i];
      const recordDead = () => {
        objHistory[i].view.push(null);
        objHistory[i].T.push(o.T);
        objHistory[i].alive.push(false);
        objHistory[i].pos.push(null);
      };
      if (!o.alive) { recordDead(); continue; }
      const vdotv_old = old_v_S[0]*o.v[0] + old_v_S[1]*o.v[1] + old_v_S[2]*o.v[2] + old_v_S[3]*o.v[3];
      if (vdotv_old <= 0) { o.alive = false; recordDead(); continue; }
      const gi_dot_aS = o.gamma[0]*a_S_old[0] + o.gamma[1]*a_S_old[1] + o.gamma[2]*a_S_old[2] + o.gamma[3]*a_S_old[3];
      const tauProp = -gi_dot_aS / vdotv_old;
      if (!Number.isFinite(tauProp) || tauProp <= 0) { o.alive = false; recordDead(); continue; }
      let dT = tauProp * dt;
      // Sub-step the linear estimate.
      let g = o.gamma, vv = o.v;
      const subs = 4;
      const ds0 = dT / subs;
      for (let s = 0; s < subs; s++) [g, vv] = rk4Step(g, vv, params.alpha, ds0);
      // Newton refinement against gamma_i . v_S(new) = 0.
      for (let it = 0; it < 3; it++) {
        const F = g[0]*ship.v[0] + g[1]*ship.v[1] + g[2]*ship.v[2] + g[3]*ship.v[3];
        if (Math.abs(F) < 1e-7) break;
        const Fp = vv[0]*ship.v[0] + vv[1]*ship.v[1] + vv[2]*ship.v[2] + vv[3]*ship.v[3];
        if (Math.abs(Fp) < 1e-9) break;
        const delta = -F / Fp;
        if (Math.abs(delta) < 1e-8) break;
        const sub2 = 2;
        const dds = delta / sub2;
        for (let s = 0; s < sub2; s++) [g, vv] = rk4Step(g, vv, params.alpha, dds);
        dT += delta;
      }
      // Drop if velocity alignment with new ship has flipped.
      const vdotv_new = ship.v[0]*vv[0] + ship.v[1]*vv[1] + ship.v[2]*vv[2] + ship.v[3]*vv[3];
      if (vdotv_new <= 0) { o.alive = false; recordDead(); continue; }
      o.gamma = g; o.v = vv; o.T += dT;
      const view = viewCoords(o.gamma, o.v, ship);
      objHistory[i].view.push(view);
      objHistory[i].T.push(o.T);
      objHistory[i].alive.push(true);
      objHistory[i].pos.push(o.gamma.slice());
    }
  }
  const helioFrames = shipHistory.map(s => buildHelioFrame(s.gamma));
  return { shipPositions, shipHistory, helioFrames, objHistory, objs, objs0, dt };
}

// ====================================================================
// Ship-sim 3D worldlines, 3D current-position markers, and 2D canvas
// ====================================================================
const simGroup = new THREE.Group();
scene.add(simGroup);
const simMarkerGroup = new THREE.Group();
scene.add(simMarkerGroup);
let shipMarker = null;
let testMarkers = [];
let simCurrentStep = 0;

function clearSimGroup() {
  while (simGroup.children.length) {
    const c = simGroup.children.pop();
    c.geometry?.dispose();
    c.material?.dispose();
  }
}

function rebuildSimGroup() {
  clearSimGroup();
  if (!simHistory) return;
  // Ship worldline (yellow).
  addPolylineFromS3Points(simHistory.shipPositions, 0xffd166, 1.0);
  // Each test object: re-trace its refractive geodesic from the saved initial
  // state (gamma0, v0) by its accumulated proper time.
  for (let i = 0; i < simHistory.objs0.length; i++) {
    const hist = simHistory.objHistory[i];
    const T_final = hist.T[hist.T.length - 1];
    if (T_final < 1e-6) continue;
    const o0 = simHistory.objs0[i];
    const traj = traceGeodesic(o0.gamma, o0.v, params.alpha, T_final, Math.max(20, Math.floor(T_final * 80)));
    const colorH = ((Math.atan2(o0.y0, o0.x0) / (2*Math.PI)) + 0.5) % 1;
    const color = new THREE.Color().setHSL(colorH, 0.65, 0.62);
    addPolylineFromS3Points(traj, color.getHex(), 0.85);
  }
}

function addPolylineFromS3Points(pts, color, opacity) {
  const maxSegs = Math.max(1, pts.length - 1);
  const positions = new Float32Array(maxSegs * 2 * 3);
  const geom = new THREE.BufferGeometry();
  const attr = new THREE.BufferAttribute(positions, 3);
  attr.setUsage(THREE.DynamicDrawUsage);
  geom.setAttribute('position', attr);
  geom.setDrawRange(0, 0);
  const mat = new THREE.LineBasicMaterial({ color, transparent: true, opacity });
  const line = new THREE.LineSegments(geom, mat);
  line.userData = { positions, attr, pts };
  simGroup.add(line);
}

function buildSimMarkers() {
  while (simMarkerGroup.children.length) {
    const c = simMarkerGroup.children.pop();
    c.geometry?.dispose();
    c.material?.dispose();
  }
  shipMarker = null;
  testMarkers = [];
  shipMarker = new THREE.Mesh(
    new THREE.SphereGeometry(0.07, 18, 14),
    new THREE.MeshBasicMaterial({ color: 0xffd166 })
  );
  shipMarker.visible = false;
  simMarkerGroup.add(shipMarker);
  if (!simHistory) return;
  for (let i = 0; i < simHistory.objs0.length; i++) {
    const o0 = simHistory.objs0[i];
    const colorH = ((Math.atan2(o0.y0, o0.x0) / (2*Math.PI)) + 0.5) % 1;
    const color = new THREE.Color().setHSL(colorH, 0.65, 0.62);
    const m = new THREE.Mesh(
      new THREE.SphereGeometry(0.04, 14, 10),
      new THREE.MeshBasicMaterial({ color })
    );
    m.visible = false;
    simMarkerGroup.add(m);
    testMarkers.push(m);
  }
}

function updateSimMarkers(k) {
  if (!simHistory || !shipMarker) {
    if (shipMarker) shipMarker.visible = false;
    for (const m of testMarkers) m.visible = false;
    return;
  }
  const kc = Math.max(0, Math.min(simHistory.shipPositions.length - 1, k));
  const sp = simHistory.shipPositions[kc];
  const sproj = projectS3(sp);
  shipMarker.position.set(sproj.x, sproj.y, sproj.z);
  shipMarker.visible = true;
  for (let i = 0; i < testMarkers.length; i++) {
    const m = testMarkers[i];
    const pos = simHistory.objHistory[i].pos[kc];
    if (!pos) { m.visible = false; continue; }
    const proj = projectS3(pos);
    m.position.set(proj.x, proj.y, proj.z);
    m.visible = true;
  }
}

function reprojectSimGroup() {
  for (const line of simGroup.children) {
    const { positions, attr, pts } = line.userData;
    let segIdx = 0;
    let prev = null;
    for (const p of pts) {
      const proj = projectS3(p);
      if (prev !== null && prev.ball === proj.ball) {
        const o = segIdx * 6;
        positions[o + 0] = prev.x; positions[o + 1] = prev.y; positions[o + 2] = prev.z;
        positions[o + 3] = proj.x; positions[o + 4] = proj.y; positions[o + 5] = proj.z;
        segIdx++;
      }
      prev = proj;
    }
    attr.needsUpdate = true;
    line.geometry.setDrawRange(0, segIdx * 2);
    line.geometry.computeBoundingSphere();
  }
}

// 2D canvas drawing
const simCanvas = document.getElementById('simcanvas');
const simCtx = simCanvas.getContext('2d');

function drawSimView(timeStep) {
  const w = simCanvas.width, h = simCanvas.height;
  simCtx.fillStyle = '#0e1116';
  simCtx.fillRect(0, 0, w, h);
  if (!simHistory) {
    simCtx.fillStyle = '#5a6573';
    simCtx.font = '12px ui-sans-serif';
    simCtx.textAlign = 'center';
    simCtx.fillText('Run a simulation to populate the view.', w/2, h/2);
    return;
  }
  const ext = simParams.viewSize;
  const scale = w / (2 * ext);
  const toPx = (x, y) => [w/2 + x*scale, h/2 - y*scale];
  const helio = simParams.heliocentric && simHistory.shipHistory.length > 0;
  const ts = Math.max(0, Math.min(simHistory.shipHistory.length - 1, timeStep));
  const shipNow = simHistory.shipHistory[ts];
  // Heliocentric frame: round-S^3 inverse-exp at the closest source point to
  // the current ship, projected onto the 2D subspace perpendicular to the
  // source curve. gamma_src(t) thus always sits at (0, 0).
  const helioFrameNow = helio ? simHistory.helioFrames[ts] : null;
  const projHelio = (p, fr) => helioProject(p, fr);
  // Grid hairlines (only in ship-centred view; in heliocentric the grid is meaningless).
  if (!helio) {
    simCtx.strokeStyle = '#1c2128';
    simCtx.lineWidth = 1;
    const gN = simParams.N;
    const gridStep = gN <= 1 ? simParams.extent : (2 * simParams.extent) / (gN - 1);
    for (let i = -gN; i <= gN; i++) {
      const wp = i * gridStep;
      if (Math.abs(wp) > ext) continue;
      const [px, ] = toPx(wp, 0);
      simCtx.beginPath(); simCtx.moveTo(px, 0); simCtx.lineTo(px, h); simCtx.stroke();
      const [, py] = toPx(0, wp);
      simCtx.beginPath(); simCtx.moveTo(0, py); simCtx.lineTo(w, py); simCtx.stroke();
    }
  }
  // Axes.
  simCtx.strokeStyle = '#2a313c';
  simCtx.lineWidth = 1.2;
  simCtx.beginPath(); simCtx.moveTo(w/2, 0); simCtx.lineTo(w/2, h); simCtx.stroke();
  simCtx.beginPath(); simCtx.moveTo(0, h/2); simCtx.lineTo(w, h/2); simCtx.stroke();
  // Project source curve.
  if (circleSamples.length) {
    const pts = [];
    let closest = null;
    if (helio) {
      for (let i = 0; i < circleSamples.length; i++) {
        const ph = projHelio(circleSamples[i], helioFrameNow);
        pts.push([ph.x, ph.y]);
      }
      // Closest source point IS gamma_src, which always sits at the origin.
      const shipR = projHelio(shipNow.gamma, helioFrameNow).r;
      closest = [0, 0, shipR];
    } else {
      for (let i = 0; i < circleSamples.length; i++) {
        const vp = viewPoint(circleSamples[i], shipNow);
        const x = vp.r * Math.cos(vp.phi), y = vp.r * Math.sin(vp.phi);
        pts.push([x, y]);
        if (!closest || vp.r < closest[2]) closest = [x, y, vp.r];
      }
    }
    simCtx.strokeStyle = 'rgba(255, 85, 102, 0.55)';
    simCtx.lineWidth = 1.5;
    simCtx.beginPath();
    for (let i = 0; i <= pts.length; i++) {
      const [x, y] = pts[i % pts.length];
      const [px, py] = toPx(x, y);
      if (i === 0) simCtx.moveTo(px, py); else simCtx.lineTo(px, py);
    }
    simCtx.stroke();
    if (closest) {
      const [cx, cy, cd] = closest;
      const [cpx, cpy] = toPx(cx, cy);
      simCtx.fillStyle = '#ff5566';
      simCtx.beginPath(); simCtx.arc(cpx, cpy, 4.5, 0, Math.PI*2); simCtx.fill();
      simCtx.strokeStyle = 'rgba(255, 192, 200, 0.75)';
      simCtx.lineWidth = 1;
      simCtx.beginPath(); simCtx.arc(cpx, cpy, 8, 0, Math.PI*2); simCtx.stroke();
      simCtx.fillStyle = '#ffb0bb';
      simCtx.font = '10.5px ui-monospace, Menlo, monospace';
      simCtx.textAlign = 'left';
      simCtx.fillText(`d=${cd.toFixed(2)}`, cpx + 10, cpy - 6);
    }
  }
  // Test object trails up to timeStep.
  for (let i = 0; i < simHistory.objHistory.length; i++) {
    const hist = simHistory.objHistory[i];
    const o0 = simHistory.objs0[i];
    const colorH = ((Math.atan2(o0.y0, o0.x0) / (2*Math.PI)) + 0.5) % 1;
    simCtx.strokeStyle = `hsla(${(colorH*360)|0}, 65%, 65%, 0.55)`;
    simCtx.lineWidth = 1.2;
    simCtx.beginPath();
    let drawing = false;
    for (let k = 0; k <= timeStep && k < hist.view.length; k++) {
      let xPos, yPos;
      if (helio) {
        const pos = hist.pos[k];
        if (!pos) { drawing = false; continue; }
        const ph = projHelio(pos, simHistory.helioFrames[k]);
        xPos = ph.x; yPos = ph.y;
      } else {
        const vw = hist.view[k];
        if (!vw) { drawing = false; continue; }
        xPos = vw.r * Math.cos(vw.phi);
        yPos = vw.r * Math.sin(vw.phi);
      }
      const [px, py] = toPx(xPos, yPos);
      if (!drawing) { simCtx.moveTo(px, py); drawing = true; } else { simCtx.lineTo(px, py); }
    }
    simCtx.stroke();
  }
  // Test object current positions and tau rings.
  for (let i = 0; i < simHistory.objHistory.length; i++) {
    const hist = simHistory.objHistory[i];
    if (timeStep >= hist.view.length) continue;
    const vw = hist.view[timeStep];
    if (!vw) continue;
    let xPos, yPos;
    if (helio) {
      const pos = hist.pos[timeStep];
      if (!pos) continue;
      const ph = projHelio(pos, helioFrameNow);
      xPos = ph.x; yPos = ph.y;
    } else {
      xPos = vw.r * Math.cos(vw.phi);
      yPos = vw.r * Math.sin(vw.phi);
    }
    const [px, py] = toPx(xPos, yPos);
    const o0 = simHistory.objs0[i];
    const colorH = ((Math.atan2(o0.y0, o0.x0) / (2*Math.PI)) + 0.5) % 1;
    simCtx.fillStyle = `hsl(${(colorH*360)|0}, 70%, 64%)`;
    simCtx.beginPath(); simCtx.arc(px, py, 3.5, 0, Math.PI*2); simCtx.fill();
    const dilation = Math.max(0, 1 - vw.tau);
    if (dilation > 0.01) {
      simCtx.strokeStyle = `hsla(${(colorH*360)|0}, 80%, 80%, ${0.35 + 0.45*dilation})`;
      simCtx.lineWidth = 1;
      simCtx.beginPath(); simCtx.arc(px, py, 4 + dilation*7, 0, Math.PI*2); simCtx.stroke();
    }
  }
  // Ship marker (with trail in heliocentric mode).
  if (helio) {
    simCtx.strokeStyle = 'rgba(255, 209, 102, 0.45)';
    simCtx.lineWidth = 1.5;
    simCtx.beginPath();
    for (let k = 0; k <= ts; k++) {
      const ph = projHelio(simHistory.shipHistory[k].gamma, simHistory.helioFrames[k]);
      const [px, py] = toPx(ph.x, ph.y);
      if (k === 0) simCtx.moveTo(px, py); else simCtx.lineTo(px, py);
    }
    simCtx.stroke();
    const shipP = projHelio(shipNow.gamma, helioFrameNow);
    const [px, py] = toPx(shipP.x, shipP.y);
    simCtx.fillStyle = '#ffd166';
    simCtx.beginPath(); simCtx.arc(px, py, 5, 0, Math.PI*2); simCtx.fill();
    simCtx.strokeStyle = '#fff4d6';
    simCtx.lineWidth = 1;
    simCtx.beginPath(); simCtx.arc(px, py, 7, 0, Math.PI*2); simCtx.stroke();
  } else {
    simCtx.fillStyle = '#ffd166';
    simCtx.beginPath(); simCtx.arc(w/2, h/2, 5, 0, Math.PI*2); simCtx.fill();
    simCtx.strokeStyle = '#fff4d6';
    simCtx.lineWidth = 1;
    simCtx.beginPath(); simCtx.arc(w/2, h/2, 7, 0, Math.PI*2); simCtx.stroke();
  }
}

drawSimView(0);

// ====================================================================
// Dirty flags & main loop
// ====================================================================
let dirtyCircle = true;
let dirtyPotential = true;
let dirtyGeodesics = true;
let dirtyCells = true;
let dirtyReproject = true;

function rebuildAll() {
  if (dirtyCircle)    { rebuildCircleSamples(); dirtyCircle = false; dirtyPotential = true; dirtyReproject = true; }
  if (dirtyPotential) { buildPotential();       dirtyPotential = false; }
  // dirtyGeodesics / dirtyCells intentionally unused: the cell wireframes and
  // geodesic fan are not rendered any more.
}

buildMarkers();
rebuildAll();
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
  if (rotated) dirtyReproject = true;

  if (dirtyCircle || dirtyPotential) rebuildAll();
  if (dirtyReproject) { repositionMarkers(); reprojectSimGroup(); updateSimMarkers(simCurrentStep); dirtyReproject = false; }

  cellGroup.visible = false;
  geodesicGroup.visible = false;
  markerGroup.visible = params.showMarkers;
  simGroup.visible = simParams.showSim;
  simMarkerGroup.visible = simParams.showSim;

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

bindRange('alpha', 'alpha-v', x => x.toFixed(3), (x) => { params.alpha = x; });
bindRange('rad', 'rad-v', x => x.toFixed(2), (x) => { circleSource.radius = x * Math.PI; dirtyCircle = true; });
bindRange('voxN', 'voxN-v', x => String(x|0), (x) => { params.N = x|0; dirtyPotential = true; });
bindRange('eps', 'eps-v', x => Math.pow(10, x).toExponential(1), (x) => { params.eps = Math.pow(10, x); dirtyPotential = true; });
document.getElementById('kernel').addEventListener('change', (e) => {
  params.kernel = e.target.value;
  dirtyPotential = true;
});
document.getElementById('sourceMode').addEventListener('change', (e) => {
  params.sourceMode = e.target.value;
  dirtyPotential = true;
});

// Spaceship sim wiring.
bindRange('simext', 'simext-v', x => x.toFixed(2), (x) => { simParams.extent = x; });
bindRange('simN', 'simN-v', x => String(x|0), (x) => { simParams.N = x|0; });
bindRange('simdt', 'simdt-v', x => x.toFixed(3), (x) => { simParams.dt = x; });
bindRange('simsteps', 'simsteps-v', x => String(x|0), (x) => { simParams.steps = x|0; });
const simTimeEl = document.getElementById('simtime');
const simTimeLabel = document.getElementById('simtime-v');
simTimeEl.addEventListener('input', () => {
  const k = parseInt(simTimeEl.value, 10);
  simCurrentStep = k;
  drawSimView(k);
  updateSimMarkers(k);
  if (simHistory) simTimeLabel.textContent = `t = ${(k * simHistory.dt).toFixed(3)}`;
});
bindRange('simview-size', 'simview-size-v', x => `view ${x.toFixed(2)}`, (x) => {
  simParams.viewSize = x;
  drawSimView(simCurrentStep);
});
document.getElementById('heliocentric').addEventListener('change', (e) => {
  simParams.heliocentric = e.target.checked;
  drawSimView(simCurrentStep);
});
document.getElementById('reroll-ship').addEventListener('click', () => {
  if (!orbitParams.enabled) sourcePoint = randomS3();
  shipState = initialShip();
});
document.getElementById('orbit-init').addEventListener('change', (e) => {
  orbitParams.enabled = e.target.checked;
  shipState = initialShip();
});
bindRange('orbDist', 'orbDist-v', x => x.toFixed(3), (x) => {
  orbitParams.distance = x;
  if (orbitParams.enabled) shipState = initialShip();
});
bindRange('orbOrth', 'orbOrth-v', x => x.toFixed(3), (x) => {
  orbitParams.orthVel = x;
  if (orbitParams.enabled) shipState = initialShip();
});
document.getElementById('run-sim').addEventListener('click', () => {
  simHistory = runShipSim();
  rebuildSimGroup();
  reprojectSimGroup();
  buildSimMarkers();
  simCurrentStep = 0;
  updateSimMarkers(0);
  simTimeEl.max = String(simParams.steps);
  simTimeEl.value = '0';
  simTimeLabel.textContent = `t = ${(0).toFixed(3)}`;
  drawSimView(0);
});
document.getElementById('show-sim').addEventListener('change', (e) => { simParams.showSim = e.target.checked; });
