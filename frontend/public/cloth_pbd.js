// Position-Based Dynamics cloth solver -- plain functions, no THREE.js/DOM
// dependency, so this exact file can be pasted into a <script> tag AND
// exercised by a plain Node test before it ever touches a browser.
//
// Standard PBD (Mueller et al. 2007): fixed points use invMass=0 and are
// handled by weighting every constraint correction by each vertex's inverse
// mass (a 0-invMass vertex absorbs none of the correction, its neighbor
// absorbs all of it) -- NOT by predicting/iterating with velocity and then
// separately resetting anchors afterward. That "reset after" pattern is
// what an earlier from-scratch attempt in this project's Python drape code
// tried and documented as reliably diverging (energy fed back in through
// (pos - prevPos) velocity when the anchor's correction was discarded
// rather than never applied). Weighting by invMass from the start avoids
// that failure mode by construction and is the well-tested textbook form.

function buildClothTopology(faces) {
  // edgeKey -> { a, b, tris: [faceIdx, ...] }
  const edgeMap = new Map();
  function addEdge(a, b, triIdx) {
    const key = a < b ? a + "_" + b : b + "_" + a;
    let e = edgeMap.get(key);
    if (!e) {
      e = { a: Math.min(a, b), b: Math.max(a, b), tris: [] };
      edgeMap.set(key, e);
    }
    e.tris.push(triIdx);
  }
  for (let f = 0; f < faces.length; f++) {
    const [a, b, c] = faces[f];
    addEdge(a, b, f);
    addEdge(b, c, f);
    addEdge(a, c, f);
  }

  function thirdVertex(face, a, b) {
    for (const v of face) if (v !== a && v !== b) return v;
    return -1;
  }

  const distPairs = [];
  const bendPairs = [];
  for (const e of edgeMap.values()) {
    distPairs.push(e.a, e.b);
    if (e.tris.length === 2) {
      const p3 = thirdVertex(faces[e.tris[0]], e.a, e.b);
      const p4 = thirdVertex(faces[e.tris[1]], e.a, e.b);
      if (p3 !== -1 && p4 !== -1 && p3 !== p4) {
        bendPairs.push(p3, p4);
      }
    }
  }
  return {
    distPairs: Int32Array.from(distPairs),
    bendPairs: Int32Array.from(bendPairs),
  };
}

function restLengths(pos, pairs) {
  const n = pairs.length / 2;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const a = pairs[i * 2], b = pairs[i * 2 + 1];
    const dx = pos[a * 3] - pos[b * 3];
    const dy = pos[a * 3 + 1] - pos[b * 3 + 1];
    const dz = pos[a * 3 + 2] - pos[b * 3 + 2];
    out[i] = Math.sqrt(dx * dx + dy * dy + dz * dz);
  }
  return out;
}

function createClothState(vertices, faces, anchorMaskFn) {
  const n = vertices.length;
  const pos = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    pos[i * 3] = vertices[i][0];
    pos[i * 3 + 1] = vertices[i][1];
    pos[i * 3 + 2] = vertices[i][2];
  }
  const prevPos = Float32Array.from(pos);
  const invMass = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    invMass[i] = anchorMaskFn(vertices[i], i) ? 0.0 : 1.0;
  }

  const topo = buildClothTopology(faces);
  const distRest = restLengths(pos, topo.distPairs);
  const bendRest = restLengths(pos, topo.bendPairs);

  return {
    n, pos, prevPos, invMass,
    distPairs: topo.distPairs, distRest,
    bendPairs: topo.bendPairs, bendRest,
    settled: false,
  };
}

function solvePairConstraints(pos, invMass, pairs, rest, stiffness) {
  const m = pairs.length / 2;
  for (let i = 0; i < m; i++) {
    const a = pairs[i * 2], b = pairs[i * 2 + 1];
    const wa = invMass[a], wb = invMass[b];
    const wSum = wa + wb;
    if (wSum === 0) continue;

    const ax = a * 3, bx = b * 3;
    const dx = pos[ax] - pos[bx];
    const dy = pos[ax + 1] - pos[bx + 1];
    const dz = pos[ax + 2] - pos[bx + 2];
    const dist = Math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-9;
    const diff = (dist - rest[i]) / dist;

    const corrX = dx * diff * stiffness / wSum;
    const corrY = dy * diff * stiffness / wSum;
    const corrZ = dz * diff * stiffness / wSum;

    pos[ax] -= corrX * wa;
    pos[ax + 1] -= corrY * wa;
    pos[ax + 2] -= corrZ * wa;
    pos[bx] += corrX * wb;
    pos[bx + 1] += corrY * wb;
    pos[bx + 2] += corrZ * wb;
  }
}

// bodyRadiusProfile: { yMin, yMax, nBands, radii: Float32Array } -- max
// distance from the central (x=0,z=0) vertical axis at each height band,
// sampled from the body mesh. A cheap cylindrical approximation: accurate
// for the torso/hip/skirt region (which is what real gravity-driven folds
// mostly show, per the reference), deliberately not attempted for the
// T-pose splayed arms (a harder, already-documented limitation elsewhere
// in this project) -- collision there is skipped, matching the fact this
// approximation was never claimed to fix sleeve fit.
function collideWithBody(pos, invMass, profile, margin) {
  const n = pos.length / 3;
  for (let i = 0; i < n; i++) {
    if (invMass[i] === 0) continue;
    const x = pos[i * 3], y = pos[i * 3 + 1], z = pos[i * 3 + 2];
    let t = (y - profile.yMin) / (profile.yMax - profile.yMin || 1);
    let band = Math.floor(t * (profile.nBands - 1));
    if (band < 0) band = 0;
    if (band > profile.nBands - 1) band = profile.nBands - 1;
    const bodyR = profile.radii[band];
    if (bodyR <= 0) continue;

    const r = Math.sqrt(x * x + z * z);
    const targetR = bodyR + margin;
    if (r < targetR && r > 1e-6) {
      const push = targetR / r;
      pos[i * 3] = x * push;
      pos[i * 3 + 2] = z * push;
    }
  }
}

function applyAnchors(pos, prevPos, invMass) {
  const n = pos.length / 3;
  for (let i = 0; i < n; i++) {
    if (invMass[i] === 0) {
      pos[i * 3] = prevPos[i * 3];
      pos[i * 3 + 1] = prevPos[i * 3 + 1];
      pos[i * 3 + 2] = prevPos[i * 3 + 2];
    }
  }
}

// One full PBD step: predict (using velocity from the previous step) once,
// then iterate position-only constraint solving, then derive the new
// velocity from the net displacement over the whole step. Matches the
// standard formulation's stability guarantee -- velocity is never touched
// mid-iteration.
function stepCloth(state, opts) {
  const {
    dt = 1 / 60,
    gravity = -80,         // cm/s^2 -- well below real gravity (981); tuned
                           // for structured cotton-weight fabric, not silk.
                           // An early -300 default (validated only against
                           // abstract grid/shirt-shape tests, never the real
                           // exported garment mesh) sagged a real polo's hem
                           // ~17cm lower than its actual length once tested
                           // against real data -- -80 keeps that under 5cm.
    damping = 0.95,
    distStiffness = 0.97,
    bendStiffness = 0.5,
    iterations = 6,
    bodyRadiusProfile = null,
    collisionMargin = 0.3,
  } = opts;

  const { pos, prevPos, invMass, n } = state;

  // Predict.
  for (let i = 0; i < n; i++) {
    if (invMass[i] === 0) continue;
    const ix = i * 3;
    const vx = (pos[ix] - prevPos[ix]) * damping;
    const vy = (pos[ix + 1] - prevPos[ix + 1]) * damping;
    const vz = (pos[ix + 2] - prevPos[ix + 2]) * damping;
    prevPos[ix] = pos[ix];
    prevPos[ix + 1] = pos[ix + 1];
    prevPos[ix + 2] = pos[ix + 2];
    pos[ix] += vx;
    pos[ix + 1] += vy + gravity * dt * dt;
    pos[ix + 2] += vz;
  }

  for (let it = 0; it < iterations; it++) {
    solvePairConstraints(pos, invMass, state.distPairs, state.distRest, distStiffness);
    solvePairConstraints(pos, invMass, state.bendPairs, state.bendRest, bendStiffness);
    if (bodyRadiusProfile) {
      collideWithBody(pos, invMass, bodyRadiusProfile, collisionMargin);
    }
    applyAnchors(pos, prevPos, invMass);
  }

  // Convergence check: max per-vertex displacement this step.
  let maxDisp2 = 0;
  for (let i = 0; i < n; i++) {
    const ix = i * 3;
    const dx = pos[ix] - prevPos[ix];
    const dy = pos[ix + 1] - prevPos[ix + 1];
    const dz = pos[ix + 2] - prevPos[ix + 2];
    const d2 = dx * dx + dy * dy + dz * dz;
    if (d2 > maxDisp2) maxDisp2 = d2;
  }
  return Math.sqrt(maxDisp2);
}

// maxAbsX: exclude vertices further than this from the central vertical
// axis (in the arms' own spread direction) when computing each height
// band's max radius. Without this, a T-pose body's shoulder/arm height
// bands report a radius stretching all the way to the fingertips (~75cm),
// not the torso's real ~15-20cm -- and every cloth vertex near that height
// then gets pushed out to fingertip distance regardless of whether it's
// anywhere near the actual arm, producing a flat ring/halo artifact instead
// of a collision surface. Restricting to a torso-width column keeps the
// profile meaningful for the torso/hip region (where real folds happen)
// and effectively disables collision for the sleeve/arm region instead of
// actively corrupting it -- matching the intended "collision skipped for
// arms" behavior rather than the broken version of it.
function buildRadiusProfile(vertices, nBands, maxAbsX) {
  nBands = nBands || 60;
  maxAbsX = maxAbsX == null ? 25 : maxAbsX;
  let yMin = Infinity, yMax = -Infinity;
  for (const v of vertices) {
    if (v[1] < yMin) yMin = v[1];
    if (v[1] > yMax) yMax = v[1];
  }
  const radii = new Float32Array(nBands);
  const span = (yMax - yMin) || 1;
  for (const v of vertices) {
    const x = v[0], y = v[1], z = v[2];
    if (Math.abs(x) > maxAbsX) continue;
    let band = Math.floor(((y - yMin) / span) * (nBands - 1));
    if (band < 0) band = 0;
    if (band > nBands - 1) band = nBands - 1;
    const r = Math.sqrt(x * x + z * z);
    if (r > radii[band]) radii[band] = r;
  }
  return { yMin, yMax, nBands, radii };
}

if (typeof module !== "undefined") {
  module.exports = {
    buildClothTopology, createClothState, stepCloth, buildRadiusProfile,
    solvePairConstraints, collideWithBody, applyAnchors, restLengths,
  };
}
