/* Human motion generation comparison frontend — multi-model, multi-person viewer */

// HumanML3D 22-joint kinematic chains (mirrors paramUtil; same topology for all models)
const KINEMATIC_CHAIN = [
  [0, 2, 5, 8, 11],        // right leg
  [0, 1, 4, 7, 10],        // left leg
  [0, 3, 6, 9, 12, 15],    // spine -> neck -> head
  [9, 14, 17, 19, 21],     // right arm
  [9, 13, 16, 18, 20],     // left arm
];
const HEAD_JOINT = 15;

// SOMA77 edge list (a, b, radius) generated from nv-tlabs/kimodo hierarchy;
// facial joints (Jaw/Eyes) skipped. Mirrors webapp/render_utils.py.
function buildSomaEdges() {
  const FINGERS = { Thumb: [1, 2, 3], Index: [1, 2, 3, 4], Middle: [1, 2, 3, 4], Ring: [1, 2, 3, 4], Pinky: [1, 2, 3, 4] }; // thumb has no 4
  const names = ['Hips','Spine1','Spine2','Chest','Neck1','Neck2','Head','HeadEnd','Jaw','LeftEye','RightEye'];
  const parents = { Spine1:'Hips', Spine2:'Spine1', Chest:'Spine2', Neck1:'Chest', Neck2:'Neck1',
                    Head:'Neck2', HeadEnd:'Head', Jaw:'Head', LeftEye:'Head', RightEye:'Head' };
  for (const s of ['Left','Right']) {
    names.push(`${s}Shoulder`,`${s}Arm`,`${s}ForeArm`,`${s}Hand`);
    parents[`${s}Shoulder`]='Chest'; parents[`${s}Arm`]=`${s}Shoulder`; parents[`${s}ForeArm`]=`${s}Arm`; parents[`${s}Hand`]=`${s}ForeArm`;
    for (const f of Object.keys(FINGERS)) {
      for (const i of FINGERS[f]) {
        names.push(`${s}Hand${f}${i}`);
        parents[`${s}Hand${f}${i}`] = i>1 ? `${s}Hand${f}${i-1}` : `${s}Hand`;
      }
      names.push(`${s}Hand${f}End`);
      parents[`${s}Hand${f}End`] = `${s}Hand${f}${FINGERS[f][FINGERS[f].length-1]}`;
    }
  }
  names.push('LeftLeg','LeftShin','LeftFoot','LeftToeBase','LeftToeEnd','RightLeg','RightShin','RightFoot','RightToeBase','RightToeEnd');
  Object.assign(parents, { LeftLeg:'Hips', LeftShin:'LeftLeg', LeftFoot:'LeftShin', LeftToeBase:'LeftFoot', LeftToeEnd:'LeftToeBase',
                           RightLeg:'Hips', RightShin:'RightLeg', RightFoot:'RightShin', RightToeBase:'RightFoot', RightToeEnd:'RightToeBase' });
  if (names.length !== 77) throw new Error('SOMA joint count mismatch: ' + names.length);
  const skip = new Set(['Jaw','LeftEye','RightEye']);
  const thin = new Set(names.map((n,i)=>({n,i})).filter(x=>x.n.includes('Hand')||x.n.includes('Toe')).map(x=>x.i));
  const spine = new Set([0,1,2,3,4,5,6]);
  const edges = [];
  names.forEach((n, i) => {
    const p = parents[n];
    if (!p || skip.has(n) || skip.has(p)) return;
    const pi = names.indexOf(p);
    const r = (thin.has(i)||thin.has(pi)) ? 0.008 : (spine.has(pi) ? 0.024 : 0.013);
    edges.push({ a: pi, b: i, radius: r });
  });
  return { joints: names.length, head: 7, edges };
}

const SKELETONS = { smpl22: null, soma77: buildSomaEdges() };
{
  // smpl22 edges from chains
  const seen = new Set(); const edges = [];
  KINEMATIC_CHAIN.forEach((chain) => {
    for (let i = 0; i < chain.length - 1; i++) {
      const key = Math.min(chain[i], chain[i+1]) + '-' + Math.max(chain[i], chain[i+1]);
      if (!seen.has(key)) { seen.add(key); edges.push({ a: chain[i], b: chain[i+1], radius: 0.014 }); }
    }
  });
  SKELETONS.smpl22 = { joints: 22, head: HEAD_JOINT, edges };
}

const PERSON_COLORS = [
  { joint: 0x5b8cff, bone: 0x7f93c9, head: 0xe8ecf3, trail: 0x38e0c8, label: '人物 A' },
  { joint: 0xff9e5b, bone: 0xc98a5f, head: 0xf3e3d0, trail: 0xffd166, label: '人物 B' },
];

const MODEL_INFO = {
  momask: 'MoMask 单人',
  momask_dual: 'MoMask 拼合',
  intergen: 'InterGen',
  in2in: 'in2IN',
  kimodo: 'Kimodo (NVIDIA)',
};

const EXAMPLES_SINGLE = [
  'A person is running on a treadmill.',
  'A person jumps up and then lands.',
  'The person does a salsa dance.',
  'A man bends down and picks something up with his right hand.',
];
const EXAMPLES_INTER = [
  'In an intense boxing match, one is continuously punching while the other is defending and counterattacking.',
  'Two fencers engage in a thrilling duel, their sabres clashing as they strive for victory.',
  'With fiery passion two dancers entwine in Latin dance sublime.',
  'Two good friends jump in the same rhythm to celebrate.',
  'Two people embrace each other.',
];

// ---------------------------------------------------------------- state
let motion = null;        // {persons: [ [nf][22][3] ...], fps, model, ...}
let playing = false;
let frameIdx = 0;
let speed = 1;
let acc = 0;
let currentModel = 'momask';
const history = [];       // generation result records

// ---------------------------------------------------------------- dom
const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------- three.js scene
const wrap = $('canvas-wrap');
let renderer = null, controls = null, camera = null, scene = null;
let webglOk = true;

try {
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0e1014);
  scene.fog = new THREE.Fog(0x0e1014, 10, 26);

  camera = new THREE.PerspectiveCamera(45, 1, 0.05, 100);
  camera.position.set(3.4, 2.1, 4.2);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  wrap.appendChild(renderer.domElement);

  controls = new THREE.OrbitControls(camera, renderer.domElement);
} catch (e) {
  webglOk = false;
  console.error('WebGL init failed:', e);
  const banner = document.createElement('div');
  banner.style.cssText = 'position:absolute;top:14px;left:50%;transform:translateX(-50%);' +
    'background:#3a2b1f;color:#ffb86b;border:1px solid #6b4f2f;border-radius:8px;' +
    'padding:8px 14px;font-size:13px;z-index:5;';
  banner.textContent = '当前浏览器不支持 WebGL，无法显示 3D 预览（生成与下载功能不受影响）';
  wrap.appendChild(banner);
}

if (webglOk) {
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.maxDistance = 18;
  controls.minDistance = 1.2;
  controls.target.set(0, 0.9, 0);

  scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x2c3145, 1.05));
  const dirLight = new THREE.DirectionalLight(0xffffff, 0.75);
  dirLight.position.set(3, 6, 4);
  scene.add(dirLight);

  scene.add(new THREE.GridHelper(16, 32, 0x3a4356, 0x1c212b));
}

// per-person skeleton objects, built lazily
const jointGeo = webglOk ? new THREE.SphereGeometry(1, 18, 14) : null;
const boneGeo = webglOk ? new THREE.CylinderGeometry(1, 1, 1, 10, 1, true) : null;
const skeletons = [];

function buildSkeleton(ci, skeleton) {
  const spec = SKELETONS[skeleton] || SKELETONS.smpl22;
  const c = PERSON_COLORS[ci % PERSON_COLORS.length];
  const group = new THREE.Group();
  const joints = [];
  for (let j = 0; j < spec.joints; j++) {
    const mat = new THREE.MeshPhongMaterial({ color: j === spec.head ? c.head : c.joint, shininess: 70 });
    const m = new THREE.Mesh(jointGeo, mat);
    m.scale.setScalar(j === spec.head ? 0.055 : (spec.joints > 22 ? 0.016 : 0.024));
    group.add(m);
    joints.push(m);
  }
  const bones = spec.edges.map(({ a, b, radius }) => {
    const mat = new THREE.MeshPhongMaterial({ color: c.bone, shininess: 40 });
    const mesh = new THREE.Mesh(boneGeo, mat);
    mesh.userData = { a, b, radius };
    group.add(mesh);
    return mesh;
  });
  scene.add(group);
  return { group, joints, bones, color: c };
}

const trailLines = [];

function buildTrail(ci, jointsData) {
  if (trailLines[ci]) { scene.remove(trailLines[ci]); trailLines[ci].geometry.dispose(); trailLines[ci] = null; }
  const n = jointsData.length;
  const pos = new Float32Array(n * 3);
  for (let f = 0; f < n; f++) {
    const root = jointsData[f][0];
    pos[f * 3] = root[0]; pos[f * 3 + 1] = 0.004; pos[f * 3 + 2] = root[2];
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  const line = new THREE.Line(geo, new THREE.LineBasicMaterial({
    color: PERSON_COLORS[ci % PERSON_COLORS.length].trail, transparent: true, opacity: 0.55,
  }));
  scene.add(line);
  trailLines[ci] = line;
}

const UP = new THREE.Vector3(0, 1, 0);
const tmpA = new THREE.Vector3(), tmpB = new THREE.Vector3(), tmpDir = new THREE.Vector3();

function applyFrame(f) {
  if (!webglOk || !motion) return;
  const skeleton = motion.skeleton || 'smpl22';
  motion.persons.forEach((jointsData, ci) => {
    if (!skeletons[ci] || skeletons[ci].builtFor !== skeleton) {
      if (skeletons[ci]) { scene.remove(skeletons[ci].group); skeletons[ci] = null; }
      skeletons[ci] = buildSkeleton(ci, skeleton);
      skeletons[ci].builtFor = skeleton;
    }
    const frame = jointsData[Math.min(f, jointsData.length - 1)];
    const sk = skeletons[ci];
    for (let j = 0; j < sk.joints.length && j < frame.length; j++) {
      sk.joints[j].position.set(frame[j][0], frame[j][1], frame[j][2]);
    }
    for (const mesh of sk.bones) {
      const { a, b, radius } = mesh.userData;
      if (a >= frame.length || b >= frame.length) { mesh.visible = false; continue; }
      tmpA.set(frame[a][0], frame[a][1], frame[a][2]);
      tmpB.set(frame[b][0], frame[b][1], frame[b][2]);
      tmpDir.subVectors(tmpB, tmpA);
      const len = tmpDir.length();
      if (len < 1e-6) { mesh.visible = false; continue; }
      mesh.visible = true;
      mesh.position.copy(tmpA).addScaledVector(tmpDir, 0.5);
      mesh.quaternion.setFromUnitVectors(UP, tmpDir.normalize());
      mesh.scale.set(radius, len, radius);
    }
  });
  // hide unused skeletons (e.g. switched from dual to single model)
  for (let ci = motion.persons.length; ci < skeletons.length; ci++) {
    if (skeletons[ci]) skeletons[ci].group.visible = false;
    if (trailLines[ci]) trailLines[ci].visible = false;
  }
  for (let ci = 0; ci < motion.persons.length; ci++) {
    if (skeletons[ci]) skeletons[ci].group.visible = true;
    if (trailLines[ci]) trailLines[ci].visible = true;
  }
}

function resize() {
  if (!webglOk) return;
  const w = wrap.clientWidth, h = wrap.clientHeight;
  if (w === 0 || h === 0) return;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
}
window.addEventListener('resize', resize);

const followTarget = new THREE.Vector3(0, 0.9, 0);

const clock = new THREE.Clock();
function animate() {
  requestAnimationFrame(animate);
  if (!webglOk) return;
  const dt = clock.getDelta();

  if (playing && motion) {
    acc += dt * motion.fps * speed;
    const n = motion.m_length || motion.persons[0].length;
    while (acc >= 1) { acc -= 1; frameIdx = (frameIdx + 1) % n; }
    applyFrame(frameIdx);
    syncHud();
  }

  if ($('follow').checked && motion) {
    // follow the centroid of all person roots
    let cx = 0, cy = 0, cz = 0, cnt = 0;
    motion.persons.forEach((jd) => {
      const root = jd[Math.min(frameIdx, jd.length - 1)][0];
      cx += root[0]; cy += root[1]; cz += root[2]; cnt++;
    });
    cx /= cnt; cy /= cnt; cz /= cnt;
    followTarget.set(cx, cy * 0.55 + 0.55, cz);
    tmpDir.subVectors(followTarget, controls.target).multiplyScalar(0.1);
    controls.target.add(tmpDir);
    camera.position.add(tmpDir);
  }

  controls.update();
  renderer.render(scene, camera);
}

function syncHud() {
  $('timeline').value = frameIdx;
  const n = motion ? (motion.m_length || motion.persons[0].length) : 0;
  $('frame-label').textContent = (frameIdx + 1) + ' / ' + n;
}

function setPlaying(v) {
  playing = v;
  $('play').textContent = v ? '⏸' : '▶';
}

function loadMotion(resp) {
  motion = resp;
  frameIdx = 0;
  acc = 0;
  applyFrame(0);
  motion.persons.forEach((jd, ci) => buildTrail(ci, jd));
  $('timeline').max = (motion.m_length || motion.persons[0].length) - 1;
  $('placeholder').classList.add('hidden');
  $('hud').classList.remove('hidden');
  setPlaying(true);
  syncHud();

  const legend = $('legend');
  if (motion.persons.length > 1) {
    legend.innerHTML = motion.persons.map((_, ci) => {
      const c = PERSON_COLORS[ci % PERSON_COLORS.length];
      const hex = '#' + c.joint.toString(16).padStart(6, '0');
      return `<span><i style="background:${hex}"></i>${c.label}</span>`;
    }).join('');
    legend.classList.remove('hidden');
  } else {
    legend.classList.add('hidden');
  }
}

// ---------------------------------------------------------------- ui wiring
function addChips(containerId, texts, targetId) {
  const el = $(containerId);
  texts.forEach((t) => {
    const chip = document.createElement('button');
    chip.className = 'chip';
    chip.type = 'button';
    chip.textContent = t.length > 34 ? t.slice(0, 32) + '…' : t;
    chip.title = t;
    chip.onclick = () => { $(targetId).value = t; };
    el.appendChild(chip);
  });
}
addChips('chips', EXAMPLES_SINGLE, 'prompt');
addChips('chips-ig', EXAMPLES_INTER, 'ig-prompt');
addChips('chips-kd', [
  'A person walks forward, then turns around and walks back.',
  'A person performs a slow tai chi form.',
  'A person does jumping jacks quickly.',
  'A person picks up a box and puts it down gently.',
  'A person dances energetically, spinning around.',
], 'kd-prompt');

document.querySelectorAll('.mtab').forEach((btn) => {
  btn.onclick = () => {
    document.querySelectorAll('.mtab').forEach((b) => b.classList.remove('active'));
    btn.classList.add('active');
    currentModel = btn.dataset.model;
    document.querySelectorAll('.model-pane').forEach((p) => p.classList.add('hidden'));
    $('pane-' + currentModel).classList.remove('hidden');
    // controls only relevant to momask variants and kimodo (duration)
    const usesLength = currentModel.startsWith('momask') || currentModel === 'kimodo';
    const isMomask = currentModel.startsWith('momask');
    $('len-label').parentElement.style.display = usesLength ? '' : 'none';
    $('length').style.display = usesLength ? '' : 'none';
    document.querySelector('.range-marks').style.display = usesLength ? '' : 'none';
    $('ik-row').style.display = isMomask ? '' : 'none';
    if (currentModel === 'kimodo') {
      $('len-label').textContent = $('length').value === '0' ? '5 秒(默认)' : (+$('length').value).toFixed(1) + ' 秒';
    }
  };
});

$('length').addEventListener('input', () => {
  if (currentModel === 'kimodo') {
    $('len-label').textContent = $('length').value === '0' ? '5 秒(默认)' : (+$('length').value).toFixed(1) + ' 秒';
  } else {
    $('len-label').textContent = $('length').value === '0' ? '自动' : (+$('length').value).toFixed(1) + ' 秒';
  }
});
$('offset-x').addEventListener('input', () => {
  $('offset-label').textContent = (+$('offset-x').value).toFixed(1) + ' m';
});
$('rand-seed').onclick = () => { $('seed').value = Math.floor(Math.random() * 100000); };
$('play').onclick = () => setPlaying(!playing);

$('timeline').addEventListener('input', () => {
  if (!motion) return;
  frameIdx = Math.min(+$('timeline').value, motion.m_length - 1);
  applyFrame(frameIdx);
  syncHud();
});

$('speed').addEventListener('change', () => { speed = +$('speed').value; });

document.addEventListener('keydown', (e) => {
  if (e.code === 'Space' && !['TEXTAREA', 'INPUT', 'SELECT'].includes(document.activeElement.tagName)) {
    e.preventDefault();
    if (motion) setPlaying(!playing);
  }
});

// ---------------------------------------------------------------- history
function pushHistory(resp) {
  const rec = { ...resp };
  rec.label = MODEL_INFO[resp.model] || resp.model;
  history.unshift(rec);
  if (history.length > 8) history.pop();
  renderHistory();
}

function renderHistory() {
  const wrapEl = $('history');
  wrapEl.innerHTML = '';
  history.forEach((rec, idx) => {
    const card = document.createElement('button');
    card.className = 'hcard' + (motion && motion.id === rec.id ? ' active' : '');
    card.innerHTML = `<b>${rec.label}</b><span>${rec.seconds}s · ${rec.gen_time}s · ${(rec.text || '').slice(0, 18)}…</span>`;
    card.onclick = () => loadMotion(rec);
    wrapEl.appendChild(card);
  });
  $('history-wrap').classList.remove('hidden');
}

// ---------------------------------------------------------------- generation
async function generate() {
  let body, endpoint, spinnerNote;
  const seed = +$('seed').value || 10107;

  if (currentModel === 'momask') {
    const text = $('prompt').value.trim();
    if (!text) return setStatus('请输入动作描述', 'error');
    endpoint = '/api/generate';
    body = { text, length: +$('length').value, use_ik: $('use-ik').checked, seed,
             auto_translate: $('auto-translate').checked,
             render_video: $('render-video').checked };
    spinnerNote = 'MoMask 生成中…';
  } else if (currentModel === 'momask_dual') {
    const a = $('dual-a').value.trim(), b = $('dual-b').value.trim();
    if (!a || !b) return setStatus('请输入两个人的动作描述', 'error');
    endpoint = '/api/generate_momask_dual';
    body = { text_a: a, text_b: b, length: +$('length').value, use_ik: $('use-ik').checked, seed,
             offset_x: +$('offset-x').value, auto_translate: $('auto-translate').checked,
             render_video: $('render-video').checked };
    spinnerNote = 'MoMask 生成两个动作中…';
  } else if (currentModel === 'intergen') {
    const text = $('ig-prompt').value.trim();
    if (!text) return setStatus('请输入交互描述', 'error');
    endpoint = '/api/generate_interaction';
    body = { model: 'intergen', interaction: text, seed,
             auto_translate: $('auto-translate').checked,
             render_video: $('render-video').checked };
    spinnerNote = 'InterGen 扩散采样中（约 20~60 秒）…';
  } else if (currentModel === 'in2in') {
    const text = $('i2-interaction').value.trim();
    if (!text) return setStatus('请输入交互描述', 'error');
    endpoint = '/api/generate_interaction';
    body = {
      model: 'in2in', interaction: text,
      ind1: $('i2-ind1').value.trim(), ind2: $('i2-ind2').value.trim(), seed,
      auto_translate: $('auto-translate').checked,
      render_video: $('render-video').checked,
    };
    spinnerNote = 'in2IN 扩散采样中（约 20~60 秒）…';
  } else if (currentModel === 'kimodo') {
    const text = $('kd-prompt').value.trim();
    if (!text) return setStatus('请输入动作描述', 'error');
    endpoint = '/api/generate_kimodo';
    const len = +$('length').value;
    body = { text, duration: len > 0 ? len : 5.0, seed,
             auto_translate: $('auto-translate').checked,
             render_video: $('render-video').checked };
    spinnerNote = 'Kimodo 扩散采样中（首次需加载模型，2~5 分钟）…';
  }

  $('generate').disabled = true;
  setStatus('', '');
  $('spinner').classList.remove('hidden');
  $('spinner-text').textContent = spinnerNote;
  setPlaying(false);

  try {
    const resp = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await resp.json();
    if (!resp.ok || data.error) throw new Error(data.error || `HTTP ${resp.status}`);

    if (data.persons) {
      loadMotion(data);
      pushHistory(data);
    } else {
      // single-person endpoint: wrap into persons[] shape
      data.persons = [data.joints];
      delete data.joints;
      loadMotion(data);
      pushHistory(data);
    }
    setStatus('✓ 生成完成', 'ok');

    // show translations applied by the backend (zh -> en)
    const trEl = $('translations');
    const trs = data.translations || {};
    const trKeys = Object.keys(trs);
    if (trKeys.length) {
      const fieldLabel = { text: '提示词', text_a: '人物A', text_b: '人物B',
                           interaction: '交互', ind1: '个体A', ind2: '个体B' };
      trEl.innerHTML = trKeys.map((k) =>
        `<div class="tr-row"><span class="tr-src">${trs[k].src}</span><span class="tr-arrow">→</span><span class="tr-en">${trs[k].en}</span></div>`
      ).join('');
      trEl.classList.remove('hidden');
    } else {
      trEl.classList.add('hidden');
    }

    $('info-model').textContent = MODEL_INFO[data.model] || data.model;
    $('info-frames').textContent = data.m_length + ' 帧 @ ' + data.fps + 'fps';
    $('info-time').textContent = data.gen_time + ' s';
    const dl = $('dl-links');
    dl.innerHTML = '';
    const links = [];
    if (data.files) {
      if (data.files.video) links.push(['MP4 视频', data.files.video]);
      if (data.files.bvh) links.push(['BVH', data.files.bvh]);
      if (data.files.npy0) links.push(['NPY 人物A', data.files.npy0]);
      if (data.files.npy1) links.push(['NPY 人物B', data.files.npy1]);
      if (data.files.npy_ik) links.push(['NPY (IK)', data.files.npy_ik]);
      else if (data.files.npy) links.push(['NPY', data.files.npy]);
    }
    for (const [label, url] of links) {
      const a = document.createElement('a');
      a.href = url; a.textContent = label;
      dl.appendChild(a);
    }
    // inline video player when a rendered mp4 is available
    const videoEl = $('gen-video');
    if (data.files && data.files.video) {
      videoEl.src = data.files.video;
      videoEl.classList.remove('hidden');
    } else {
      videoEl.classList.add('hidden');
      videoEl.removeAttribute('src');
    }
    $('result-info').classList.remove('hidden');
  } catch (err) {
    setStatus('生成失败：' + err.message, 'error');
  } finally {
    $('generate').disabled = false;
    $('spinner').classList.add('hidden');
  }
}

function setStatus(msg, cls) {
  $('status').textContent = msg;
  $('status').className = 'status' + (cls ? ' ' + cls : '');
}

$('generate').onclick = generate;
document.querySelectorAll('textarea').forEach((ta) => {
  ta.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) generate();
  });
});

resize();
animate();
