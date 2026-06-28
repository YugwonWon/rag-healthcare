// WebGL 음성 오브 — ChatGPT/Gemini 음성모드 느낌의 액체/3D 일렁임.
// app.js와는 느슨하게 결합: app.js가 발행하는 상태(#mic-btn의 클래스 active/speaking)와
// 진폭(window.__orbLevel, 0~1)만 읽어서 시각화한다. WebGL 생성이 실패하면 아무것도
// 하지 않고 조용히 빠져(CSS 오브가 그대로 폴백으로 보임), 구형/저사양 기기 안전.
(function () {
  'use strict';

  // 진단은 콘솔 로그로만 남긴다(페이지 로드 시 화면에 안내 배지가 떠서 거슬린다는
  // 피드백 반영). WebGL 성공/실패는 [orb-gl] 콘솔 메시지로 확인.
  function showBadge() {}

  function init() {
    const orbBtn = document.getElementById('mic-btn');
    if (!orbBtn) return;

    const canvas = document.createElement('canvas');
    canvas.style.cssText =
      'position:absolute;inset:0;width:100%;height:100%;border-radius:50%;pointer-events:none;';
    let gl;
    try {
      gl = canvas.getContext('webgl', { premultipliedAlpha: false, antialias: true })
        || canvas.getContext('experimental-webgl');
    } catch (e) {
      showBadge('🔴 오브: WebGL 컨텍스트 예외 — ' + e.message);
    }
    if (!gl) {
      console.warn('[orb-gl] WebGL 미지원 — CSS 오브로 폴백');
      showBadge('🟡 오브: WebGL 미지원, CSS로 표시 중');
      return;
    }

    const vertSrc = 'attribute vec2 p; void main(){ gl_Position = vec4(p, 0.0, 1.0); }';
    const fragSrc = [
      'precision highp float;',
      'uniform vec2 u_res; uniform float u_time; uniform float u_level;',
      'uniform vec3 u_colA; uniform vec3 u_colB;',
      // Ashima 3D simplex noise
      'vec4 permute(vec4 x){return mod(((x*34.0)+1.0)*x,289.0);}',
      'vec4 taylorInvSqrt(vec4 r){return 1.79284291400159 - 0.85373472095314 * r;}',
      'float snoise(vec3 v){',
      '  const vec2 C = vec2(1.0/6.0, 1.0/3.0); const vec4 D = vec4(0.0,0.5,1.0,2.0);',
      '  vec3 i = floor(v + dot(v, C.yyy)); vec3 x0 = v - i + dot(i, C.xxx);',
      '  vec3 g = step(x0.yzx, x0.xyz); vec3 l = 1.0 - g;',
      '  vec3 i1 = min(g.xyz, l.zxy); vec3 i2 = max(g.xyz, l.zxy);',
      '  vec3 x1 = x0 - i1 + C.xxx; vec3 x2 = x0 - i2 + C.yyy; vec3 x3 = x0 - D.yyy;',
      '  i = mod(i, 289.0);',
      '  vec4 p = permute(permute(permute(i.z + vec4(0.0,i1.z,i2.z,1.0))',
      '        + i.y + vec4(0.0,i1.y,i2.y,1.0)) + i.x + vec4(0.0,i1.x,i2.x,1.0));',
      '  float n_=1.0/7.0; vec3 ns = n_*D.wyz - D.xzx;',
      '  vec4 j = p - 49.0*floor(p*ns.z*ns.z);',
      '  vec4 x_=floor(j*ns.z); vec4 y_=floor(j - 7.0*x_);',
      '  vec4 x = x_*ns.x + ns.yyyy; vec4 y = y_*ns.x + ns.yyyy;',
      '  vec4 h = 1.0 - abs(x) - abs(y);',
      '  vec4 b0 = vec4(x.xy, y.xy); vec4 b1 = vec4(x.zw, y.zw);',
      '  vec4 s0 = floor(b0)*2.0+1.0; vec4 s1 = floor(b1)*2.0+1.0;',
      '  vec4 sh = -step(h, vec4(0.0));',
      '  vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy; vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww;',
      '  vec3 p0=vec3(a0.xy,h.x); vec3 p1=vec3(a0.zw,h.y); vec3 p2=vec3(a1.xy,h.z); vec3 p3=vec3(a1.zw,h.w);',
      '  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0),dot(p1,p1),dot(p2,p2),dot(p3,p3)));',
      '  p0*=norm.x; p1*=norm.y; p2*=norm.z; p3*=norm.w;',
      '  vec4 m = max(0.6 - vec4(dot(x0,x0),dot(x1,x1),dot(x2,x2),dot(x3,x3)), 0.0); m=m*m;',
      '  return 42.0*dot(m*m, vec4(dot(p0,x0),dot(p1,x1),dot(p2,x2),dot(p3,x3)));',
      '}',
      // 옥타브를 줄여(3) 잔점(speckle) 대신 크고 부드러운 너울만 남긴다.
      'float fbm(vec3 p){ float a=0.55,s=0.0; for(int i=0;i<3;i++){ s+=a*snoise(p); p*=1.9; a*=0.5;} return s; }',
      'void main(){',
      '  vec2 uv = (gl_FragCoord.xy - 0.5*u_res)/min(u_res.x,u_res.y); uv*=2.0;',
      '  float r = length(uv); float t = u_time*0.18; float lvl = clamp(u_level,0.0,1.0);',
      // 둥근 실루엣 유지(가장자리 부드럽게) — 진폭에 또렷이 반응.
      '  float radius = 0.60 + 0.015*sin(t*0.7) + 0.15*lvl;',
      '  float edge = smoothstep(radius+0.04, radius-0.06, r);',
      '  float z = sqrt(max(0.0, radius*radius - r*r));',
      '  vec3 nrm = normalize(vec3(uv, z));',
      // 표면을 따라 천천히 흐르는 너울 — 낮은 주파수 fbm 두 겹을 서로 다른
      // 방향으로 느리게 흘려 ChatGPT 오브처럼 큰 파도 같은 일렁임을 만든다.
      // 흐름 자체는 소리와 무관하게 항상 같은 속도로 돈다(크기만 소리에 반응).
      '  vec3 sp = nrm * 1.0;',
      '  float f1 = fbm(sp*0.9 + vec3(0.0, 0.0, t*0.6));',
      '  float f2 = fbm(sp*1.5 + vec3(t*0.3, -t*0.22, t*0.12));',
      '  float surf = 0.5 + 0.5*sin(f1*2.1 + f2*1.5 + t*0.8);',
      '  vec3 lightDir = normalize(vec3(-0.25, 0.45, 0.85));',
      '  float diff = clamp(dot(nrm, lightDir), 0.0, 1.0);',
      '  float fres = pow(1.0 - clamp(z/max(radius,0.001),0.0,1.0), 2.0);',
      '  vec3 col = mix(u_colA, u_colB, surf);',          // 흐르는 색
      // 내부 너울을 또렷이 — 밝기를 surf로 분명하게 출렁이게 한다(일렁임 가시화).
      '  col *= 0.70 + 0.52*surf;',                       // 밝기 너울(보이는 일렁임)
      '  col *= 0.92 + 0.12*diff;',                       // 아주 약한 음영(볼록함 완화)
      '  col += fres*0.28*u_colB;',                       // 부드러운 림 글로우
      '  col += (1.0 - clamp(r/radius,0.0,1.0))*0.05*u_colA;', // 중심 은은한 발광
      '  gl_FragColor = vec4(col, edge);',
      '}'
    ].join('\n');

    function compile(type, src) {
      const sh = gl.createShader(type);
      gl.shaderSource(sh, src); gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
        const log = gl.getShaderInfoLog(sh);
        console.warn('[orb-gl] shader 컴파일 실패:', log);
        showBadge('🔴 오브: 셰이더 컴파일 실패 — ' + (log || '').slice(0, 60));
        return null;
      }
      return sh;
    }
    const vs = compile(gl.VERTEX_SHADER, vertSrc);
    const fs = compile(gl.FRAGMENT_SHADER, fragSrc);
    if (!vs || !fs) return;
    const prog = gl.createProgram();
    gl.attachShader(prog, vs); gl.attachShader(prog, fs); gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(prog);
      console.warn('[orb-gl] program 링크 실패:', log);
      showBadge('🔴 오브: 프로그램 링크 실패 — ' + (log || '').slice(0, 60));
      return;
    }
    gl.useProgram(prog);

    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 3,-1, -1,3]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(prog, 'p');
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    const U = {
      res: gl.getUniformLocation(prog, 'u_res'),
      time: gl.getUniformLocation(prog, 'u_time'),
      level: gl.getUniformLocation(prog, 'u_level'),
      colA: gl.getUniformLocation(prog, 'u_colA'),
      colB: gl.getUniformLocation(prog, 'u_colB'),
    };

    // WebGL 셰이더가 정상 동작하니 CSS 오브 내부 레이어는 숨긴다(캔버스가 다 그림).
    orbBtn.classList.add('gl');
    orbBtn.appendChild(canvas);

    // 상태별 색 — 대기(보라/인디고), 듣는중(빨강), 말하는중(초록).
    const COLORS = {
      idle:   [[0.62, 0.65, 0.98], [0.78, 0.62, 0.97]],
      listen: [[0.99, 0.55, 0.52], [0.92, 0.38, 0.36]],
      speak:  [[0.50, 0.88, 0.62], [0.32, 0.75, 0.52]],
    };
    let curA = COLORS.idle[0].slice(), curB = COLORS.idle[1].slice();

    function sizeCanvas() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2.5);
      const w = Math.max(1, Math.round(orbBtn.clientWidth * dpr));
      const h = Math.max(1, Math.round(orbBtn.clientHeight * dpr));
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w; canvas.height = h;
        gl.viewport(0, 0, w, h);
      }
    }

    const start = performance.now();
    function frame() {
      requestAnimationFrame(frame);
      // 보이지 않을 때(음성 패널 닫힘)는 그리지 않아 배터리 절약.
      if (orbBtn.offsetParent === null) return;
      sizeCanvas();

      let target;
      if (orbBtn.classList.contains('speaking')) target = COLORS.speak;
      else if (orbBtn.classList.contains('active')) target = COLORS.listen;
      else target = COLORS.idle;
      // 색 부드럽게 전환
      for (let i = 0; i < 3; i++) {
        curA[i] += (target[0][i] - curA[i]) * 0.08;
        curB[i] += (target[1][i] - curB[i]) * 0.08;
      }

      const level = parseFloat(window.__orbLevel) || 0;
      gl.uniform2f(U.res, canvas.width, canvas.height);
      gl.uniform1f(U.time, (performance.now() - start) / 1000);
      gl.uniform1f(U.level, level);
      gl.uniform3fv(U.colA, curA);
      gl.uniform3fv(U.colB, curB);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }
    requestAnimationFrame(frame);
    console.log('[orb-gl] WebGL 오브 활성화');
    showBadge('🟢 오브: WebGL 활성화됨');
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
