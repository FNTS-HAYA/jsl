// ===================================================
// HANDIT — カメラ（全ページ共通）
//
// これまでは各ページで
//   1) 自分で getUserMedia
//   2) MediaPipe の Camera を start（中でもう一度 getUserMedia）
// をしていたため、カメラが2本開いていた。
// しかも1本目は stop しても閉じられず、レッスンを始めるたびに1本ずつ増えていた。
//
// ここでは1本だけ、学習データと同じ 640x480 / 30fps で開き、
// フレームごとに onFrame を呼ぶ。stop() で確実に閉じる。
//
// 高性能なカメラ（4K / 60fps 対応など）への対策:
//   ・解像度とfpsは「ideal」なので、カメラによっては無視して
//     1080p や 60fps で来ることがある。開いた後に実際の値を確かめ、
//     大きすぎたら applyConstraints で下げ直す。
//   ・それでも 60fps で来る場合に備えて、処理は時間で間引き、
//     MediaPipe に渡すのは最大 30回/秒 にする。
//   ・実際に開いた値はコンソールに [camera] で出す。
// ===================================================

export const CAM_WIDTH  = 640;
export const CAM_HEIGHT = 480;
export const MAX_FPS    = 30;

// 前の処理からこれだけ空いていないフレームは捨てる。
// 60fps（16.7ms間隔）なら1枚おき＝30回/秒になり、
// 30fps（33ms間隔、多少ぶれる）なら全部処理される値にしてある。
const MIN_GAP_MS = 25;

// video: <video> 要素
// onFrame: async (video) => {}  … 新しいフレームごとに呼ばれる（前の処理が終わるまで次は呼ばない）
export async function openCamera(video, onFrame) {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      facingMode: 'user',
      width:      { ideal: CAM_WIDTH },
      height:     { ideal: CAM_HEIGHT },
      frameRate:  { ideal: MAX_FPS, max: MAX_FPS },
      resizeMode: { ideal: 'crop-and-scale' },   // カメラが640x480を持っていなくてもブラウザで縮める
    },
  });

  const track = stream.getVideoTracks()[0];
  await shrinkIfNeeded(track);

  video.muted = true;
  video.playsInline = true;
  video.srcObject = stream;
  await new Promise(r => {
    if (video.readyState >= 1) r();
    else video.addEventListener('loadedmetadata', r, { once: true });
  });
  await video.play().catch(() => {});

  const s = track.getSettings ? track.getSettings() : {};
  console.info(`[camera] ${video.videoWidth}x${video.videoHeight} ${s.frameRate ? Math.round(s.frameRate) + 'fps' : ''} (${track.label || 'camera'})`);

  let stopped = false;
  let lastTime = -1;
  let lastRun = 0;
  const hasRVFC = 'requestVideoFrameCallback' in HTMLVideoElement.prototype;

  const schedule = () => {
    if (stopped) return;
    if (hasRVFC) video.requestVideoFrameCallback(tick);
    else requestAnimationFrame(tick);
  };

  async function tick() {
    if (stopped) return;
    const now = performance.now();
    // 同じフレームを2回処理しない ＆ 30回/秒を超えない（60fpsのカメラや144Hzのモニタでも）
    if (video.currentTime !== lastTime && !video.paused && now - lastRun >= MIN_GAP_MS) {
      lastTime = video.currentTime;
      lastRun = now;
      try { await onFrame(video); } catch (e) { console.error(e); }
    }
    schedule();
  }
  schedule();

  return {
    stream,
    width:  video.videoWidth,
    height: video.videoHeight,
    stop() {
      stopped = true;
      stream.getTracks().forEach(t => t.stop());
      if (video.srcObject === stream) video.srcObject = null;
    },
  };
}

// カメラが ideal を無視して大きな解像度・高いfpsで開いたときに下げ直す
async function shrinkIfNeeded(track) {
  if (!track || !track.getSettings || !track.applyConstraints) return;
  const s = track.getSettings();
  const tooBig  = (s.width || 0) > CAM_WIDTH * 1.5 || (s.height || 0) > CAM_HEIGHT * 1.5;
  const tooFast = (s.frameRate || 0) > MAX_FPS + 1;
  if (!tooBig && !tooFast) return;
  try {
    await track.applyConstraints({
      width:      { ideal: CAM_WIDTH },
      height:     { ideal: CAM_HEIGHT },
      frameRate:  { max: MAX_FPS },
      resizeMode: 'crop-and-scale',
    });
  } catch (e) {
    console.warn('[camera] 解像度を下げられませんでした', e);
  }
}
