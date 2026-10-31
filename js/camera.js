// ===================================================
// HANDIT — カメラ（全ページ共通）
//
// これまでは各ページで
//   1) 自分で getUserMedia（解像度指定なし）
//   2) MediaPipe の Camera を start（中でもう一度 getUserMedia）
// をしていたため、カメラが2本開いていた。
// 1本目は解像度指定がないので、良いWebカメラだと 1080p/60fps で開き、
// しかも stop しても閉じられず、レッスンを始めるたびに1本ずつ増えていた。
// 高性能PCほどファンが回っていたのはこれが一番の原因。
//
// ここでは1本だけ、学習データと同じ 640x480 / 30fps で開き、
// フレームごとに onFrame を呼ぶ。stop() で確実に閉じる。
// ===================================================

export const CAM_WIDTH  = 640;
export const CAM_HEIGHT = 480;

// video: <video> 要素
// onFrame: async (video) => {}  … 新しいフレームごとに呼ばれる（前の処理が終わるまで次は呼ばない）
export async function openCamera(video, onFrame) {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      facingMode: 'user',
      width:     { ideal: CAM_WIDTH },
      height:    { ideal: CAM_HEIGHT },
      frameRate: { ideal: 30, max: 30 },
    },
  });
  video.muted = true;
  video.playsInline = true;
  video.srcObject = stream;
  await new Promise(r => {
    if (video.readyState >= 1) r();
    else video.addEventListener('loadedmetadata', r, { once: true });
  });
  await video.play().catch(() => {});

  let stopped = false;
  let lastTime = -1;
  const hasRVFC = 'requestVideoFrameCallback' in HTMLVideoElement.prototype;

  const schedule = () => {
    if (stopped) return;
    if (hasRVFC) video.requestVideoFrameCallback(tick);
    else requestAnimationFrame(tick);
  };

  async function tick() {
    if (stopped) return;
    // 同じフレームを2回処理しない（120Hz/144Hz のモニタでも30回/秒まで）
    if (video.currentTime !== lastTime && !video.paused) {
      lastTime = video.currentTime;
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
