// ===================================================
// HANDIT — クラウド連携（Firestore）
//
// 単語とプロトタイプは Firestore に置き、全ユーザーが読む。
// 書き込めるのは admins に登録された人だけ（＝自分だけ）。
//
// 読み取りを節約するため、単語もプロトタイプも
// 「1つのドキュメントにまとめて」保存している。
// 1ページの読み込みにつき Firestore の読み取りは2回だけ。
//
//   shared/catalog     … 単語の一覧（words.js の代わり）
//   shared/prototypes  … 即席登録した単語の埋め込み
//   contributions/{id} … 収録した生データ（学習に使う）
//   admins/{uid}       … 書き込めるユーザー
// ===================================================

const CACHE_CATALOG = 'handit_cloud_catalog';
const CACHE_PROTOS  = 'handit_cloud_prototypes';

let app = null, db = null, auth = null, fs = null;
let currentUser = null, isAdmin = false;
let ready = null;

// ---------------------------------------------------
// 初期化（firebase-config.js が無い環境でも落ちない）
// ---------------------------------------------------
export function initCloud() {
  if (ready) return ready;
  ready = (async () => {
    try {
      const [appMod, authMod, fsMod, cfg] = await Promise.all([
        import('https://www.gstatic.com/firebasejs/10.12.0/firebase-app.js'),
        import('https://www.gstatic.com/firebasejs/10.12.0/firebase-auth.js'),
        import('https://www.gstatic.com/firebasejs/10.12.0/firebase-firestore.js'),
        import('../firebase-config.js'),
      ]);
      app  = appMod.initializeApp(cfg.FIREBASE_CONFIG);
      auth = authMod.getAuth(app);
      db   = fsMod.getFirestore(app);
      fs   = fsMod;

      await new Promise(resolve => {
        let done = false;
        authMod.onAuthStateChanged(auth, async user => {
          currentUser = user;
          isAdmin = false;
          if (user) {
            try {
              const snap = await fs.getDoc(fs.doc(db, 'admins', user.uid));
              isAdmin = snap.exists();
            } catch { isAdmin = false; }
          }
          if (!done) { done = true; resolve(); }
        });
        setTimeout(() => { if (!done) { done = true; resolve(); } }, 4000);
      });
      return true;
    } catch (e) {
      console.warn('クラウドに接続できませんでした。ローカルのデータだけで動きます。', e);
      return false;
    }
  })();
  return ready;
}

export function cloudUser()  { return currentUser; }
export function cloudAdmin() { return isAdmin; }
export function cloudOnline(){ return !!db; }

// ---------------------------------------------------
// 読み込み（キャッシュを先に返し、裏で更新する）
// ---------------------------------------------------
function readCache(key) {
  try { return JSON.parse(localStorage.getItem(key) || 'null'); } catch { return null; }
}
function writeCache(key, v) {
  try { localStorage.setItem(key, JSON.stringify(v)); } catch {}
}

// 単語一覧。Firestore が使えなければ words.js の内容をそのまま返す。
export async function fetchCatalog(fallbackWords = []) {
  await initCloud();
  if (!db) return readCache(CACHE_CATALOG)?.words || fallbackWords;
  try {
    const snap = await fs.getDoc(fs.doc(db, 'shared', 'catalog'));
    if (!snap.exists()) return fallbackWords;
    const data = snap.data();
    writeCache(CACHE_CATALOG, data);
    return data.words || fallbackWords;
  } catch (e) {
    console.warn('単語一覧を取得できませんでした', e);
    return readCache(CACHE_CATALOG)?.words || fallbackWords;
  }
}

// プロトタイプ。エンコーダのバージョンが合わないものは捨てる。
export async function fetchPrototypes(encoderVersion) {
  await initCloud();
  const use = (d) => {
    if (!d) return {};
    if (encoderVersion && d.encoderVersion && d.encoderVersion !== encoderVersion) {
      console.info('エンコーダが更新されたため、古いプロトタイプは使いません');
      return {};
    }
    return d.words || {};
  };
  if (!db) return use(readCache(CACHE_PROTOS));
  try {
    const snap = await fs.getDoc(fs.doc(db, 'shared', 'prototypes'));
    if (!snap.exists()) return {};
    const data = snap.data();
    writeCache(CACHE_PROTOS, data);
    return use(data);
  } catch (e) {
    console.warn('プロトタイプを取得できませんでした', e);
    return use(readCache(CACHE_PROTOS));
  }
}

// ---------------------------------------------------
// 書き込み（admins のみ）
// ---------------------------------------------------
function requireAdmin() {
  if (!db) throw new Error('クラウドに接続していません');
  if (!currentUser) throw new Error('ログインしてください');
  if (!isAdmin) throw new Error('単語を追加する権限がありません');
}

// 単語一覧をまるごと差し替える
export async function saveCatalog(words) {
  requireAdmin();
  await fs.setDoc(fs.doc(db, 'shared', 'catalog'), {
    words, updatedAt: fs.serverTimestamp(), updatedBy: currentUser.uid,
  });
  writeCache(CACHE_CATALOG, { words });
}

// プロトタイプを1単語ぶん追加・更新する
export async function savePrototype(word, vectors, encoderVersion) {
  requireAdmin();
  const ref = fs.doc(db, 'shared', 'prototypes');
  const round = v => v.map(x => Math.round(x * 1e4) / 1e4);
  await fs.setDoc(ref, {
    encoderVersion,
    words: { [word]: vectors.map(round) },
    updatedAt: fs.serverTimestamp(),
  }, { merge: true });
}

export async function deletePrototype(word) {
  requireAdmin();
  await fs.setDoc(fs.doc(db, 'shared', 'prototypes'), {
    words: { [word]: fs.deleteField() },
  }, { merge: true });
}

// 収録した生データを1テイクずつ送る（学習に使う）
// Float32Array を base64 にして送る。配列のまま送るより小さく、索引も張られない。
export async function uploadTake(word, flatFloat32) {
  requireAdmin();
  const b64 = btoa(String.fromCharCode(...new Uint8Array(flatFloat32.buffer)));
  await fs.addDoc(fs.collection(db, 'contributions'), {
    uid: currentUser.uid,
    word,
    consent: true,
    frames: 64,
    dim: 168,
    data: b64,
    createdAt: fs.serverTimestamp(),
  });
}

// ---------------------------------------------------
// GitHub Actions を起動して学習を回す
//
// トークンはこの端末の localStorage にだけ置く。
// リポジトリには絶対に含めないこと。
// ---------------------------------------------------
const GH_KEY = 'handit_gh_token';
const GH_REPO_KEY = 'handit_gh_repo';

export function setGitHubToken(token, repo) {
  if (token) localStorage.setItem(GH_KEY, token); else localStorage.removeItem(GH_KEY);
  if (repo)  localStorage.setItem(GH_REPO_KEY, repo);
}
export function hasGitHubToken() { return !!localStorage.getItem(GH_KEY); }
export function getGitHubRepo()  { return localStorage.getItem(GH_REPO_KEY) || ''; }

export async function triggerTraining(reason = 'studio') {
  const token = localStorage.getItem(GH_KEY);
  const repo  = getGitHubRepo();
  if (!token || !repo) throw new Error('GitHubの設定がまだです');
  const res = await fetch(`https://api.github.com/repos/${repo}/dispatches`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${token}`,
      'Accept': 'application/vnd.github+json',
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ event_type: 'handit-train', client_payload: { reason } }),
  });
  if (res.status !== 204) {
    const body = await res.text().catch(() => '');
    throw new Error(`GitHub が ${res.status} を返しました ${body.slice(0, 120)}`);
  }
  return true;
}

export function actionsUrl() {
  const repo = getGitHubRepo();
  return repo ? `https://github.com/${repo}/actions` : null;
}
