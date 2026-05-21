/* =========================================================
   LabVLA · Project Page — Interactions
   ========================================================= */

/* ---------------------------------------------------------
   Hero 视频/图片墙配置
   ---------------------------------------------------------
   · 外圈：mosaic/column-01.jpg … column-08.jpg（合成列图，避免首屏几十个请求）
   · 中央 2×2：assets/media/gifs/01–04.mp4
   · 下方各板块 .placeholder 保持 HTML 占位，不自动填图
   支持的扩展名：.mp4 / .webm（用 <video>）；.gif / .jpg / .png（用 <img>）
*/

/* 中央 2×2 四个动图（已由 GIF 转 MP4，体积 ~96% 减少） */
const HERO_BIG_VIDEOS = [
  'assets/media/gifs/01.mp4',
  'assets/media/gifs/02.mp4',
  'assets/media/gifs/04.mp4',
  'assets/media/gifs/03.mp4',
];

const HERO_BIG_POSTERS = [
  'assets/media/posters/01.jpg',
  'assets/media/posters/02.jpg',
  'assets/media/posters/04.jpg',
  'assets/media/posters/03.jpg',
];

const HERO_MOSAIC_COLUMNS = Array.from(
  { length: 8 },
  (_, i) => `assets/media/mosaic/column-${String(i + 1).padStart(2, '0')}.jpg`
);

/* 外圈小图格子总数 */
const HERO_SMALL_COUNT = 96;

(() => {
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  /* ---------- 年份 ---------- */
  const yearEl = $('#year');
  if (yearEl) yearEl.textContent = new Date().getFullYear();

  /* ---------- 导航滚动态 ---------- */
  const nav = $('.nav');
  const onScroll = () => {
    if (!nav) return;
    if (window.scrollY > 12) nav.classList.add('scrolled');
    else nav.classList.remove('scrolled');
  };
  onScroll();
  window.addEventListener('scroll', onScroll, { passive: true });

  /* ---------- Hero 视频/图片墙生成 ---------- */
  const isImageSrc = (s) => /\.(gif|jpe?g|png|webp|avif)(\?|#|$)/i.test(s);

  const buildTileInner = (mediaPath, hue, delay, dur, opts = {}) => {
    if (mediaPath) {
      if (isImageSrc(mediaPath)) {
        const img = document.createElement('img');
        img.src = mediaPath;
        img.loading = opts.loading || 'lazy';
        img.decoding = 'async';
        if (opts.fetchPriority) img.fetchPriority = opts.fetchPriority;
        img.alt = '';
        return img;
      }
      const v = document.createElement('video');
      if (opts.poster) v.poster = opts.poster;
      v.autoplay = true;
      v.loop = true;
      v.muted = true;
      v.playsInline = true;
      v.preload = opts.preload || 'metadata';
      if (opts.deferSrc) {
        v.dataset.src = mediaPath;
      } else {
        v.src = mediaPath;
      }
      return v;
    }
    const ph = document.createElement('div');
    ph.className = 'tile-anim';
    ph.style.setProperty('--h', String(hue));
    ph.style.setProperty('--delay', `-${delay.toFixed(2)}s`);
    ph.style.setProperty('--dur', `${dur.toFixed(2)}s`);
    return ph;
  };

  const mosaicBg = $('#mosaicBg');
  const stripLoadPromises = [];
  if (mosaicBg) {
    if (HERO_MOSAIC_COLUMNS.length) {
      // 多列纵向无限流动
      mosaicBg.classList.add('mosaic-bg--flow');
      const frag = document.createDocumentFragment();
      HERO_MOSAIC_COLUMNS.forEach((src, c) => {
        const col = document.createElement('div');
        col.className = 'mosaic-col';
        col.style.setProperty('--dur', `${36 + (c % 4) * 9}s`);
        col.style.setProperty('--dir', c % 2 === 0 ? 'normal' : 'reverse');
        col.style.setProperty('--delay', `-${(c * 3.4).toFixed(2)}s`);

        const img = document.createElement('img');
        img.className = 'mosaic-strip';
        img.loading = 'eager';
        img.decoding = 'async';
        img.fetchPriority = 'high';
        img.alt = '';
        stripLoadPromises.push(new Promise(resolve => {
          const done = () => {
            img.classList.add('is-loaded');
            resolve();
          };
          if (img.complete) done();
          else {
            img.addEventListener('load', done, { once: true });
            img.addEventListener('error', done, { once: true });
          }
        }));
        img.src = src;
        col.appendChild(img);
        frag.appendChild(col);
      });
      mosaicBg.appendChild(frag);
    } else {
      // 兜底：占位动画格子
      const frag = document.createDocumentFragment();
      for (let i = 0; i < HERO_SMALL_COUNT; i++) {
        const tile = document.createElement('div');
        tile.className = 'tile';
        const hue = Math.floor(180 + Math.random() * 180);
        const delay = Math.random() * 8;
        const dur = 7 + Math.random() * 6;
        tile.appendChild(buildTileInner(undefined, hue, delay, dur));
        frag.appendChild(tile);
      }
      mosaicBg.appendChild(frag);
    }
  }

  /* 中央 2×2 大动图 */
  $$('.big-tile').forEach((el, idx) => {
    const hue = [260, 200, 320, 160][idx] ?? 240;
    const delay = idx * 1.7;
    const dur = 10;
    el.appendChild(buildTileInner(HERO_BIG_VIDEOS[idx], hue, delay, dur, {
      poster: HERO_BIG_POSTERS[idx],
      preload: 'none',
      deferSrc: true,
    }));
  });

  const startDeferredVideos = () => {
    $$('video[data-src]').forEach((video) => {
      const src = video.dataset.src;
      if (!src) return;
      video.src = src;
      delete video.dataset.src;
      video.load();
      video.play().catch(() => {});
    });
  };
  if (stripLoadPromises.length) {
    const stripsReady = Promise.allSettled(stripLoadPromises);
    const maxWait = new Promise(resolve => setTimeout(resolve, 1400));
    Promise.race([stripsReady, maxWait]).then(startDeferredVideos);
  } else {
    startDeferredVideos();
  }

  /* ---------- 锚点高亮 ---------- */
  const sections = $$('main section[id]');
  const navLinks = $$('.nav-link');
  const linkBySection = new Map(
    navLinks
      .map(a => [a.getAttribute('href')?.replace('#', ''), a])
      .filter(([id]) => !!id)
  );
  const io = new IntersectionObserver(
    entries => {
      entries.forEach(e => {
        if (e.isIntersecting) {
          navLinks.forEach(a => a.classList.remove('active'));
          const link = linkBySection.get(e.target.id);
          if (link) link.classList.add('active');
        }
      });
    },
    { rootMargin: '-45% 0px -50% 0px', threshold: 0 }
  );
  sections.forEach(s => io.observe(s));

  /* ---------- 滚动出现动画 ---------- */
  const revealTargets = $$('.section, .feature-card, .stat-card, .result-tile, .member-card, .panel');
  revealTargets.forEach(el => el.classList.add('reveal'));
  const revealIO = new IntersectionObserver(
    entries => {
      entries.forEach(e => {
        if (e.isIntersecting) {
          e.target.classList.add('in');
          revealIO.unobserve(e.target);
        }
      });
    },
    { threshold: 0.08 }
  );
  revealTargets.forEach(el => revealIO.observe(el));

  /* ---------- BibTeX 复制 ---------- */
  $$('.copy-btn').forEach(btn => {
    btn.addEventListener('click', async () => {
      const target = $(btn.dataset.copyTarget);
      if (!target) return;
      try {
        await navigator.clipboard.writeText(target.innerText.trim());
        btn.classList.add('copied');
        setTimeout(() => btn.classList.remove('copied'), 1800);
      } catch {
        // fallback
        const r = document.createRange();
        r.selectNodeContents(target);
        const sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(r);
      }
    });
  });
})();
