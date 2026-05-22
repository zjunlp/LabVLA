/* =========================================================
   LabVLA · Project Page — Interactions
   ========================================================= */

/* ---------------------------------------------------------
   Hero 视频/图片墙配置
   ---------------------------------------------------------
   · 外圈：frames/0.jpg … 连续编号（保持原始 8 列 × 9 图 × 2 份方格结构）
   · 中央 2×2：assets/media/gifs/01–04.mp4
   · 下方各板块 .placeholder 保持 HTML 占位，不自动填图
   支持的扩展名：.mp4 / .webm（用 <video>）；.gif / .jpg / .png（用 <img>）
*/

const FRAME_DIR = 'assets/media/frames';
const FRAME_COUNT = 104;
const FRAME_VERSION = '20260521-gridrestore';
const HERO_SMALL_VIDEOS = Array.from(
  { length: FRAME_COUNT },
  (_, i) => `${FRAME_DIR}/${i}.jpg?v=${FRAME_VERSION}`
);

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

/* 外圈小图格子总数 */
const HERO_SMALL_COUNT = 96;

(() => {
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const I18N = {
    en: {
      'meta.description': 'LabVLA project page for vision-language-action research.',
      'meta.og_title': 'LabVLA',
      'meta.og_description': 'LabVLA project page for vision-language-action research.',
      'lang.toggle': '中文',
      'lang.aria': 'Switch to Chinese',
      'nav.features': 'Highlights',
      'nav.dataset': 'Dataset',
      'nav.method': 'Method',
      'nav.results': 'Results',
      'nav.sim2real': 'Sim-to-Real',
      'nav.bibtex': 'Citation',
      'nav.team': 'Team',
      'link.paper': 'Paper',
      'link.code': 'Code',
      'link.dataset': 'Dataset',
      'link.demo': 'Demo Video',
      'features.title': 'Highlights',
      'features.subtitle': "Add a concise summary of the project's key differentiators here.",
      'placeholder.demo_square': 'Demo · 1:1',
      'features.demo1': 'Highlight 1 demo video/image',
      'features.demo2': 'Highlight 2 demo video/image',
      'features.demo3': 'Highlight 3 demo video/image',
      'features.card1.title': 'Highlight 1 Title',
      'features.card2.title': 'Highlight 2 Title',
      'features.card3.title': 'Highlight 3 Title',
      'features.card1.desc': 'Add a one-sentence description of highlight 1.',
      'features.card2.desc': 'Add a one-sentence description of highlight 2.',
      'features.card3.desc': 'Add a one-sentence description of highlight 3.',
      'dataset.title': 'Dataset Overview',
      'dataset.subtitle': 'Add a one-sentence overview of dataset scale, diversity, and composition.',
      'stat.number': 'Number',
      'dataset.metric1': 'Metric 1',
      'dataset.metric2': 'Metric 2',
      'dataset.metric3': 'Metric 3',
      'dataset.metric4': 'Metric 4',
      'dataset.metric5': 'Metric 5',
      'dataset.metric6': 'Metric 6',
      'placeholder.figure_16_7': 'Figure · 16:7',
      'dataset.chart': 'Dataset statistics figure (task/object/embodiment distributions)',
      'dataset.caption': 'Figure: add caption here.',
      'method.title': 'Method',
      'method.subtitle': 'Add a concise subtitle for the method section.',
      'placeholder.figure_16_9': 'Figure · 16:9',
      'method.figure': 'Method / pipeline architecture figure',
      'method.subhead': 'Method Subheading',
      'method.desc': 'Add a 3-5 line method description covering the core idea, inputs/outputs, and key innovations.',
      'method.point1': 'Key point 1',
      'method.point2': 'Key point 2',
      'method.point3': 'Key point 3',
      'results.title': 'Qualitative Results',
      'results.subtitle': 'Add representative rollouts across tasks and embodiments.',
      'placeholder.video': 'Video',
      'label.task': 'Task',
      'results.video1': 'Task video 1',
      'results.video2': 'Task video 2',
      'results.video3': 'Task video 3',
      'results.video4': 'Task video 4',
      'results.video5': 'Task video 5',
      'results.video6': 'Task video 6',
      'results.task1': 'Task name 1',
      'results.task2': 'Task name 2',
      'results.task3': 'Task name 3',
      'results.task4': 'Task name 4',
      'results.task5': 'Task name 5',
      'results.task6': 'Task name 6',
      'results.quant': 'Quantitative Comparison',
      'table.method': 'Method',
      'table.metric1': 'Metric 1',
      'table.metric2': 'Metric 2',
      'table.metric3': 'Metric 3',
      'table.avg': 'Average',
      'table.baseline1': 'Baseline 1',
      'table.baseline2': 'Baseline 2',
      'table.ours': 'Ours',
      'table.value': 'Value',
      'sim.title': 'Sim-to-Real Transfer',
      'sim.subtitle': 'Add a summary of training with simulation data and deploying directly in the real world.',
      'label.sim': 'Simulation',
      'label.real': 'Real',
      'label.task_name': 'Task name',
      'sim.video1': 'Simulation training video 1',
      'sim.video2': 'Simulation training video 2',
      'sim.real1': 'Real-world execution video 1',
      'sim.real2': 'Real-world execution video 2',
      'bib.title': 'Citation',
      'bib.subtitle': 'If this work is useful to you, please consider citing it.',
      'bib.copy': 'Copy',
      'bib.copied': 'Copied',
      'bib.copy_aria': 'Copy BibTeX',
      'bib.code': '@article{labvla2026,\n  title   = {Paper Title},\n  author  = {Author List},\n  journal = {arXiv preprint arXiv:2026.xxxxx},\n  year    = {2026}\n}',
      'team.title': 'Team',
      'team.subtitle': 'Core contributors and advisors.',
      'team.contributors': 'Core Contributors',
      'team.advisors': 'Advisors',
      'team.avatar': 'Avatar',
      'team.contributor1': 'Contributor 1',
      'team.contributor2': 'Contributor 2',
      'team.contributor3': 'Contributor 3',
      'team.contributor4': 'Contributor 4',
      'team.contributor5': 'Contributor 5',
      'team.contributor6': 'Contributor 6',
      'team.advisor1': 'Advisor 1',
      'team.advisor2': 'Advisor 2',
      'team.advisor3': 'Advisor 3',
      'ack.title': 'Acknowledgements',
      'ack.text': 'Add acknowledgements here, including funding, collaborators, and template credits.',
      'footer.desc': 'Add a concise project description here.',
      'footer.links': 'Links',
      'footer.contact': 'Contact',
      'footer.email': 'Contact Email',
      'footer.social': 'Twitter / Weibo',
      'footer.rights': 'LabVLA. All rights reserved.',
      'footer.note': 'Replace placeholders before release.',
    },
    zh: {
      'meta.description': 'LabVLA 视觉-语言-动作研究项目页面。',
      'meta.og_title': 'LabVLA',
      'meta.og_description': 'LabVLA 视觉-语言-动作研究项目页面。',
      'lang.toggle': 'English',
      'lang.aria': '切换到英文',
      'nav.features': '亮点',
      'nav.dataset': '数据集',
      'nav.method': '方法',
      'nav.results': '结果',
      'nav.sim2real': '仿真到真实',
      'nav.bibtex': '引用',
      'nav.team': '团队',
      'link.paper': '论文',
      'link.code': '代码',
      'link.dataset': '数据集',
      'link.demo': '演示视频',
      'features.title': '核心亮点',
      'features.subtitle': '此处填写：本节副标题，简述项目的差异化优势。',
      'placeholder.demo_square': '演示 · 1:1',
      'features.demo1': '亮点 1 演示视频/图',
      'features.demo2': '亮点 2 演示视频/图',
      'features.demo3': '亮点 3 演示视频/图',
      'features.card1.title': '亮点 1 标题',
      'features.card2.title': '亮点 2 标题',
      'features.card3.title': '亮点 3 标题',
      'features.card1.desc': '此处填写：亮点 1 一句话描述。',
      'features.card2.desc': '此处填写：亮点 2 一句话描述。',
      'features.card3.desc': '此处填写：亮点 3 一句话描述。',
      'dataset.title': '数据集概览',
      'dataset.subtitle': '此处填写：数据集规模、多样性与构成的一句话总览。',
      'stat.number': '数字',
      'dataset.metric1': '指标 1',
      'dataset.metric2': '指标 2',
      'dataset.metric3': '指标 3',
      'dataset.metric4': '指标 4',
      'dataset.metric5': '指标 5',
      'dataset.metric6': '指标 6',
      'placeholder.figure_16_7': '图 · 16:7',
      'dataset.chart': '数据集统计图（任务/物体/本体分布等）',
      'dataset.caption': '图：此处填写图注。',
      'method.title': '方法',
      'method.subtitle': '此处填写：方法部分副标题。',
      'placeholder.figure_16_9': '图 · 16:9',
      'method.figure': '方法 / Pipeline 架构图',
      'method.subhead': '方法子标题',
      'method.desc': '此处填写：方法描述（3-5 行），介绍核心思想、输入输出、关键创新点。',
      'method.point1': '要点 1',
      'method.point2': '要点 2',
      'method.point3': '要点 3',
      'results.title': '定性结果',
      'results.subtitle': '此处填写：跨任务 / 跨本体的代表性 rollout 展示。',
      'placeholder.video': '视频',
      'label.task': '任务',
      'results.video1': '任务视频 1',
      'results.video2': '任务视频 2',
      'results.video3': '任务视频 3',
      'results.video4': '任务视频 4',
      'results.video5': '任务视频 5',
      'results.video6': '任务视频 6',
      'results.task1': '任务名 1',
      'results.task2': '任务名 2',
      'results.task3': '任务名 3',
      'results.task4': '任务名 4',
      'results.task5': '任务名 5',
      'results.task6': '任务名 6',
      'results.quant': '定量对比',
      'table.method': '方法',
      'table.metric1': '指标 1',
      'table.metric2': '指标 2',
      'table.metric3': '指标 3',
      'table.avg': '平均',
      'table.baseline1': '基线 1',
      'table.baseline2': '基线 2',
      'table.ours': '本方法',
      'table.value': '数值',
      'sim.title': '仿真到真实迁移',
      'sim.subtitle': '此处填写：仅用仿真数据训练，直接部署到真实世界。',
      'label.sim': '仿真',
      'label.real': '真实',
      'label.task_name': '任务名',
      'sim.video1': '仿真训练视频 1',
      'sim.video2': '仿真训练视频 2',
      'sim.real1': '真实世界执行视频 1',
      'sim.real2': '真实世界执行视频 2',
      'bib.title': '引用',
      'bib.subtitle': '如果本工作对你有帮助，欢迎引用。',
      'bib.copy': '复制',
      'bib.copied': '已复制',
      'bib.copy_aria': '复制 BibTeX',
      'bib.code': '@article{labvla2026,\n  title   = {论文标题},\n  author  = {作者列表},\n  journal = {arXiv preprint arXiv:2026.xxxxx},\n  year    = {2026}\n}',
      'team.title': '团队',
      'team.subtitle': '核心贡献者与指导老师。',
      'team.contributors': '核心贡献者',
      'team.advisors': '指导老师',
      'team.avatar': '头像',
      'team.contributor1': '贡献者 1',
      'team.contributor2': '贡献者 2',
      'team.contributor3': '贡献者 3',
      'team.contributor4': '贡献者 4',
      'team.contributor5': '贡献者 5',
      'team.contributor6': '贡献者 6',
      'team.advisor1': '导师 1',
      'team.advisor2': '导师 2',
      'team.advisor3': '导师 3',
      'ack.title': '致谢',
      'ack.text': '此处填写：致谢内容（资助、合作机构、模板出处等）。',
      'footer.desc': '此处填写：底部一句话简介。',
      'footer.links': '链接',
      'footer.contact': '联系',
      'footer.email': '联系邮箱',
      'footer.social': 'Twitter / 微博',
      'footer.rights': 'LabVLA. 保留所有权利。',
      'footer.note': '替换占位即可上线。',
    },
  };

  let currentLang = 'en';
  const setMetaContent = (selector, value) => {
    const el = $(selector);
    if (el && value) el.setAttribute('content', value);
  };
  const applyLanguage = (lang) => {
    const dict = I18N[lang] || I18N.en;
    currentLang = lang;
    document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en';
    document.title = dict['meta.og_title'] || 'LabVLA';
    setMetaContent('meta[name="description"]', dict['meta.description']);
    setMetaContent('meta[property="og:title"]', dict['meta.og_title']);
    setMetaContent('meta[property="og:description"]', dict['meta.og_description']);
    $$('[data-i18n]').forEach((el) => {
      const text = dict[el.dataset.i18n];
      if (typeof text === 'string') el.textContent = text;
    });
    $$('[data-i18n-aria]').forEach((el) => {
      const text = dict[el.dataset.i18nAria];
      if (typeof text === 'string') el.setAttribute('aria-label', text);
    });
  };

  /* ---------- 年份 ---------- */
  const yearEl = $('#year');
  if (yearEl) yearEl.textContent = new Date().getFullYear();

  applyLanguage('en');
  const langToggle = $('#langToggle');
  if (langToggle) {
    langToggle.addEventListener('click', () => {
      applyLanguage(currentLang === 'en' ? 'zh' : 'en');
    });
  }

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
        img.loading = opts.loading || 'lazy';
        img.decoding = 'async';
        if (opts.fetchPriority) img.fetchPriority = opts.fetchPriority;
        img.alt = '';
        if (opts.deferSrc) img.dataset.src = mediaPath;
        else img.src = mediaPath;
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

  const seededRandom = (seed) => {
    let state = seed >>> 0;
    return () => {
      state = (state * 1664525 + 1013904223) >>> 0;
      return state / 4294967296;
    };
  };

  /* Deterministic shuffle keeps the one-shot preview aligned with the real tiles. */
  const shuffled = (arr) => {
    const a = arr.slice();
    const random = seededRandom(20260521);
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(random() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  };

  const mosaicBg = $('#mosaicBg');
  const tileLoadPromises = [];
  if (mosaicBg) {
    if (HERO_SMALL_VIDEOS.length) {
      // 多列纵向无限流动
      mosaicBg.classList.add('mosaic-bg--flow', 'mosaic-bg--priming');
      const COLUMN_COUNT = 8;     // 列数
      const PER_COLUMN = 9;       // 每列基础图片数（实际 DOM 复制 2 份以实现无缝循环）
      const pool = shuffled(HERO_SMALL_VIDEOS);
      const frag = document.createDocumentFragment();
      let cursor = 0;
      for (let c = 0; c < COLUMN_COUNT; c++) {
        const col = document.createElement('div');
        col.className = 'mosaic-col';
        col.style.setProperty('--dur', `${36 + (c % 4) * 9}s`);
        col.style.setProperty('--dir', c % 2 === 0 ? 'normal' : 'reverse');
        col.style.setProperty('--delay', `-${(c * 3.4).toFixed(2)}s`);

        const items = [];
        for (let i = 0; i < PER_COLUMN; i++) {
          items.push(pool[(cursor++) % pool.length]);
        }
        [...items, ...items].forEach((src) => {
          const tile = document.createElement('div');
          tile.className = 'tile';
          const img = document.createElement('img');
          img.loading = 'eager';
          img.decoding = 'async';
          img.fetchPriority = 'low';
          img.alt = '';
          tileLoadPromises.push(new Promise(resolve => {
            const done = () => resolve();
            img.addEventListener('load', done, { once: true });
            img.addEventListener('error', done, { once: true });
          }));
          img.src = src;
          tile.appendChild(img);
          col.appendChild(tile);
        });
        frag.appendChild(col);
      }
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

  const revealMosaicTiles = () => {
    if (!mosaicBg) return;
    mosaicBg.classList.remove('mosaic-bg--priming');
    mosaicBg.classList.add('mosaic-bg--ready');
  };
  if (tileLoadPromises.length) {
    Promise.allSettled(tileLoadPromises).then(revealMosaicTiles);
  } else {
    revealMosaicTiles();
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
  if (tileLoadPromises.length) {
    const tilesReady = Promise.allSettled(tileLoadPromises);
    const maxWait = new Promise(resolve => setTimeout(resolve, 2200));
    Promise.race([tilesReady, maxWait]).then(startDeferredVideos);
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
