/* =========================================================
   LabVLA · Project Page — Interactions
   ========================================================= */

/* ---------------------------------------------------------
   Hero mosaic media config
   ---------------------------------------------------------
   · Outer ring: frames/0.jpg … sequential numbering (6 cols × 4 imgs × 2 copies in DOM)
   · Center 2×2: assets/media/gifs/01–04.mp4
   · Section .placeholder blocks stay as HTML stubs; not auto-filled
   Supported extensions: .mp4 / .webm (<video>); .gif / .jpg / .png (<img>)
*/

const FRAME_DIR = 'assets/media/frames';
const FRAME_COUNT = 104;
const FRAME_VERSION = '20260611-compress-v1';
const HERO_SMALL_VIDEOS = Array.from(
  { length: FRAME_COUNT },
  (_, i) => `${FRAME_DIR}/${i}.jpg?v=${FRAME_VERSION}`
);

/* Center 2×2 hero clips (converted from GIF to MP4, ~96% smaller) */
const HERO_BIG_VIDEOS = [
  'assets/media/gifs/01.mp4',
  'assets/media/gifs/02.mp4',
  'assets/media/gifs/04.mp4',
  'assets/media/gifs/03.mp4?v=20260617-trim4',
];

const HERO_BIG_POSTERS = [
  'assets/media/posters/01.jpg',
  'assets/media/posters/02.jpg',
  'assets/media/posters/04.jpg',
  'assets/media/posters/03.jpg?v=20260617-trim4',
];

/* object-position per tile (null = center default) */
const HERO_BIG_OBJECT_POSITION = [
  null,
  null,
  null,
  '49.8% 98.6%',
];

/* Fallback outer-ring tile count when no media paths are set */
const HERO_SMALL_COUNT = 96;

(() => {
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const I18N = {
    en: {
      'meta.description': 'LabVLA project page for vision-language-action research.',
      'meta.og_title': 'LabVLA',
      'meta.og_description': 'LabVLA project page for vision-language-action research.',
      'lang.toggle': 'Chinese',
      'lang.aria': 'Switch to Chinese',
      'nav.features': 'Robots',
      'nav.dataset': 'Tasks',
      'nav.workflow': 'Workflow',
      'nav.dual_tasks': 'Dual-arm',
      'nav.method': 'Randomization',
      'nav.assets': 'Assets',
      'nav.results': 'Scenes',
      'nav.sim2real': 'Sim-to-Real',
      'nav.bibtex': 'Citation',
      'nav.team': 'Team',
      'link.paper': 'Paper',
      'link.homepage': 'Homepage',
      'link.code': 'Code',
      'link.model': 'Model',
      'link.contact': 'Contact',
      'link.dataset': 'Dataset',
      'link.demo': 'Demo Video',
      'features.title': 'A Programmable Workflow and Data Engine',
      'features.subtitle': 'An end-to-end multi-arm data engine — spanning tasks, workflows, randomization, assets, and scene generation.',
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
      'robots.single.title': 'Single-arm Robots',
      'robots.single.count': '9 platforms',
      'robots.single.tag': 'Single-arm',
      'robots.dual.title': 'Dual-arm Robots',
      'robots.dual.count': '3 platforms',
      'robots.dual.tag': 'Dual-arm',
      'robots.image_slot': 'Image slot',
      'robots.single.01': 'Franka',
      'robots.single.02': 'UR5e',
      'robots.single.03': 'UR16e',
      'robots.single.04': 'FR3',
      'robots.single.05': 'Festo',
      'robots.single.06': 'Rizon 4',
      'robots.single.07': 'Single-arm Robot 07',
      'robots.dual.01': 'Dual-arm Robot 01',
      'robots.dual.02': 'Dual-arm Robot 02',
      'robots.dual.03': 'Dual-arm Robot 03',
      'dataset.title': 'Multi-task Capabilities',
      'dataset.subtitle': 'A suite of atomic manipulation tasks can be demonstrated independently or composed into complete lab workflows.',
      'stat.number': 'Number',
      'dataset.metric1': 'Metric 1',
      'dataset.metric2': 'Metric 2',
      'dataset.metric3': 'Metric 3',
      'dataset.metric4': 'Metric 4',
      'dataset.metric5': 'Metric 5',
      'dataset.metric6': 'Metric 6',
      'tasks.atomic.tag': 'Atomic Task',
      'tasks.media_slot': 'Image / video slot',
      'tasks.atomic.01': 'Open Door',
      'tasks.atomic.02': 'Close Door',
      'tasks.atomic.03': 'Place',
      'tasks.atomic.04': 'Pour',
      'tasks.atomic.05': 'Pick',
      'tasks.atomic.06': 'Shake',
      'tasks.atomic.07': 'Stir',
      'tasks.atomic.08': 'Heat Liquid',
      'single_tasks.title': 'Single-arm Task Showcase',
      'single_tasks.subtitle': 'Atomic manipulation skills demonstrated across single-arm robot platforms.',
      'workflow.kicker': 'Composable Workflow',
      'workflow.page_title': 'Workflow Demonstrations',
      'workflow.page_subtitle': 'Atomic skills compose into long-horizon lab procedures across dual-arm manipulation, mobile navigation, and multi-step liquid handling.',
      'workflow.title': 'Atomic tasks can be chained into a complete lab workflow',
      'workflow.desc': 'A complete rollout composes multiple atomic skills into a longer executable lab workflow.',
      'workflow.card1': 'Franka Mobile Navigation',
      'workflow.card2': 'Franka 16-step Workflow',
      'workflow.tag.mobile': 'Mobile',
      'workflow.tag.long': '16-step',
      'workflow.step1': 'Task 01',
      'workflow.step2': 'Task 04',
      'workflow.step3': 'Task 07',
      'workflow.step4': 'Task 10',
      'dual_tasks.title': 'Dual-arm Task Showcase',
      'dual_tasks.subtitle': 'Bimanual manipulation demonstrations with coordinated dual-arm execution.',
      'dual_tasks.card1': 'Lift2',
      'dual_tasks.card2': 'Split Aloha',
      'placeholder.figure_16_7': 'Figure · 16:7',
      'dataset.chart': 'Dataset statistics figure (task/object/embodiment distributions)',
      'dataset.caption': 'Figure: add caption here.',
      'method.title': 'Randomization',
      'method.subtitle': 'We randomize scene layout, visual clutter, camera viewpoint, object appearance, lighting, and spatial placement to improve robustness.',
      'placeholder.figure_16_9': 'Figure · 16:9',
      'method.figure': 'Method / pipeline architecture figure',
      'method.subhead': 'Diverse visual conditions from the same task state.',
      'method.desc': 'Each task can be rendered under varied environments so the policy learns stable task semantics instead of overfitting to a single scene.',
      'method.point1': 'Randomize lighting, tabletop texture, and surrounding scene.',
      'method.point2': 'Change camera pose and object positions.',
      'method.point3': 'Add obstacles and object appearance variation.',
      'random.kicker': 'Domain Randomization',
      'random.base': 'Base Scene',
      'random.advanced': 'Advanced',
      'random.click': 'Click to switch',
      'random.scene': 'Scene',
      'random.clutter': 'Visual clutter',
      'random.camera': 'Camera',
      'random.object': 'Object',
      'random.lighting': 'Lighting',
      'random.spatial': 'Spatial',
      'assets.title': 'Asset Generation',
      'assets.subtitle': 'Generate reusable objects, containers, tools, and scene props for composing lab environments.',
      'assets.kicker': 'Reusable Assets',
      'assets.heading': 'Assets are generated independently, then assembled into scenes.',
      'assets.desc': 'Reserve this section for generated assets such as lab tools, containers, instruments, fixtures, and task-specific objects.',
      'assets.sample_tag': 'Asset',
      'assets.sample1': 'Generated asset 01',
      'assets.sample2': 'Generated asset 02',
      'assets.sample3': 'Generated asset 03',
      'assets.sample4': 'Generated asset 04',
      'assets.caption1': 'Fume Hoods 01',
      'assets.caption2': 'Fume Hoods 02',
      'assets.caption3': 'Fume Hoods 03',
      'assets.caption4': 'Fume Hoods 04',
      'assets.caption5': 'Ball Mills 01',
      'assets.caption6': 'Ball Mills 02',
      'assets.caption7': 'Ball Mills 03',
      'assets.caption8': 'Ball Mills 04',
      'assets.caption9': 'Critical Point Dryers 01',
      'assets.caption10': 'Critical Point Dryers 02',
      'assets.caption11': 'Critical Point Dryers 03',
      'assets.caption12': 'Critical Point Dryers 04',
      'assets.distribution': 'Asset category distribution',
      'assets.stat_total': 'Total assets',
      'assets.stat_categories': 'Categories',
      'assets.stat_fine': 'Fine classes',
      'assets.top_fine': 'Top fine-grained asset classes',
      'results.title': 'Scene Generation',
      'results.subtitle': '',
      'placeholder.video': 'Video',
      'label.task': 'Task',
      'scene.kicker': 'Procedural Generation',
      'scene.title': 'From an empty room to a complete lab',
      'scene.desc': 'The same viewpoint shows how room structure, furniture, equipment, assets, materials, safety cues, and task execution elements are added step by step.',
      'scene.process_kicker': 'Scene Build Progression',
      'scene.stage1': 'Empty room',
      'scene.stage2': 'Tables, cabinets, and benches',
      'scene.stage3': 'Large lab equipment',
      'scene.stage4': 'Desktop tools and small assets',
      'scene.stage5': 'Materials and textures',
      'scene.stage6': 'Wall fixtures, signs, and safety cues',
      'scene.stage7': 'Robot arm and task-related objects',
      'scene.preview_tag': 'Generated Scene / Asset',
      'scene.preview': 'Main scene and asset generation preview',
      'scene.token1': 'Task',
      'scene.token2': 'Objects',
      'scene.token3': 'Layout',
      'scene.token4': 'Context',
      'scene.sample_tag': 'Scene / Asset',
      'scene.sample1': 'Generated scene 01',
      'scene.sample2': 'Generated scene 02',
      'scene.sample3': 'Generated asset 01',
      'scene.sample4': 'Generated asset 02',
      'scene.sample5': 'Generated layout 01',
      'scene.sample6': 'Generated layout 02',
      'scene.caption1': 'Scene 01',
      'scene.caption2': 'Scene 02',
      'scene.caption3': 'Scene 03',
      'scene.caption4': 'Scene 04',
      'scene.caption5': 'Small Scene 02',
      'scene.caption6': 'Small Scene 03',
      'scene.small2006': 'Compact Lab Scene',
      'scene.large3010': 'Large Lab Scene',
      'scene.stage_label': 'Stage {current} / {total}',
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
      'sim.subtitle': 'Policies trained purely in simulation transfer directly to real-world deployment.',
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
      'footer.views': 'Page views',
    },
    zh: {
      'meta.description': 'LabVLA 视觉-语言-动作研究项目页面。',
      'meta.og_title': 'LabVLA',
      'meta.og_description': 'LabVLA 视觉-语言-动作研究项目页面。',
      'lang.toggle': 'English',
      'lang.aria': '切换到英文',
      'nav.features': '亮点',
      'nav.dataset': '数据集',
      'nav.workflow': '工作流',
      'nav.dual_tasks': '双臂任务',
      'nav.method': '方法',
      'nav.results': '结果',
      'nav.sim2real': '仿真到真实',
      'nav.bibtex': '引用',
      'nav.team': '团队',
      'link.paper': '论文',
      'link.homepage': '主页',
      'link.code': '代码',
      'link.model': '模型',
      'link.contact': '联系',
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
      'dataset.chart': '数据集统计图（任务 / 物体 / 机器人分布等）',
      'dataset.caption': '图：此处填写图注。',
      'single_tasks.title': '单臂任务展示',
      'single_tasks.subtitle': '展示不同单臂机器人平台上的原子操作技能。',
      'dual_tasks.title': '双臂任务展示',
      'dual_tasks.subtitle': '展示双臂协同操作任务视频。',
      'dual_tasks.card1': '双臂任务 01',
      'dual_tasks.card2': '双臂任务 02',
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
      'results.subtitle': '',
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
      'footer.views': '访问量',
    },
  };

  Object.assign(I18N.en, {
    'lang.toggle': 'Chinese',
    'dataset.subtitle': 'A suite of atomic manipulation tasks can be demonstrated independently or composed into complete lab workflows.',
    'nav.workflow': 'Workflow',
    'workflow.kicker': 'Composable Workflow',
    'workflow.page_title': 'Workflow Demonstrations',
    'workflow.page_subtitle': 'Atomic skills compose into long-horizon lab procedures across dual-arm manipulation, mobile navigation, and multi-step liquid handling.',
    'workflow.title': 'Atomic tasks can be chained into a complete lab workflow',
    'workflow.desc': 'A complete rollout composes multiple atomic skills into a longer executable lab workflow.',
    'workflow.card1': 'Franka Mobile Navigation',
    'workflow.card2': 'Franka 16-step Workflow',
    'workflow.tag.mobile': 'Mobile',
    'workflow.tag.long': '16-step',
    'workflow.step1': 'Task 01',
    'workflow.step2': 'Task 04',
    'workflow.step3': 'Task 07',
    'workflow.step4': 'Task 10',
    'single_tasks.title': 'Single-arm Task Showcase',
    'single_tasks.subtitle': 'Atomic manipulation skills demonstrated across single-arm robot platforms.',
    'nav.dual_tasks': 'Dual-arm',
    'dual_tasks.title': 'Dual-arm Task Showcase',
    'dual_tasks.subtitle': 'Bimanual manipulation demonstrations with coordinated dual-arm execution.',
    'dual_tasks.card1': 'Lift2',
    'dual_tasks.card2': 'Split Aloha',
    'tasks.robot.01': 'UR5e',
    'tasks.robot.02': 'UR16e',
    'tasks.robot.03': 'Rizon 4',
    'tasks.robot.04': 'Franka',
    'tasks.robot.05': 'Festo',
    'tasks.robot.06': 'Franka',
    'tasks.robot.07': 'UR5e',
    'tasks.robot.08': 'FR3',
    'tasks.robot.dual': 'Dual-arm',
  });

  Object.assign(I18N.zh, {
    'meta.description': 'LabVLA 视觉-语言-动作研究项目页面。',
    'meta.og_title': 'LabVLA',
    'meta.og_description': 'LabVLA 视觉-语言-动作研究项目页面。',
    'lang.toggle': 'English',
    'lang.aria': '切换到英文',
    'nav.features': '机器臂',
    'nav.dataset': '多任务',
    'nav.workflow': '工作流',
    'nav.dual_tasks': '双臂任务',
    'nav.method': '随机化',
    'nav.assets': '资产生成',
    'nav.results': '场景生成',
    'nav.sim2real': '仿真到真实',
    'nav.bibtex': '引用',
    'nav.team': '团队',
    'link.paper': '论文',
    'link.homepage': '主页',
    'link.code': '代码',
    'link.model': '模型',
    'link.contact': '联系',
    'link.dataset': '数据集',
    'link.demo': '演示视频',

    'features.title': '可编程工作流与数据引擎',
    'features.subtitle': '由多机械臂驱动的端到端数据引擎，覆盖任务、工作流、随机化、资产与场景生成。',
    'robots.single.title': '单臂机器人',
    'robots.single.count': '9 个平台',
    'robots.single.tag': '单臂',
    'robots.dual.title': '双臂机器人',
    'robots.dual.count': '3 个平台',
    'robots.dual.tag': '双臂',
    'robots.image_slot': '图片预留位',
    'robots.single.01': 'Franka',
    'robots.single.02': 'UR5e',
    'robots.single.03': 'UR16e',
    'robots.single.04': 'FR3',
    'robots.single.05': 'Festo',
    'robots.single.06': 'Rizon 4',
    'robots.single.07': '单臂机器臂 07',
    'robots.dual.01': '双臂机器臂 01',
    'robots.dual.02': '双臂机器臂 02',
    'robots.dual.03': '双臂机器臂 03',

    'dataset.title': '多任务能力',
    'dataset.subtitle': '多种原子操作任务既可独立演示，也能串联组合成完整的实验流程。',
    'tasks.atomic.tag': '原子任务',
    'tasks.media_slot': '图片 / 视频预留位',
    'tasks.atomic.01': '打开门',
    'tasks.atomic.02': '关闭门',
    'tasks.atomic.03': '放置',
    'tasks.atomic.04': '倾倒',
    'tasks.atomic.05': '抓取',
    'tasks.atomic.06': '摇晃',
    'tasks.atomic.07': '搅拌',
    'tasks.atomic.08': '加热液体',
    'tasks.robot.01': 'UR5e',
    'tasks.robot.02': 'UR16e',
    'tasks.robot.03': 'Rizon 4',
    'tasks.robot.04': 'Franka',
    'tasks.robot.05': 'Festo',
    'tasks.robot.06': 'Franka',
    'tasks.robot.07': 'UR5e',
    'tasks.robot.08': 'FR3',
    'tasks.robot.dual': '双臂',
    'single_tasks.title': '单臂任务展示',
    'single_tasks.subtitle': '展示不同单臂机器人平台上的原子操作技能。',
    'workflow.kicker': '可组合工作流',
    'workflow.page_title': '工作流演示',
    'workflow.page_subtitle': '原子操作技能可组合成长程实验流程，覆盖双臂协作、移动导航与多步骤液体操作。',
    'workflow.title': '原子任务可以串联成完整的实验工作流',
    'workflow.desc': '完整 rollout 将多个原子技能组合成更长的可执行实验工作流。',
    'workflow.card1': 'Franka 移动导航',
    'workflow.card2': 'Franka 16步工作流',
    'workflow.tag.mobile': '移动导航',
    'workflow.tag.long': '16步',
    'workflow.step1': '任务 01',
    'workflow.step2': '任务 04',
    'workflow.step3': '任务 07',
    'workflow.step4': '任务 10',
    'dual_tasks.title': '双臂任务展示',
    'dual_tasks.subtitle': '展示双臂协同操作任务视频。',
    'dual_tasks.card1': '双臂任务 01',
    'dual_tasks.card2': '双臂任务 02',

    'method.title': '随机化',
    'method.subtitle': '通过场景布局、视觉杂乱、相机视角、物体外观、光照与空间位姿的域随机化，提升策略鲁棒性。',
    'method.subhead': '同一任务状态下的多样视觉条件。',
    'method.desc': '每个任务都可以在不同环境中渲染，使策略学习稳定的任务语义，而不是过拟合单一场景。',
    'method.point1': '随机化光照、桌面材质和周围场景。',
    'method.point2': '改变物体位置和空间布局。',
    'method.point3': '加入障碍物和物体外观变化。',
    'random.kicker': '域随机化',
    'random.base': '基础场景',
    'random.advanced': '进阶版',
    'random.click': '点击切换',
    'random.scene': '场景',
    'random.clutter': '视觉杂乱',
    'random.camera': '相机',
    'random.object': '物体',
    'random.lighting': '光照',
    'random.spatial': '空间',

    'assets.title': '资产生成',
    'assets.subtitle': '生成可复用的物体、容器、工具和场景道具，用于组合多种实验室场景。',
    'assets.kicker': '可复用资产',
    'assets.heading': '资产先独立生成，再组合进场景。',
    'assets.desc': '这里预留展示生成资产的区域，例如实验工具、容器、仪器、固定装置和任务相关物体。',
    'assets.sample_tag': '资产',
    'assets.sample1': '生成资产 01',
    'assets.sample2': '生成资产 02',
    'assets.sample3': '生成资产 03',
    'assets.sample4': '生成资产 04',
    'assets.caption1': '通风柜 01',
    'assets.caption2': '通风柜 02',
    'assets.caption3': '通风柜 03',
    'assets.caption4': '通风柜 04',
    'assets.caption5': '球磨机 01',
    'assets.caption6': '球磨机 02',
    'assets.caption7': '球磨机 03',
    'assets.caption8': '球磨机 04',
    'assets.caption9': '临界点干燥仪 01',
    'assets.caption10': '临界点干燥仪 02',
    'assets.caption11': '临界点干燥仪 03',
    'assets.caption12': '临界点干燥仪 04',
    'assets.distribution': '资产类别数量分布',
    'assets.stat_total': '资产总数',
    'assets.stat_categories': '大类数量',
    'assets.stat_fine': '细分类别',
    'assets.top_fine': '高频细粒度资产类别',
    'results.title': '场景生成',
    'results.subtitle': '',
    'scene.kicker': '程序化生成',
    'scene.title': '从空房间到完整实验室',
    'scene.desc': '同一视角下逐步加入空间结构、桌柜、实验设备、小型资产、材质、安全提示，以及任务执行所需元素。',
    'scene.process_kicker': '场景进化过程',
    'scene.stage1': '空房间',
    'scene.stage2': '添加桌柜/操作台',
    'scene.stage3': '添加大型实验设备',
    'scene.stage4': '添加桌面器具和小型资产',
    'scene.stage5': '添加材质与纹理',
    'scene.stage6': '添加挂件、标识和安全提示',
    'scene.stage7': '添加机器臂和执行任务相关物品',
    'scene.preview_tag': '生成场景 / 资产',
    'scene.preview': '主场景生成预览',
    'scene.token1': '任务',
    'scene.token2': '物体',
    'scene.token3': '布局',
    'scene.token4': '上下文',
    'scene.sample_tag': '场景 / 资产',
    'scene.sample1': '生成场景 01',
    'scene.sample2': '生成场景 02',
    'scene.sample3': '生成资产 01',
    'scene.sample4': '生成资产 02',
    'scene.sample5': '生成布局 01',
    'scene.sample6': '生成布局 02',
    'scene.caption1': '场景 01',
    'scene.caption2': '场景 02',
    'scene.caption3': '场景 03',
    'scene.caption4': '场景 04',
    'scene.caption5': '小场景 02',
    'scene.caption6': '小场景 03',
    'scene.small2006': '小型实验室场景',
    'scene.large3010': '大型实验室场景',
    'scene.stage_label': '阶段 {current} / {total}',

    'sim.title': '仿真到真实迁移',
    'sim.subtitle': '展示从仿真训练到真实世界部署的迁移效果。',
    'label.sim': '仿真',
    'label.real': '真实',
    'label.task_name': '任务名称',
    'sim.video1': '仿真训练视频 1',
    'sim.video2': '仿真训练视频 2',
    'sim.real1': '真实执行视频 1',
    'sim.real2': '真实执行视频 2',

    'bib.title': '引用',
    'bib.subtitle': '如果这项工作对你有帮助，欢迎引用。',
    'bib.copy': '复制',
    'bib.copied': '已复制',
    'bib.copy_aria': '复制 BibTeX',

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
    'team.advisor1': '指导老师 1',
    'team.advisor2': '指导老师 2',
    'team.advisor3': '指导老师 3',

    'ack.title': '致谢',
    'ack.text': '此处填写资助、合作机构、数据来源和模板致谢。',
    'footer.desc': '此处填写项目的一句话简介。',
    'footer.links': '链接',
    'footer.contact': '联系方式',
    'footer.email': '联系邮箱',
    'footer.social': 'Twitter / 微博',
    'footer.rights': 'LabVLA. 保留所有权利。',
    'footer.note': '发布前请替换占位内容。',
  });

  Object.assign(I18N.en, {
    'scene.index_label': 'Scene {current} / {total}',
    'scene.small620116': 'Lab Scene 03',
    'scene.large630101': 'Lab Scene 04',
    'scene.large630109': 'Lab Scene 05',
    'scene.large630113': 'Lab Scene 06',
  });

  Object.assign(I18N.zh, {
    'scene.index_label': '\u573a\u666f {current} / {total}',
    'scene.small620116': '\u5b9e\u9a8c\u5ba4\u573a\u666f 03',
    'scene.large630101': '\u5b9e\u9a8c\u5ba4\u573a\u666f 04',
    'scene.large630109': '\u5b9e\u9a8c\u5ba4\u573a\u666f 05',
    'scene.large630113': '\u5b9e\u9a8c\u5ba4\u573a\u666f 06',
  });

  Object.assign(I18N.en, {
    'dual_tasks.card1': 'Lift2',
    'dual_tasks.card2': 'Split Aloha',
    'dual_tasks.robot1': 'Lift2',
    'dual_tasks.robot2': 'Split Aloha',
  });

  Object.assign(I18N.zh, {
    'dual_tasks.card1': 'Lift2',
    'dual_tasks.card2': 'Split Aloha',
    'dual_tasks.robot1': 'Lift2',
    'dual_tasks.robot2': 'Split Aloha',
  });

  Object.assign(I18N.en, {
    'de.eyebrow': 'LabVLA · Data Engine',
    'de.title': 'A Programmable Workflow and Data Engine',
    'de.sub': 'A multi-robot-arm data engine: Robots · Tasks · Workflow · Randomization · Assets · Scene Generation',
    'de.collapse': 'Collapse',
    'de.expand': 'View Data Engine Details',
    'de.navCollapsed': 'Data Engine',
    'de.nav.home': 'Data Engine',
    'de.nav.skills': 'Atomic Skills',
    'de.nav.workflow': 'Workflows',
    'de.nav.random': 'Randomization',
    'de.nav.assets': 'Assets',
    'de.nav.scenes': 'Scenes',
  });

  Object.assign(I18N.zh, {
    'de.eyebrow': 'LabVLA · 数据引擎',
    'de.title': '可编程工作流与数据引擎',
    'de.sub': '面向多机械臂的数据引擎：机器人 · 任务 · 工作流 · 随机化 · 资产 · 场景生成',
    'de.collapse': '收起',
    'de.expand': '查看数据引擎详情',
    'de.navCollapsed': '数据引擎',
    'de.nav.home': '数据引擎',
    'de.nav.skills': '原子技能',
    'de.nav.workflow': '工作流',
    'de.nav.random': '域随机化',
    'de.nav.assets': '资产生成',
    'de.nav.scenes': '场景生成',
  });

  /* ---------- P1: The Gap + Approach (EN) ---------- */
  Object.assign(I18N.en, {
    'hero.tagline': 'Grounding Vision–Language–Action models in scientific laboratories',
    'gap.kicker': 'The Gap',
    'gap.title1': 'AI can reason about science',
    'gap.title2': "It still can't run the experiment",
    'gap.lead': 'AI can read literature, generate hypotheses, and plan protocols — yet executing them at the bench still falls to a human. The gap between digital reasoning and real experimental work is not one of intent, but of embodiment.',
    'gap.c1.t': 'Bench-level precision',
    'gap.c1.d': 'Pipetting, cap screwing, liquid transfer, and button presses demand fine spatial precision and reliable contact control.',
    'gap.c2.t': 'Physical state changes',
    'gap.c2.d': 'Tasks hinge on liquid flow, heating, mixing, color transitions, and precise container placement.',
    'gap.c3.t': 'Cross-embodiment',
    'gap.c3.d': 'The same protocol must run on different arms, cameras, end-effectors, and action dimensions.',
    'gap.c4.t': 'Data is the bottleneck',
    'gap.c4.d': 'Real-lab collection needs instruments, calibration, supervision, and safety — pushing cost far above ordinary robot data.',
    'approach.kicker': 'Highlights',
    'approach.title1': 'What sets LabVLA apart —',
    'approach.title2': 'data and policy',
    'approach.lead': 'Building on this corpus together with broad real-robot pre-training data, we present LabVLA, a VLA pipeline for connecting written laboratory protocols to embodied robot execution in simulated scientific workspaces. LabVLA pairs protocol-conditioned data synthesis with FAST action-token pre-training and flow-matching post-training under a shared cross-embodiment schema.',
    'approach.p1.k': '01 · Data',
    'approach.p1.t': 'RoboGenesis: A Programmable Workflow and Data Engine',
    'approach.p1.d': 'We introduce RoboGenesis, a simulation-based workflow and data engine that links environment construction, configured workflow generation, domain randomization, and success-filtered export to produce laboratory demonstrations that existing robot corpora rarely cover. We use this engine to synthesize LabEmbodied-Data, a corpus of multi-camera observations, language instructions, robot states, action trajectories, and structured annotations under a shared cross-embodiment schema.',
    'approach.p2.k': '02 · Training',
    'approach.p2.t': 'LabVLA Training Recipe',
    'approach.p2.d': 'LabVLA adapts a Qwen3-VL backbone to map visual observations, robot state, and language instructions into continuous action chunks through a DiT action expert. The model is trained in two stages: FAST action tokens first align the visual-language prefix with action semantics during VLM pre-training, and flow matching then predicts continuous robot actions during post-training. A knowledge-insulation design reduces interference between language-grounded VLM representations and the continuous-action expert during post-training.',
    'approach.kpi1': 'Validated lab scenes',
    'approach.kpi2': 'Robot platforms',
    'approach.kpi3': 'Annotation streams',
    'approach.kpi4': 'Best avg success (ID)',
    'recipe.kicker': 'Training Recipe',
    'recipe.title1': 'Two stages,',
    'recipe.title2': 'one cross-embodiment policy',
    'recipe.lead': 'LabVLA adapts a Qwen3-VL backbone with FAST action-token pre-training, then a flow-matching DiT action expert — coupled by a stop-gradient that keeps language grounding intact.',
    'recipe.tab1': '① VLM Pre-training',
    'recipe.tab2': '② Flow-matching Post-training',
    'recipe.n1t': 'Corpora',
    'recipe.n4t': 'Robot actions',
    'recipe.sg': 'stop-grad',
    'recipe.p1.t': 'Make the prefix action-aware',
    'recipe.p1.d': 'The Qwen3-VL-4B backbone trains on grounded corpora (Robointer-VQA, AgiBot World Beta, OXE-AugE, Droid) to produce VQA answers, language subtasks, and discrete FAST action tokens — so the visual-language prefix becomes action-aware before any continuous action head is attached.',
    'recipe.p2.t': 'Specialize the action expert',
    'recipe.p2.d': 'The pretrained VLM is paired with a DiT action expert and supervised with flow matching on OXE-AugE together with LabEmbodied-Data. A stop-gradient (knowledge insulation) blocks flow-matching gradients from the VLM prefix, so language grounding stays intact while the action expert specializes.',
    'recipe.k1.k': '01',
    'recipe.k1.t': 'VLM Pretraining',
    'recipe.k1.d': 'We first tokenize continuous actions with FAST and train the VLM under next token supervision, so the prefix learns to predict action tokens before the DiT is attached. In this stage we do not instantiate the DiT.',
    'recipe.k2.k': '02',
    'recipe.k2.t': 'Flow Matching Posttraining',
    'recipe.k2.d': 'The second stage loads the VLM pretrained checkpoint, attaches the DiT action expert, and trains it with a flow matching objective that maps Gaussian noise to a clean action chunk through a deterministic vector field. At sampling time the deterministic vector field reaches a usable trajectory in only N=10 Euler steps, well below the hundreds needed by diffusion policies and fast enough for closed loop laboratory control.',
    'recipe.k3.k': '03',
    'recipe.k3.t': 'Knowledge Insulation',
    'recipe.k3.d': 'We insulate the VLM from the flow loss while keeping the FAST and annotation token losses active, so the prefix can still learn from cross-entropy supervision without receiving velocity space gradients from the action expert. Knowledge insulation is a training time mechanism that blocks flow matching gradients from reaching the VLM prefix while FAST and annotation losses remain active.',
    'recipe.stage1': '① Pre-training',
    'recipe.stage2': '② Post-training',
    'recipe.figcap.b': 'LabVLA training recipe.',
    'recipe.figcap.t': 'Left — FAST action-token pre-training aligns the Qwen3-VL prefix with action semantics. Right — flow-matching post-training specializes a DiT action expert, with a stop-gradient (knowledge insulation) shielding the VLM.',
    'approach.figcap.b': 'LabVLA at a glance.',
    'nav.team': 'Institutions',
    'team.kicker': 'Affiliations',
    'team.title': 'Institutions',
    'team.subtitle': 'The institutions behind LabVLA.',
    'team.note': 'This work is jointly conducted by the following institutions',
    'team.inst1': 'Zhejiang University',
    'team.inst2': 'Shanghai AI Laboratory',
    'team.inst3': 'Harbin Institute of Technology',
    'res.kicker': 'Results',
    'res.title1': 'State-of-the-art on',
    'res.title2': 'the LabUtopia benchmark',
    'res.lead': 'Six laboratory operations, evaluated in-distribution (ID) and out-of-distribution (OOD) against sub-1B, 3B, and 4B VLA baselines.',
    'res.kpi1': 'Average success · ID',
    'res.kpi2': 'Average success · OOD',
    'res.kpi3': 'over π0 (ID)',
    'res.kpi4': 'over π0 (OOD)',
    'res.bars': 'Average success rate · ID',
    'res.th.method': 'Method',
    'res.th.size': 'Size',
    'res.t.pick': 'Pick Up',
    'res.t.press': 'Press Button',
    'res.t.open': 'Open Door',
    'res.t.pour': 'Pour Liquid',
    'res.t.heat': 'Heat Beaker',
    'res.t.transport': 'Transport Beaker',
    'res.t.avg': 'Avg.',
    'res.grp.id': 'In-Distribution (ID)',
    'res.grp.ood': 'Out-of-Distribution (OOD)',
    'res.note': 'Green marks the column-best. Pour Liquid stays the hardest category — no method exceeds 50%.',
    'res.figcap.b': 'The six LabUtopia operations.',
    'res.figcap.t': 'Pick Up, Press Button, Open Door, Pour Liquid, Heat Beaker, and Transport Beaker — each evaluated under ID and OOD settings.',
    'fig.zoom': 'Enlarge',
    'approach.figttl': 'System overview',
    'recipe.figttl': 'Training recipe',
    'res.figttl': 'LabUtopia tasks',
    'approach.figcap.t': 'Web-scale & real-robot corpora warm-start the VLM (left); RoboGenesis synthesizes knowledgeable LabEmbodied-Data (right); the policy couples a pre-trained VLM with a DiT action expert (center) and is evaluated across four task families (bottom).',
    'approach.read.k': 'How to read it',
    'approach.read.t': 'One system, four moving parts.',
    'approach.read.l1': 'Warm-start · left',
    'approach.read.d1': 'Web-scale and real-robot corpora pre-train the vision–language backbone.',
    'approach.read.l2': 'Data engine · right',
    'approach.read.d2': 'RoboGenesis synthesizes knowledgeable, success-filtered LabEmbodied-Data.',
    'approach.read.l3': 'Policy · center',
    'approach.read.d3': 'A pre-trained VLM is coupled with a DiT action expert through a stop-gradient.',
    'approach.read.l4': 'Evaluation · bottom',
    'approach.read.d4': 'The policy is tested across four laboratory task families under ID and OOD.',
    'cap.kicker': 'Levels of embodied laboratory competence',
    'cap.title1': 'From apprentice',
    'cap.title2': 'to scientist',
    'cap.lead': 'Rather than a single aggregate score, laboratory manipulation is better viewed through four levels of competence modeled on real laboratory roles. We position LabVLA at Level 2 (Technician).',
    'cap.axis': 'Competence',
    'cap.tag.ours': 'LabVLA at Level 2 (Technician)',
    'cap.tag.next': 'RoboGenesis infrastructure that begins to support Level 3',
    'cap.l1.role': 'Apprentice',
    'cap.l1.d': 'Single step interactions with laboratory objects',
    'cap.l1.desc': 'Level 1 (Apprentice) covers single step interactions with laboratory objects: grasping labware, pressing a button, opening a door, or placing a container.',
    'cap.l2.role': 'Technician',
    'cap.l2.d': 'Written multistep protocol · physical state changes',
    'cap.l2.desc': 'Level 2 (Technician) requires following a written multistep protocol through physical state changes such as pouring, heating, stirring, shaking, or transporting a vessel, where a failed earlier step cascades through the rest of the procedure.',
    'cap.l3.role': 'Specialist',
    'cap.l3.d': 'Precision instruments · measurement logging · safety constraints',
    'cap.l3.desc': 'Level 3 (Specialist) adds operation of precision instruments (pipettes, centrifuges, thermal cyclers, microscopes) in longer workflows with measurement logging and safety constraints.',
    'cap.l4.role': 'Scientist',
    'cap.l4.note': 'However, the policy does not yet demonstrate the instrument competence, measurement awareness, or scientific judgment that Level 3 and Level 4 require.',
    'cap.l4.d': 'Modifies the procedure in response to observations or measurements',
    'cap.l4.desc': 'Level 4 (Scientist) modifies the procedure in response to observations or measurements: adjusting concentrations, branching to alternative protocols, or deciding when an experimental objective has been met.',
  });

  /* ---------- P1: The Gap + Approach (ZH) ---------- */
  Object.assign(I18N.zh, {
    'hero.tagline': '面向科学实验室场景的视觉-语言-动作（VLA）模型应用探索',
    'gap.kicker': '鸿沟',
    'gap.title1': 'AI 已能推理科学，',
    'gap.title2': '却仍不能亲手做实验',
    'gap.lead': 'AI 能读文献、提假设、写实验方案，但把方案在实验台上执行，依旧得靠人。数字推理与真实实验之间的鸿沟，不在意图，而在“具身”。',
    'gap.c1.t': '台面级精度',
    'gap.c1.d': '移液、拧盖、倒液、按钮——都需要精细的空间定位与可靠的接触控制。',
    'gap.c2.t': '物理状态变化',
    'gap.c2.d': '任务依赖液体流动、加热、混合、变色与精确的容器摆放。',
    'gap.c3.t': '跨本体',
    'gap.c3.d': '同一套协议要在不同机械臂、相机、末端执行器与动作维度上运行。',
    'gap.c4.t': '数据是瓶颈',
    'gap.c4.d': '真实实验室采集需要仪器、标定、人工监督与安全流程，成本远高于普通机器人数据。',
    'approach.kicker': '本工作特色',
    'approach.title1': 'LabVLA 的与众不同——',
    'approach.title2': '数据与策略',
    'approach.lead': '在 LabEmbodied-Data 与大规模真机预训练数据之上，我们提出 LabVLA——一个将书面实验协议连接到仿真科学工作空间中实体机器人执行的 VLA 流程。LabVLA 将受协议约束的数据合成与 FAST 动作 token 预训练、flow-matching 后训练相结合，并在统一的跨本体模式下进行。',
    'approach.p1.k': '01 · 数据',
    'approach.p1.t': 'RoboGenesis：可编程工作流与数据引擎',
    'approach.p1.d': '我们提出 RoboGenesis，一个基于仿真的工作流与数据引擎，将环境构建、可配置工作流生成、域随机化与按成功率过滤的导出串联起来，以产出现有机器人语料库鲜少覆盖的实验室示范。我们使用该引擎合成 LabEmbodied-Data——一个在多相机观测、语言指令、机器人状态、动作轨迹及共享跨本体 schema 下的结构化标注语料库。',
    'approach.p2.k': '02 · 训练',
    'approach.p2.t': 'LabVLA 训练方案',
    'approach.p2.d': 'LabVLA 基于 Qwen3-VL 骨干，通过 DiT 动作专家将视觉观测、机器人状态与语言指令映射为连续动作块。模型分两阶段训练：FAST 动作 token 在 VLM 预训练阶段先对齐视觉-语言前缀与动作语义，flow matching 在后训练阶段预测连续机器人动作。知识隔离设计降低了后训练过程中语言 grounding 的 VLM 表征与连续动作专家之间的干扰。',
    'approach.kpi1': '已验证实验场景',
    'approach.kpi2': '机器人平台',
    'approach.kpi3': '标注流',
    'approach.kpi4': '最佳平均成功率 (ID)',
    'recipe.kicker': '训练方案',
    'recipe.title1': '两阶段，',
    'recipe.title2': '一个跨本体策略',
    'recipe.lead': 'LabVLA 以 Qwen3-VL 为主干，先做 FAST 动作-token 预训练，再接 flow-matching 的 DiT 动作专家——两者之间用 stop-gradient 耦合，保住语言对齐能力。',
    'recipe.tab1': '① VLM 预训练',
    'recipe.tab2': '② Flow-matching 后训练',
    'recipe.n1t': '语料',
    'recipe.n4t': '机器人动作',
    'recipe.sg': '梯度截断',
    'recipe.p1.t': '让前缀“懂动作”',
    'recipe.p1.d': 'Qwen3-VL-4B 主干在 grounded 语料（Robointer-VQA、AgiBot World Beta、OXE-AugE、Droid）上训练，产出 VQA 答案、语言子任务与离散 FAST 动作 token——让视觉-语言前缀在接入任何连续动作头之前，就先具备动作语义。',
    'recipe.p2.t': '让动作专家“专精”',
    'recipe.p2.d': '预训练好的 VLM 接上 DiT 动作专家，在 OXE-AugE 与 LabEmbodied-Data 上用 flow matching 监督。stop-gradient（知识隔离）阻断 flow-matching 梯度回流到 VLM 前缀，从而在动作专家专精的同时，保持语言对齐不漂移。',
    'recipe.k1.k': '01',
    'recipe.k1.t': 'VLM 预训练',
    'recipe.k1.d': '我们先用 FAST 将连续动作 tokenize，并在 next-token 监督下训练 VLM，使前缀在接入 DiT 之前就能预测动作 token。此阶段不实例化 DiT。',
    'recipe.k2.k': '02',
    'recipe.k2.t': 'Flow Matching 后训练',
    'recipe.k2.d': '第二阶段加载预训练 VLM，接入 DiT 动作专家，并以 flow matching 目标训练：通过确定性向量场将高斯噪声映射为干净动作块。采样时该向量场仅需 N=10 步 Euler 积分即可得到可用轨迹，远少于扩散策略所需的数百步，足以支撑实验室闭环控制。',
    'recipe.k3.k': '03',
    'recipe.k3.t': 'Knowledge Insulation',
    'recipe.k3.d': '我们将 VLM 与 flow loss 隔离，同时保留 FAST 与 annotation token loss，使前缀仍可通过交叉熵监督学习，而不接收动作专家传来的速度空间梯度。知识隔离是一种训练期机制：阻断 flow matching 梯度回传到 VLM 前缀，同时 FAST 与 annotation loss 继续生效。',
    'recipe.stage1': '① 预训练',
    'recipe.stage2': '② 后训练',
    'recipe.figcap.b': 'LabVLA 训练方案。',
    'recipe.figcap.t': '左 — FAST 动作-token 预训练让 Qwen3-VL 前缀对齐动作语义；右 — flow-matching 后训练专精 DiT 动作专家，并用 stop-gradient（知识隔离）保护 VLM。',
    'approach.figcap.b': 'LabVLA 总览。',
    'nav.team': '单位',
    'team.kicker': '合作单位',
    'team.title': '研究单位',
    'team.subtitle': 'LabVLA 背后的研究单位。',
    'team.note': '本研究由以下单位联合完成',
    'team.inst1': '浙江大学',
    'team.inst2': '上海人工智能实验室',
    'team.inst3': '哈尔滨工业大学',
    'res.kicker': '实验结果',
    'res.title1': '在 LabUtopia 基准上',
    'res.title2': '全面领先',
    'res.lead': '六类实验室操作，在分布内 (ID) 与分布外 (OOD) 两种设定下，对比 sub-1B / 3B / 4B 的 VLA 基线。',
    'res.kpi1': '平均成功率 · ID',
    'res.kpi2': '平均成功率 · OOD',
    'res.kpi3': '超过 π0 (ID)',
    'res.kpi4': '超过 π0 (OOD)',
    'res.bars': '平均成功率 · ID',
    'res.th.method': '方法',
    'res.th.size': '规模',
    'res.t.pick': '拾取',
    'res.t.press': '按按钮',
    'res.t.open': '开门',
    'res.t.pour': '倒液',
    'res.t.heat': '加热烧杯',
    'res.t.transport': '搬运烧杯',
    'res.t.avg': '平均',
    'res.grp.id': '分布内 (ID)',
    'res.grp.ood': '分布外 (OOD)',
    'res.note': '绿色为该列最佳。倒液（Pour Liquid）仍是最难的一类——没有方法超过 50%。',
    'res.figcap.b': 'LabUtopia 的六类操作。',
    'res.figcap.t': '拾取、按按钮、开门、倒液、加热烧杯、搬运烧杯——每类都在 ID 与 OOD 下评测。',
    'fig.zoom': '查看大图',
    'approach.figttl': '系统总览',
    'recipe.figttl': '训练方案',
    'res.figttl': 'LabUtopia 任务',
    'approach.figcap.t': '左：网络规模与真机语料 warm-start VLM；右：RoboGenesis 合成带标注的 LabEmbodied-Data；中：策略将预训练 VLM 与 DiT 动作专家耦合；底部：在四类任务上评测。',
    'approach.read.k': '如何看这张图',
    'approach.read.t': '一个系统，四个关键部分。',
    'approach.read.l1': '热启动 · 左',
    'approach.read.d1': '网络规模与真机语料先对视觉-语言主干做预训练。',
    'approach.read.l2': '数据引擎 · 右',
    'approach.read.d2': 'RoboGenesis 合成有知识、按成功率过滤的 LabEmbodied-Data。',
    'approach.read.l3': '策略 · 中',
    'approach.read.d3': '预训练 VLM 通过 stop-gradient 与 DiT 动作专家耦合。',
    'approach.read.l4': '评测 · 底部',
    'approach.read.d4': '策略在四类实验室任务上评测，覆盖 ID 与 OOD。',
    'cap.kicker': '具身实验室能力层级',
    'cap.title1': '从学徒',
    'cap.title2': '到科学家',
    'cap.lead': '与其用单一总分衡量，实验室操作更适合用四类、对应真实实验室角色的能力层级来理解。我们将 LabVLA 定位在 Level 2（Technician）。',
    'cap.axis': '能力层级',
    'cap.tag.ours': 'LabVLA 定位于 Level 2（Technician）',
    'cap.tag.next': 'RoboGenesis 基础设施开始支撑 Level 3',
    'cap.l1.role': 'Apprentice',
    'cap.l1.d': '与实验物品的单步交互',
    'cap.l1.desc': 'Level 1（Apprentice）涵盖与实验物品的单步交互：抓取实验器皿、按压按钮、开门或放置容器。',
    'cap.l2.role': 'Technician',
    'cap.l2.d': '书面多步协议 · 物理状态变化',
    'cap.l2.desc': 'Level 2（Technician）要求遵循书面多步协议，并经历倒液、加热、搅拌、摇晃或搬运容器等物理状态变化；前一步失败会连锁影响后续步骤。',
    'cap.l3.role': 'Specialist',
    'cap.l3.d': '精密仪器 · 测量记录 · 安全约束',
    'cap.l3.desc': 'Level 3（Specialist）在此基础上操作精密仪器（移液器、离心机、热循环仪、显微镜），执行更长工作流，并进行测量记录与安全约束。',
    'cap.l4.role': 'Scientist',
    'cap.l4.note': '然而，当前策略尚未展现 Level 3 与 Level 4 所需的仪器操作能力、测量意识或科学判断。',
    'cap.l4.d': '根据观测或测量修改流程',
    'cap.l4.desc': 'Level 4（Scientist）会根据观测或测量修改流程：调整浓度、切换到替代协议，或判断实验目标是否达成。',
  });

  // ===== Stage 3 · Analysis (real-world sim-to-real + data transferability) =====
  Object.assign(I18N.en, {
    'nav.sim2real': 'Analysis',
    'nav.real': 'Sim2Real',
    'sim.kicker': 'Analysis',
    'sim.title1': 'The data transfers,',
    'sim.title2': 'lifting external policies too',
    'sim.lead': 'A study beyond LabUtopia: an external X-VLA baseline also benefits from fine-tuning on LabEmbodied-Data — the supervision is not tied to the LabVLA architecture.',
    'sim.a.k': '01 · Real-world transfer',
    'sim.a.title': 'Simulation-trained, deployed on a real Franka',
    'sim.a.figttl': 'Real-world setup',
    'sim.a.figcap.b': 'Franka lab cell.',
    'sim.a.figcap.t': 'Beakers, flasks, a magnetic stirrer and a heating plate, shared across the four real-robot tasks.',
    'sim.a.desc': 'We deploy LabVLA on a physical Franka alongside DreamZero and π0.5. Four tasks each compose 2–4 atomic skills, with 30–50 demonstrations and the target randomized within a 5×5 cm region. Every policy is scored over 50 rollouts under four conditions — crossing target position (in-domain vs. out-of-domain) with workspace clutter (clean vs. cluttered).',
    'sim.a.kpi1': 'Avg success · clean, in-domain',
    'sim.a.kpi2': 'Avg success · clean, OOD (best)',
    'sim.a.kpi3': 'Real-robot tasks',
    'sim.a.kpi4': 'Rollouts per condition',
    'sim.th.set': 'Task · Setting',
    'sim.col.ours': 'LabVLA (Ours)',
    'sim.col.pi': '\u03c00.5',
    'sim.task.shake': 'Shake Liquid · pick → shake → place',
    'sim.task.pour': 'Pour Liquid · pick → pour → place',
    'sim.task.stir': 'Magnetic Stir · pick → place → press',
    'sim.task.stopper': 'Funnel Plug/Unplug · pick → place → pick → place',
    'sim.task.avg': 'Average',
    'sim.set.id_clean': 'In-domain · Clean',
    'sim.set.id_clut': 'In-domain · Cluttered',
    'sim.set.ood_clean': 'Out-of-domain · Clean',
    'sim.set.ood_clut': 'Out-of-domain · Cluttered',
    'sim.note.real': 'Success rate (%) over 50 rollouts per setting; bold = per-row best. LabVLA leads the clean out-of-domain average.',
    'sim.a.watch': 'Watch Videos',
    'sim.a.modal.title': 'Real-robot rollouts',
    'sim.a.modal.lead': 'Uncut Franka execution clips from the real laboratory — one per evaluation task.',
    'sim.a.modal.hint': 'Click any video to play.',
    'sim.a.modal.play': 'Click to play',
    'sim.a.demo.kicker': 'Field Rollouts',
    'sim.a.demo.lead': 'Uncut Franka execution clips from the real laboratory — one per evaluation task.',
    'sim.a.demo.stat.shake': '92% · ID Clean',
    'sim.a.demo.stat.pour': '86% · ID Clean',
    'sim.a.demo.stat.stir': '88% · ID Clean',
    'sim.a.demo.stat.stopper': '80% · ID Clean',
    'sim.b.k': '02 · Data transferability',
    'sim.b.title': 'LabEmbodied-Data lifts an external policy',
    'sim.b.desc': 'Fine-tuning X-VLA — a sub-1B baseline from the LabUtopia comparison — on LabEmbodied-Data raises its five-task average by +15.0% (ID) and +19.3% (OOD). The biggest gains land on instrument-specific contact tasks the original data never saw.',
    'sim.b.kpi1': '5-task avg gain · ID',
    'sim.b.kpi2': '5-task avg gain · OOD',
    'sim.b.kpi3': 'Heat Beaker · ID',
    'sim.b.kpi4': 'Pour Liquid · OOD',
    'sim.b.aug': 'X-VLA + LabEmbodied',
    'sim.note.transfer': 'Five non-saturated LabUtopia tasks (Press Button excluded as near-saturated for all baselines). Δ is the change in five-task average from adding LabEmbodied-Data.',
  });
  Object.assign(I18N.zh, {
    'nav.sim2real': '分析',
    'nav.real': 'Sim2Real',
    'sim.kicker': '深入分析',
    'sim.title1': '数据可迁移，',
    'sim.title2': '亦能赋能外部策略',
    'sim.lead': 'LabUtopia 之外的一项研究：把外部基线 X-VLA 在 LabEmbodied-Data 上微调，同样能稳定受益——这份监督信号并不依赖 LabVLA 架构。',
    'sim.a.k': '01 · 真机迁移',
    'sim.a.title': '仿真训练，直接部署到真实 Franka',
    'sim.a.figttl': '真机实验布置',
    'sim.a.figcap.b': 'Franka 实验台。',
    'sim.a.figcap.t': '烧杯、烧瓶、磁力搅拌器与加热板，四个真机任务共用同一套布置。',
    'sim.a.desc': '我们在真实 Franka 平台上部署 LabVLA，并与 DreamZero、π0.5 对比。四个任务各由 2–4 个原子技能组成，每个任务采集 30–50 条示范，目标物在 5×5 cm 区域内随机摆放。每个策略在四种条件下各跑 50 次：目标位置（分布内 / 分布外）× 工作区杂乱程度（干净 / 杂乱）。',
    'sim.a.kpi1': '平均成功率 · 干净·分布内',
    'sim.a.kpi2': '平均成功率 · 干净·分布外（最佳）',
    'sim.a.kpi3': '真机任务数',
    'sim.a.kpi4': '每种条件 rollout 次数',
    'sim.th.set': '任务 · 设定',
    'sim.col.ours': 'LabVLA (Ours)',
    'sim.col.pi': '\u03c00.5',
    'sim.task.shake': '摇晃液体 · 抓取 → 摇晃 → 放置',
    'sim.task.pour': '倾倒液体 · 抓取 → 倾倒 → 放置',
    'sim.task.stir': '磁力搅拌 · 抓取 → 放置 → 按压',
    'sim.task.stopper': '漏斗插拔 · 抓取 → 放置 → 抓取 → 放置',
    'sim.task.avg': '综合平均',
    'sim.set.id_clean': '分布内 · 干净',
    'sim.set.id_clut': '分布内 · 杂乱',
    'sim.set.ood_clean': '分布外 · 干净',
    'sim.set.ood_clut': '分布外 · 杂乱',
    'sim.note.real': '每种设定下 50 次 rollout 的成功率（%），加粗为该行最佳；LabVLA 在「干净·分布外」平均上领先。',
    'sim.a.watch': '观看视频',
    'sim.a.modal.title': '真机执行视频',
    'sim.a.modal.lead': '真实实验室 Franka 未剪辑执行片段——每个评测任务一条。',
    'sim.a.modal.hint': '点击任意视频即可播放。',
    'sim.a.modal.play': '点击播放',
    'sim.a.demo.kicker': '真机执行',
    'sim.a.demo.lead': '真实实验室 Franka 未剪辑执行片段——每个评测任务一条。',
    'sim.a.demo.stat.shake': '92% · 分布内·干净',
    'sim.a.demo.stat.pour': '86% · 分布内·干净',
    'sim.a.demo.stat.stir': '88% · 分布内·干净',
    'sim.a.demo.stat.stopper': '80% · 分布内·干净',
    'sim.b.k': '02 · 数据可迁移性',
    'sim.b.title': 'LabEmbodied-Data 能提升外部策略',
    'sim.b.desc': '把 X-VLA（LabUtopia 对比中的 sub-1B 基线）在 LabEmbodied-Data 上微调后，五任务平均提升 +15.0%（ID）/ +19.3%（OOD）。最大增益集中在原始数据未覆盖、需要特定仪器接触模式的任务上。',
    'sim.b.kpi1': '五任务平均增益 · ID',
    'sim.b.kpi2': '五任务平均增益 · OOD',
    'sim.b.kpi3': '加热烧杯 · ID',
    'sim.b.kpi4': '倒液 · OOD',
    'sim.b.aug': 'X-VLA + LabEmbodied',
    'sim.note.transfer': '五个未饱和的 LabUtopia 任务（Press Button 因各基线接近饱和而排除）。Δ 为加入 LabEmbodied-Data 后五任务平均的变化。',
  });

  // ===== Unify data-engine section heads (kicker + gradient title) =====
  Object.assign(I18N.en, {
    'features.kicker':'RoboGenesis','features.t1':'A Programmable Workflow and ','features.t2':'Data Engine',
    'dataset.kicker':'Atomic Skills','dataset.t1':'Multi-task',    'dataset.t2':'capabilities',
    'workflow.kicker':'Long-horizon','workflow.t1':'Workflows &','workflow.t2':'dual-arm showcase',
    'method.kicker':'Robustness','method.t1':'Domain','method.t2':'randomization',
    'assets.kicker':'Reusable Assets','assets.t1':'Asset','assets.t2':'generation',
    'results.kicker':'Composable Scenes','results.t1':'Scene','results.t2':'generation',
  });
  Object.assign(I18N.zh, {
    'features.kicker':'RoboGenesis','features.t1':'可编程工作流与 ','features.t2':'数据引擎',
    'dataset.kicker':'原子技能','dataset.t1':'多任务',    'dataset.t2':'能力',
    'workflow.kicker':'长程流程','workflow.t1':'工作流 &','workflow.t2':'双臂展示',
    'method.kicker':'鲁棒性','method.t1':'域','method.t2':'随机化',
    'assets.kicker':'可复用资产','assets.t1':'资产','assets.t2':'生成',
    'results.kicker':'可组合场景','results.t1':'场景','results.t2':'生成',
  });

  // ===== Nav restructure: top-level sections (Highlights/Data Engine/Method/Results/Capability/Analysis/Outlook/Citation/Team) =====
  Object.assign(I18N.en, {
    'nav.approach':'Highlights','nav.features':'Data Engine','nav.recipe':'Method',
    'nav.benchmark':'Results','nav.capability':'Capability',
    'de.navCollapsed':'Data Engine',
  });
  Object.assign(I18N.zh, {
    'nav.approach':'亮点','nav.features':'数据引擎','nav.recipe':'方法',
    'nav.benchmark':'结果','nav.capability':'能力',
    'de.navCollapsed':'数据引擎',
  });

  // ===== Results table -> authentic paper table image =====
  Object.assign(I18N.en, {
    'res.tbl.ttl': 'LabUtopia benchmark',
    'res.note': 'Bold marks the column-best; LabVLA rows are highlighted. Pour Liquid stays the hardest category — no method exceeds 50%.',
    'tbl.method': 'Method',
    'tbl.size': 'Size',
    'tbl.avg': 'Avg.',
    'tbl.task': 'Task',
    'tbl.id': 'In-Distribution',
    'tbl.ood': 'Out-of-Distribution',
    'tbl.view.id': 'ID',
    'tbl.view.ood': 'OOD',
    'tbl.switch.aria': 'Select benchmark split',
    'sim.th.setcol': 'Setting',
  });
  Object.assign(I18N.zh, {
    'res.tbl.ttl': 'LabUtopia 基准',
    'res.note': '加粗为该列最佳，LabVLA 行高亮。倒液（Pour Liquid）仍是最难的一类——没有方法超过 50%。',
    'tbl.method': '方法',
    'tbl.size': '规模',
    'tbl.avg': '平均',
    'tbl.task': '任务',
    'tbl.id': '分布内 (ID)',
    'tbl.ood': '分布外 (OOD)',
    'sim.th.setcol': '设定',
  });

  Object.assign(I18N.en, {
    'sim.a.tblttl': 'Real-robot evaluation · Franka',
    'sim.b.tblttl': 'LabEmbodied-Data transferability',
    'real.switch.aria': 'Select real-robot task',
    'sim.task.btn.shake': 'Shake Liquid',
    'sim.task.btn.pour': 'Pour Liquid',
    'sim.task.btn.stir': 'Magnetic Stir',
    'sim.task.btn.stopper': 'Funnel Plug/Unplug',
    'sim.task.name.stopper': 'Funnel Plug/\nUnplug',
  });
  Object.assign(I18N.zh, {
    'sim.a.tblttl': 'Franka 真机实验',
    'sim.b.tblttl': 'LabEmbodied-Data 可迁移性',
    'real.switch.aria': '选择真机实验任务',
    'sim.task.btn.shake': '摇晃液体',
    'sim.task.btn.pour': '倾倒液体',
    'sim.task.btn.stir': '磁力搅拌',
    'sim.task.btn.stopper': '漏斗插拔',
    'sim.task.name.stopper': '漏斗插拔',
  });

  Object.assign(I18N.en, { 'res.tbl.ttl': 'LabUtopia benchmark' });
  Object.assign(I18N.zh, { 'res.tbl.ttl': 'LabUtopia 基准' });
  // Analysis order: transferability (§5.1) before sim2real; standalone real-robot headings; engine comparison (Table 1)
  Object.assign(I18N.en, {
    'sim.a.kicker': 'Real-World Validation',
    'sim.a.t1': 'Simulation-trained,',
    'sim.a.t2': 'deployed on a real Franka',
    'cmp.kicker': 'Engine Comparison',
    'cmp.title1': 'One engine,',
    'cmp.title2': 'every box checked',
    'cmp.lead': 'Against recent robot simulation and data-generation engines, RoboGenesis is the only one that checks every box — automatic scene and task generation, domain randomization, success QA, structured annotations, cross-embodiment reuse, and lab protocols.',
    'cmp.tbl.ttl': 'Simulation & data-generation engines · feature comparison',
    'cmp.note': '✓ = explicitly supported in the cited source; — = not targeted or not clearly reported. RoboGenesis pairs 10+ robot embodiments with the complete feature set.',
    'cmp.det.ttl': 'Structured annotation example',
    'cmp.det.cap': 'Structured annotations in action — per-frame detections of the task-relevant objects (source / target beaker) from a pour episode camera stream.',
  });
  Object.assign(I18N.zh, {
    'sim.a.kicker': '真机验证',
    'sim.a.t1': '纯仿真训练，',
    'sim.a.t2': '直接部署到真实 Franka',
    'cmp.kicker': '引擎对比',
    'cmp.title1': '同类引擎对比，',
    'cmp.title2': '全项能力覆盖',
    'cmp.lead': '与近年的机器人仿真与数据生成引擎相比，RoboGenesis 是唯一全项支持的引擎——自动场景与任务生成、域随机化、成功率质检、结构化标注、跨机器人复用与实验室协议。',
    'cmp.tbl.ttl': '仿真与数据生成引擎 · 能力对比',
    'cmp.note': '✓ 表示原文明确支持；— 表示未涉及或未明确说明。RoboGenesis 以 10+ 种机器人覆盖全部能力项。',
    'cmp.det.ttl': '结构化标注示例',
    'cmp.det.cap': '结构化标注示例——pour 任务相机画面中，对任务相关物体（源烧杯 / 目标烧杯）的逐帧检测框。',
  });


  /* ---------- Edit mode: text override storage ---------- */
  const TEXT_EDITS_KEY = 'labvla-text-edits';
  let TEXT_EDITS = {};
  try { TEXT_EDITS = JSON.parse(localStorage.getItem(TEXT_EDITS_KEY)) || {}; } catch (_) { TEXT_EDITS = {}; }
  let refreshEditUi = null;

  const DEFAULT_LANG = 'en';
  let currentLang = DEFAULT_LANG;
  const setMetaContent = (selector, value) => {
    const el = $(selector);
    if (el && value) el.setAttribute('content', value);
  };
  let refreshSceneLabel = null;
  const applyLanguage = (lang) => {
    const dict = I18N[lang] || I18N.en;
    currentLang = lang;
    document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en';
    document.title = dict['meta.og_title'] || 'LabVLA';
    setMetaContent('meta[name="description"]', dict['meta.description']);
    setMetaContent('meta[property="og:title"]', dict['meta.og_title']);
    setMetaContent('meta[property="og:description"]', dict['meta.og_description']);
    $$('[data-i18n]').forEach((el) => {
      const edited = TEXT_EDITS[lang] ? TEXT_EDITS[lang][el.dataset.i18n] : undefined;
      const text = (typeof edited === 'string') ? edited : dict[el.dataset.i18n];
      if (typeof text === 'string') el.textContent = text;
    });
    $$('[data-i18n-aria]').forEach((el) => {
      const text = dict[el.dataset.i18nAria];
      if (typeof text === 'string') el.setAttribute('aria-label', text);
    });
    $$('[data-i18n-alt]').forEach((el) => {
      const text = dict[el.dataset.i18nAlt];
      if (typeof text === 'string') el.setAttribute('alt', text);
    });
    if (typeof refreshSceneLabel === 'function') refreshSceneLabel();
    if (typeof refreshEditUi === 'function') refreshEditUi();
  };

  /* ---------- Year stamp ---------- */
  const yearEl = $('#year');
  if (yearEl) yearEl.textContent = new Date().getFullYear();

  applyLanguage(DEFAULT_LANG);
  const langToggle = $('#langToggle');
  if (langToggle) {
    langToggle.addEventListener('click', () => {
      applyLanguage(currentLang === 'en' ? 'zh' : 'en');
    });
  }
  /* ---------- Nav scroll state ---------- */
  const benchmarkSwitch = $('.table-switch');
  if (benchmarkSwitch) {
    const splitButtons = $$('[data-result-split]', benchmarkSwitch);
    const splitGroups = $$('[data-result-group]');
    const setBenchmarkSplit = (split) => {
      splitButtons.forEach((btn) => {
        const active = btn.dataset.resultSplit === split;
        btn.classList.toggle('is-active', active);
        btn.setAttribute('aria-selected', active ? 'true' : 'false');
        btn.setAttribute('aria-pressed', active ? 'true' : 'false');
        btn.tabIndex = active ? 0 : -1;
      });
      splitGroups.forEach((group) => {
        group.hidden = group.dataset.resultGroup !== split;
      });
    };

    benchmarkSwitch.addEventListener('click', (event) => {
      const btn = event.target.closest && event.target.closest('[data-result-split]');
      if (!btn || !benchmarkSwitch.contains(btn)) return;
      event.preventDefault();
      event.stopPropagation();
      setBenchmarkSplit(btn.dataset.resultSplit);
    }, true);

    splitButtons.forEach((btn) => {
      btn.addEventListener('keydown', (event) => {
        if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
        event.preventDefault();
        const currentIndex = splitButtons.indexOf(btn);
        const dir = event.key === 'ArrowRight' ? 1 : -1;
        const next = splitButtons[(currentIndex + dir + splitButtons.length) % splitButtons.length];
        next.focus();
        setBenchmarkSplit(next.dataset.resultSplit);
      });
    });
    setBenchmarkSplit('id');
  }

  const realTaskSwitch = $('#sim2real .table-switch--real');
  if (realTaskSwitch) {
    const realTable = $('#sim2real .results-table--real');
    const realDemoCards = $$('#sim2real [data-real-demo]');
    const taskButtons = $$('[data-real-task]', realTaskSwitch);
    const setRealTask = (task) => {
      taskButtons.forEach((btn) => {
        const active = btn.dataset.realTask === task;
        btn.classList.toggle('is-active', active);
        btn.setAttribute('aria-selected', active ? 'true' : 'false');
        btn.setAttribute('aria-pressed', active ? 'true' : 'false');
        btn.tabIndex = active ? 0 : -1;
      });
      if (realTable) realTable.dataset.realView = task;
      realDemoCards.forEach((card) => {
        const match = task !== 'avg' && card.dataset.realDemo === task;
        card.classList.toggle('is-active', match);
      });
    };

    taskButtons.forEach((btn) => {
      btn.addEventListener('click', () => setRealTask(btn.dataset.realTask));
      btn.addEventListener('keydown', (event) => {
        if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
        event.preventDefault();
        const currentIndex = taskButtons.indexOf(btn);
        const dir = event.key === 'ArrowRight' ? 1 : -1;
        const next = taskButtons[(currentIndex + dir + taskButtons.length) % taskButtons.length];
        next.focus();
        setRealTask(next.dataset.realTask);
      });
    });
    realDemoCards.forEach((card) => {
      card.addEventListener('click', () => {
        const task = card.dataset.realDemo;
        if (!task) return;
        setRealTask(task);
        const btn = $(`[data-real-task="${task}"]`, realTaskSwitch);
        btn?.focus({ preventScroll: true });
      });
    });
    setRealTask('avg');
  }

  const nav = $('.nav');
  const onScroll = () => {
    if (!nav) return;
    if (window.scrollY > 12) nav.classList.add('scrolled');
    else nav.classList.remove('scrolled');
  };
  onScroll();
  window.addEventListener('scroll', onScroll, { passive: true });

  /* ---------- Hero mosaic generation ---------- */
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
      if (opts.objectPosition) v.style.objectPosition = opts.objectPosition;
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
      // Multi-column vertical infinite scroll
      mosaicBg.classList.add('mosaic-bg--flow', 'mosaic-bg--priming');
      const COLUMN_COUNT = 6;     // number of columns
      const PER_COLUMN = 4;       // base images per column (DOM duplicates ×2 for seamless loop)
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
          img.loading = 'lazy';
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
      // Fallback: animated color placeholder tiles
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

  /* Center 2×2 hero clips */
  $$('.big-tile').forEach((el, idx) => {
    const hue = [260, 200, 320, 160][idx] ?? 240;
    const delay = idx * 1.7;
    const dur = 10;
    el.appendChild(buildTileInner(HERO_BIG_VIDEOS[idx], hue, delay, dur, {
      poster: HERO_BIG_POSTERS[idx],
      preload: 'none',
      deferSrc: true,
      objectPosition: HERO_BIG_OBJECT_POSITION[idx],
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
  setTimeout(startDeferredVideos, 1500);

  /* ---------- Section anchor highlight ---------- */
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

  /* ---------- Scroll reveal animations ---------- */
  const revealTargets = $$('.section, .feature-card, .stat-card, .result-tile, .member-card, .panel');
  revealTargets.forEach(el => el.classList.add('reveal'));
  // Opt-in: single [data-reveal] nodes and staggered .reveal-children containers (styles in style.css)
  const revealOptIn = $$('[data-reveal], .reveal-children');
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
  [...revealTargets, ...revealOptIn].forEach(el => revealIO.observe(el));

  /* ---------- KPI count-up (respects prefers-reduced-motion) ---------- */
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const fmtCount = (val, dec) => val.toLocaleString(undefined, { minimumFractionDigits: dec, maximumFractionDigits: dec });
  const animateCount = (el) => {
    const target = parseFloat(el.dataset.count);
    if (Number.isNaN(target)) return;
    const dur = parseInt(el.dataset.countDur || '1300', 10);
    const dec = el.dataset.countDec != null ? parseInt(el.dataset.countDec, 10) : ((String(el.dataset.count).split('.')[1] || '').length);
    const prefix = el.dataset.countPrefix || '';
    const suffix = el.dataset.countSuffix || '';
    if (reduceMotion) { el.textContent = prefix + fmtCount(target, dec) + suffix; return; }
    const start = performance.now();
    const step = (now) => {
      const p = Math.min((now - start) / dur, 1);
      const eased = 1 - Math.pow(1 - p, 3);
      el.textContent = prefix + fmtCount(target * eased, dec) + suffix;
      if (p < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  };
  const countIO = new IntersectionObserver(
    entries => {
      entries.forEach(e => {
        if (e.isIntersecting) { animateCount(e.target); countIO.unobserve(e.target); }
      });
    },
    { threshold: 0.4 }
  );
  $$('[data-count]').forEach(el => countIO.observe(el));

  /* ---------- Image lightbox (click to enlarge) ---------- */
  const lightbox = $('#lightbox');
  if (lightbox) {
    const lbImg = $('img', lightbox);
    const openLB = (src, alt) => {
      if (!src) return;
      lbImg.src = src; lbImg.alt = alt || '';
      lightbox.classList.add('open'); lightbox.setAttribute('aria-hidden', 'false');
    };
    const closeLB = () => {
      lightbox.classList.remove('open'); lightbox.setAttribute('aria-hidden', 'true'); lbImg.src = '';
    };
    $$('.fig__plate img').forEach((img) => img.addEventListener('click', () => openLB(img.currentSrc || img.src, img.alt)));
    $$('.fig__zoom').forEach((btn) => btn.addEventListener('click', () => {
      const im = $('.fig__plate img', btn.closest('.fig'));
      if (im) openLB(im.currentSrc || im.src, im.alt);
    }));
    lightbox.addEventListener('click', (e) => {
      if (e.target === lightbox || e.target.classList.contains('lightbox__close')) closeLB();
    });
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && lightbox.classList.contains('open')) closeLB(); });
  }

  /* ---------- Real-robot inline player (FIG. 03 play buttons) ---------- */
  const realPlayGrid = $('#sim2real .real-play-grid');
  const realPlayViewer = $('#realPlayViewer');
  const realPlayVideo = $('#realPlayVideo');
  if (realPlayGrid && realPlayViewer && realPlayVideo) {
    const playBtns = $$('.real-play-btn', realPlayGrid);
    const playClip = (btn) => {
      const src = btn.dataset.realSrc;
      const poster = btn.dataset.realPoster;
      if (!src) return;
      playBtns.forEach((b) => b.classList.toggle('is-active', b === btn));
      realPlayVideo.src = src;
      if (poster) realPlayVideo.poster = poster;
      realPlayViewer.classList.remove('is-empty');
      realPlayVideo.currentTime = 0;
      realPlayVideo.play().catch(() => {});
    };
    playBtns.forEach((btn) => {
      btn.addEventListener('click', () => playClip(btn));
    });
    realPlayVideo.addEventListener('volumechange', () => { realPlayVideo.muted = true; });
    if (playBtns[0]) playClip(playBtns[0]);
  }

  /* ---------- Capability pyramid: level switch (highlight tier + detail panel) ---------- */
  const capTiers = $$('.tier');
  if (capTiers.length) {
    const capDetails = $$('.pdetail');
    const showCap = (lvl) => {
      capTiers.forEach((t) => {
        const on = t.dataset.level === lvl;
        t.classList.toggle('is-active', on);
        t.setAttribute('aria-selected', on ? 'true' : 'false');
      });
      capDetails.forEach((d) => d.classList.toggle('is-shown', d.dataset.level === lvl));
    };
    capTiers.forEach((t) => t.addEventListener('click', () => showCap(t.dataset.level)));
    showCap('2'); // default: LabVLA / Technician
  }

  /* ---------- LabVLA training recipe: stage switch (highlight diagram + panel) ---------- */
  const recipeTabs = $$('.recipe-tab');
  const recipeFlow = $('#recipeFlow');
  const recipePanels = $$('[data-stage-panel]');
  if (recipeTabs.length && recipeFlow) {
    recipeTabs.forEach((tab) => {
      tab.addEventListener('click', () => {
        const stage = tab.dataset.stage;
        recipeTabs.forEach((t) => {
          const on = t === tab;
          t.classList.toggle('is-active', on);
          t.setAttribute('aria-selected', String(on));
        });
        recipeFlow.dataset.stage = stage;
        recipePanels.forEach((p) => { p.hidden = p.dataset.stagePanel !== stage; });
      });
    });
  }

  /* ---------- Accordion ---------- */
  $$('.accordion--approach').forEach((accordion) => {
    const items = $$('.acc-item', accordion);
    items.forEach((item) => {
      const summary = item.querySelector('summary');
      if (!summary) return;
      summary.addEventListener('click', (e) => {
        e.preventDefault();
        const other = items.find((el) => el !== item);
        if (item.open) {
          item.open = false;
          if (other) other.open = true;
        } else {
          item.open = true;
          if (other) other.open = false;
        }
      });
    });
  });
  $$('.accordion:not(.accordion--approach)').forEach((accordion) => {
    const items = $$('.acc-item', accordion);
    if (!items.length) return;
    items.forEach((item) => {
      item.addEventListener('toggle', () => {
        if (item.open) return;
        if (items.every((el) => !el.open)) item.open = true;
      });
    });
  });

  /* ---------- Randomization card toggles ---------- */
  $$('[data-random-src]').forEach(btn => {
    const img = $('img', btn);
    btn.dataset.state = 'advanced';
    btn.addEventListener('click', () => {
      const nextState = btn.dataset.state === 'advanced' ? 'base' : 'advanced';
      const src = nextState === 'base' ? btn.dataset.baseSrc : btn.dataset.randomSrc;
      if (!src || !img) return;
      img.src = src;
      btn.dataset.state = nextState;
      btn.classList.toggle('is-base', nextState === 'base');
    });
  });

  /* ---------- BibTeX copy ---------- */
  /* ---------- Scene progression switcher ---------- */
  const scenePreview = $('[data-scene-preview]');
  const scenePreviewLabel = $('[data-scene-preview-label]');
  const sceneCountLabel = $('[data-scene-count-label]');
  const sceneStageLabel = $('[data-scene-stage-label]');
  const sceneStageControls = $$('[data-scene-stage]');
  const sceneStageItems = $$('[data-scene-stage-item]');
  const sceneProgress = $('.scene-progress');
  const sceneProgressSteps = $$('.scene-progress__step[data-scene-stage]');
  const sceneStageSlugs = [
    '01-empty-room',
    '02-tables-counters',
    '03-lab-equipment',
    '04-tabletop-kit',
    '05-materials-textures',
    '06-wall-signs',
    '07-robot-task-objects',
  ];
  const sceneStageKeys = [
    'scene.stage1',
    'scene.stage2',
    'scene.stage3',
    'scene.stage4',
    'scene.stage5',
    'scene.stage6',
    'scene.stage7',
  ];
  const sceneItems = [
    { prefix: 'small620116', labelKey: 'scene.small620116', label: 'Lab Scene 03' },
    { prefix: 'large630101', labelKey: 'scene.large630101', label: 'Lab Scene 04' },
    { prefix: 'large630109', labelKey: 'scene.large630109', label: 'Lab Scene 05' },
    { prefix: 'large630113', labelKey: 'scene.large630113', label: 'Lab Scene 06' },
  ];
  const defaultSceneStageIndex = sceneStageSlugs.length - 1;
  let activeSceneIndex = 0;
  let activeStageIndex = defaultSceneStageIndex;
  let isSceneProgressDragging = false;
  let sceneProgressPointerId = null;
  let sceneProgressSuppressClick = false;
  const twoDigit = (value) => String(value).padStart(2, '0');
  const clampStage = (index) => Math.min(Math.max(index, 0), sceneStageSlugs.length - 1);
  const stageFromProgressPoint = (clientX) => {
    if (!sceneProgress || !sceneProgressSteps.length) return activeStageIndex;
    let bestStage = activeStageIndex;
    let bestDistance = Infinity;
    sceneProgressSteps.forEach((step, index) => {
      const rect = step.getBoundingClientRect();
      const distance = Math.abs(clientX - (rect.left + rect.width / 2));
      if (distance < bestDistance) {
        bestDistance = distance;
        bestStage = index;
      }
    });
    return clampStage(bestStage);
  };
  const setSceneProgressThumb = (stageIndex, clientX = null) => {
    if (!sceneProgress || !sceneProgressSteps.length) return;
    const stage = clampStage(stageIndex);
    const step = sceneProgressSteps[stage];
    const progressRect = sceneProgress.getBoundingClientRect();
    const stepRect = step.getBoundingClientRect();
    if (!progressRect.width || !stepRect.width) return;
    let left = stepRect.left - progressRect.left;
    const width = stepRect.width;
    if (typeof clientX === 'number') {
      const firstRect = sceneProgressSteps[0].getBoundingClientRect();
      const lastRect = sceneProgressSteps[sceneProgressSteps.length - 1].getBoundingClientRect();
      const minLeft = firstRect.left - progressRect.left;
      const maxLeft = lastRect.left - progressRect.left;
      left = Math.min(Math.max(clientX - progressRect.left - width / 2, minLeft), maxLeft);
    }
    sceneProgress.style.setProperty('--scene-active-left', `${left}px`);
    sceneProgress.style.setProperty('--scene-active-width', `${width}px`);
  };
  const applySceneProgressDrag = (clientX) => {
    const stage = stageFromProgressPoint(clientX);
    if (stage !== activeStageIndex) setScene(activeSceneIndex, stage);
    setSceneProgressThumb(stage, clientX);
  };
  const getStageLabel = (dict, index) => {
    const template = dict['scene.stage_label'] || I18N.en['scene.stage_label'];
    return template
      .replace('{current}', twoDigit(index + 1))
      .replace('{total}', twoDigit(sceneStageSlugs.length));
  };
  const getSceneCountLabel = (dict, index) => {
    const template = dict['scene.index_label'] || I18N.en['scene.index_label'];
    return template
      .replace('{current}', twoDigit(index + 1))
      .replace('{total}', twoDigit(sceneItems.length));
  };
  const setScene = (index, stageIndex = activeStageIndex) => {
    if (!scenePreview) return;
    activeSceneIndex = (index + sceneItems.length) % sceneItems.length;
    activeStageIndex = clampStage(stageIndex);
    const item = sceneItems[activeSceneIndex];
    const dict = I18N[currentLang] || I18N.en;
    const sceneLabel = dict[item.labelKey] || item.label;
    const stageText = dict[sceneStageKeys[activeStageIndex]] || I18N.en[sceneStageKeys[activeStageIndex]];
    const stageLabel = getStageLabel(dict, activeStageIndex);
    scenePreview.src = `assets/media/scene-generation/progression-${item.prefix}-${sceneStageSlugs[activeStageIndex]}.jpg`;
    scenePreview.alt = `${sceneLabel} - ${stageText}`;
    if (scenePreviewLabel) {
      scenePreviewLabel.dataset.i18n = item.labelKey;
      scenePreviewLabel.textContent = sceneLabel;
    }
    if (sceneCountLabel) sceneCountLabel.textContent = getSceneCountLabel(dict, activeSceneIndex);
    if (sceneStageLabel) sceneStageLabel.textContent = stageLabel;
    sceneStageControls.forEach(control => {
      const controlStage = clampStage(Number(control.dataset.sceneStage) || 0);
      const isActive = controlStage === activeStageIndex;
      const controlStageText = dict[sceneStageKeys[controlStage]] || I18N.en[sceneStageKeys[controlStage]];
      const controlStageLabel = getStageLabel(dict, controlStage);
      if (control.classList.contains('scene-progress__step')) {
        control.classList.toggle('is-active', isActive);
      }
      control.setAttribute('aria-pressed', String(isActive));
      control.setAttribute('aria-label', `${controlStageLabel}: ${controlStageText}`);
    });
    sceneStageItems.forEach(itemEl => {
      const isActive = Number(itemEl.dataset.sceneStageItem) === activeStageIndex;
      itemEl.classList.toggle('is-active', isActive);
      if (isActive) itemEl.setAttribute('aria-current', 'step');
      else itemEl.removeAttribute('aria-current');
    });
    if (!isSceneProgressDragging) {
      requestAnimationFrame(() => setSceneProgressThumb(activeStageIndex));
    }
  };
  refreshSceneLabel = () => {
    setScene(activeSceneIndex, activeStageIndex);
  };
  $$('[data-scene-step]').forEach(btn => {
    btn.addEventListener('click', () => {
      const step = Number(btn.dataset.sceneStep) || 1;
      setScene(activeSceneIndex + step, defaultSceneStageIndex);
    });
  });
  sceneStageControls.forEach(btn => {
    btn.addEventListener('click', () => {
      setScene(activeSceneIndex, Number(btn.dataset.sceneStage) || 0);
    });
  });
  if (sceneProgress) {
    sceneProgress.addEventListener('click', (event) => {
      if (!sceneProgressSuppressClick) return;
      event.preventDefault();
      event.stopPropagation();
      sceneProgressSuppressClick = false;
    }, true);
    sceneProgress.addEventListener('pointerdown', (event) => {
      if (event.button && event.button !== 0) return;
      isSceneProgressDragging = true;
      sceneProgressPointerId = event.pointerId;
      sceneProgressSuppressClick = true;
      sceneProgress.classList.add('is-dragging');
      if (sceneProgress.setPointerCapture) {
        try { sceneProgress.setPointerCapture(event.pointerId); } catch (_) {}
      }
      applySceneProgressDrag(event.clientX);
      event.preventDefault();
    });
    sceneProgress.addEventListener('pointermove', (event) => {
      if (!isSceneProgressDragging || event.pointerId !== sceneProgressPointerId) return;
      applySceneProgressDrag(event.clientX);
      event.preventDefault();
    });
    const endSceneProgressDrag = (event) => {
      if (!isSceneProgressDragging || event.pointerId !== sceneProgressPointerId) return;
      const stage = stageFromProgressPoint(event.clientX);
      isSceneProgressDragging = false;
      sceneProgressPointerId = null;
      sceneProgress.classList.remove('is-dragging');
      if (sceneProgress.releasePointerCapture) {
        try { sceneProgress.releasePointerCapture(event.pointerId); } catch (_) {}
      }
      setScene(activeSceneIndex, stage);
    };
    sceneProgress.addEventListener('pointerup', endSceneProgressDrag);
    sceneProgress.addEventListener('pointercancel', endSceneProgressDrag);
    window.addEventListener('resize', () => setSceneProgressThumb(activeStageIndex), { passive: true });
  }
  setScene(0, defaultSceneStageIndex);

  /* ---------- Atomic task video lazy loading ---------- */
  const taskVideos = $$('#stats .task-card__media video[data-task-src]');
  taskVideos.forEach(video => {
    const setTaskVideoSpeed = () => {
      video.playbackRate = 2.5;
    };
    setTaskVideoSpeed();
    video.addEventListener('loadedmetadata', setTaskVideoSpeed);
    video.addEventListener('play', setTaskVideoSpeed);
  });
  const loadTaskVideo = (video) => {
    if (!video.dataset.taskSrc) return;
    video.src = video.dataset.taskSrc;
    delete video.dataset.taskSrc;
    video.load();
  };
  const taskVideoIO = new IntersectionObserver(
    entries => {
      entries.forEach(entry => {
        const video = entry.target;
        if (entry.isIntersecting) {
          loadTaskVideo(video);
          video.play().catch(() => {});
        } else if (!video.paused) {
          video.pause();
        }
      });
    },
    { rootMargin: '320px 0px', threshold: 0.05 }
  );
  taskVideos.forEach(video => taskVideoIO.observe(video));

  /* ---------- Showcase video lazy loading ---------- */
  const showcaseVideos = $$('video[data-video-src]');
  const loadShowcaseVideo = (video) => {
    if (!video.dataset.videoSrc) return;
    video.src = video.dataset.videoSrc;
    delete video.dataset.videoSrc;
    video.load();
  };
  const showcaseVideoIO = new IntersectionObserver(
    entries => {
      entries.forEach(entry => {
        const video = entry.target;
        if (entry.isIntersecting) {
          loadShowcaseVideo(video);
          video.play().catch(() => {});
        } else if (!video.paused) {
          video.pause();
        }
      });
    },
    { rootMargin: '320px 0px', threshold: 0.05 }
  );
  showcaseVideos.forEach(video => showcaseVideoIO.observe(video));

  /* ---------- Warm all deferred media after hero is ready (compressed assets) ---------- */
  const warmAllDeferredVideos = () => {
    taskVideos.forEach(loadTaskVideo);
    showcaseVideos.forEach(loadShowcaseVideo);
  };
  const scheduleWarmAll = () => {
    if ('requestIdleCallback' in window) {
      requestIdleCallback(warmAllDeferredVideos, { timeout: 5000 });
    } else {
      setTimeout(warmAllDeferredVideos, 5000);
    }
  };
  if (tileLoadPromises.length) {
    Promise.allSettled(tileLoadPromises).then(scheduleWarmAll);
  } else {
    scheduleWarmAll();
  }

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

  /* ---------- Data Engine collapse module ---------- */
  const deGroup = $('#data-engine');
  const deToggle = $('#dataEngineToggle');
  const deBody = $('#dataEngineBody');
  if (deGroup && deToggle && deBody) {
    const deHint = $('.de-portrait-toggle__label', deToggle);
    // Inner Data Engine section ids + global nav links (hidden when collapsed, restored when expanded)
    const deInnerIds = $$('section[id]', deBody).map((s) => s.id);
    const deNavLinks = $$('#globalNav .nav-link').filter((a) =>
      deInnerIds.includes((a.getAttribute('href') || '').replace('#', ''))
    );
    // Cover nav link (#features): relabel to "Data Engine" when collapsed, restore when expanded
    const deCoverNavLink = $$('#globalNav .nav-link').find((a) => (a.getAttribute('href') || '') === '#features');
    // Swap global nav out / Data Engine sub-nav in when expanded and section is in view
    const deHeader = $('.nav');
    let deInView = false;
    const deNavEl = $('#deNav');
    const globalNavEl = $('#globalNav');
    const updateDeNav = () => {
      const active = !deGroup.classList.contains('is-collapsed') && deInView;
      if (deHeader) deHeader.classList.toggle('is-de-nav', active);
      if (deNavEl) deNavEl.setAttribute('aria-hidden', String(!active));
      if (globalNavEl) globalNavEl.setAttribute('aria-hidden', String(active));
    };
    const deNavIO = new IntersectionObserver((entries) => {
      entries.forEach((e) => { deInView = e.isIntersecting; });
      updateDeNav();
    }, { rootMargin: '-70px 0px -42% 0px', threshold: 0 });
    deNavIO.observe(deBody);
    const NAV_ANIM_MS = 340;
    // Inner nav links: fade up and hide on collapse; staggered fade-in on expand
    const animateNavLinks = (collapsed) => {
      deNavLinks.forEach((a, i) => {
        if (collapsed) {
          a.classList.add('de-nav-hide');
          window.setTimeout(() => {
            if (deGroup.classList.contains('is-collapsed')) a.hidden = true;
            a.classList.remove('de-nav-hide');
          }, NAV_ANIM_MS);
        } else {
          a.hidden = false;
          a.classList.add('de-nav-hide');
          requestAnimationFrame(() => requestAnimationFrame(() => {
            a.style.transitionDelay = (i * 45) + 'ms';
            a.classList.remove('de-nav-hide');
            window.setTimeout(() => { a.style.transitionDelay = ''; }, NAV_ANIM_MS + i * 45);
          }));
        }
      });
    };
    // Cover nav label: quick fade out → swap text → fade in
    const setCoverLabel = (collapsed, animate) => {
      if (!deCoverNavLink) return;
      const dict = I18N[currentLang] || I18N.en;
      deCoverNavLink.dataset.i18n = collapsed ? 'de.navCollapsed' : 'nav.features';
      const text = dict[deCoverNavLink.dataset.i18n] || deCoverNavLink.textContent;
      if (!animate) { deCoverNavLink.textContent = text; return; }
      deCoverNavLink.classList.add('de-relabel');
      window.setTimeout(() => {
        deCoverNavLink.textContent = text;
        deCoverNavLink.classList.remove('de-relabel');
      }, 170);
    };
    const setDeCollapsed = (collapsed, { animate = true } = {}) => {
      deGroup.classList.toggle('is-collapsed', collapsed);
      deToggle.setAttribute('aria-expanded', String(!collapsed));
      deBody.setAttribute('aria-hidden', String(collapsed));
      if (animate) animateNavLinks(collapsed);
      else deNavLinks.forEach((a) => { a.hidden = collapsed; });
      setCoverLabel(collapsed, animate);
      if (deHint) {
        deHint.dataset.i18n = collapsed ? 'de.expand' : 'de.collapse';
        const dict = I18N[currentLang] || I18N.en;
        deHint.textContent = dict[deHint.dataset.i18n] || deHint.textContent;
      }
      if (collapsed && deHeader) deHeader.classList.remove('is-de-nav');
      updateDeNav();
    };
    // Default collapsed: show portrait cover only; expand on "View Data Engine details"
    setDeCollapsed(true, { animate: false });
    const NAV_OFFSET = 80; // land first section title just below fixed nav (h-16 = 64px)
    const scrollToDetail = () => {
      const firstSection = deBody.querySelector('section[id]');
      // Anchor to .section-head, not section top (section has 6rem padding)
      const anchor = (firstSection && firstSection.querySelector('.section-head')) || firstSection || deBody;
      const top = anchor.getBoundingClientRect().top + window.pageYOffset - NAV_OFFSET;
      window.scrollTo({ top: Math.max(top, 0), behavior: 'smooth' });
    };
    deToggle.addEventListener('click', () => {
      const willExpand = deGroup.classList.contains('is-collapsed');
      setDeCollapsed(!willExpand);
      if (willExpand) {
        // After expand, scroll to first detail screen; wait for grid-rows transition (~0.5s)
        let done = false;
        const go = () => { if (done) return; done = true; scrollToDetail(); };
        const onEnd = (e) => {
          if (e.target === deBody && e.propertyName === 'grid-template-rows') {
            deBody.removeEventListener('transitionend', onEnd);
            go();
          }
        };
        deBody.addEventListener('transitionend', onEnd);
        setTimeout(() => { deBody.removeEventListener('transitionend', onEnd); go(); }, 600);
      }
    });
    /* When collapsed, deep links into inner sections auto-expand then smooth-scroll */
    document.addEventListener('click', (e) => {
      const link = e.target.closest('a[href^="#"]');
      if (!link) return;
      const id = (link.getAttribute('href') || '').replace('#', '');
      if (!id || !deInnerIds.includes(id)) return;
      if (deGroup.classList.contains('is-collapsed')) {
        setDeCollapsed(false);
        const target = document.getElementById(id);
        if (target) {
          e.preventDefault();
          requestAnimationFrame(() =>
            target.scrollIntoView({ behavior: 'smooth', block: 'start' })
          );
        }
      }
    });
  }

  /* ---------- Edit mode: in-page visual text editing ---------- */
  (() => {
    const editToggle = $('#editToggle');
    if (!editToggle) return;

    const L = {
      en: { edit: 'Edit', done: 'Done', exp: 'Export', reset: 'Reset', confirmReset: 'Discard all text edits for the current language?' },
      zh: { edit: '编辑', done: '完成', exp: '导出', reset: '重置', confirmReset: '清除当前语言的全部文字修改？' },
    };
    const t = (k) => (L[currentLang] || L.en)[k];

    const bar = document.createElement('div');
    bar.className = 'edit-bar';
    const btnExport = document.createElement('button');
    const btnReset = document.createElement('button');
    const btnDone = document.createElement('button');
    btnExport.type = btnReset.type = btnDone.type = 'button';
    btnDone.className = 'edit-bar__done';
    bar.append(btnExport, btnReset, btnDone);
    document.body.appendChild(bar);

    let editing = false;
    const save = () => { try { localStorage.setItem(TEXT_EDITS_KEY, JSON.stringify(TEXT_EDITS)); } catch (_) {} };
    const refreshLabels = () => {
      editToggle.textContent = editing ? t('done') : t('edit');
      btnExport.textContent = t('exp');
      btnReset.textContent = t('reset');
      btnDone.textContent = t('done');
    };
    refreshEditUi = refreshLabels;

    const setEditing = (on) => {
      editing = on;
      document.body.classList.toggle('is-editing', on);
      editToggle.classList.toggle('is-on', on);
      $$('[data-i18n]').forEach((el) => {
        if (on) {
          el.setAttribute('contenteditable', 'plaintext-only');
          if (!el.isContentEditable) el.setAttribute('contenteditable', 'true');
        } else {
          el.removeAttribute('contenteditable');
        }
      });
      refreshLabels();
    };

    editToggle.addEventListener('click', () => setEditing(!editing));
    btnDone.addEventListener('click', () => setEditing(false));

    /* While editing: persist overrides for the active language on input */
    document.addEventListener('input', (e) => {
      if (!editing) return;
      const el = e.target.closest && e.target.closest('[data-i18n]');
      if (!el) return;
      (TEXT_EDITS[currentLang] = TEXT_EDITS[currentLang] || {})[el.dataset.i18n] = el.textContent;
      save();
    }, true);

    /* While editing: block link/button defaults; Enter blurs the active field */
    document.addEventListener('click', (e) => {
      if (!editing || bar.contains(e.target) || e.target === editToggle) return;
      if (e.target.closest && e.target.closest('[data-i18n]')) { e.preventDefault(); e.stopPropagation(); }
    }, true);
    document.addEventListener('keydown', (e) => {
      if (!editing || e.key !== 'Enter') return;
      const el = e.target.closest && e.target.closest('[data-i18n]');
      if (el) { e.preventDefault(); el.blur(); }
    }, true);

    btnExport.addEventListener('click', () => {
      const blob = new Blob([JSON.stringify(TEXT_EDITS, null, 2)], { type: 'application/json' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'labvla-text-edits.json';
      a.click();
      URL.revokeObjectURL(a.href);
    });
    btnReset.addEventListener('click', () => {
      if (!window.confirm(t('confirmReset'))) return;
      delete TEXT_EDITS[currentLang];
      save();
      applyLanguage(currentLang);
    });

    refreshLabels();
  })();

  /* ---------- Site analytics: PV only (Vercount API) ---------- */
  const pvEl = $('#busuanzi_value_site_pv');
  if (pvEl) {
    const PV_KEY = 'labvla-site-pv';
    const VERCOUNT_API = 'https://events.vercount.one/api/v2/log';
    pvEl.classList.add('is-loading');

    const showCount = (n) => {
      if (Number.isFinite(n)) pvEl.textContent = n.toLocaleString();
    };
    const finish = () => pvEl.classList.remove('is-loading');

    const bumpLocal = () => {
      const pv = (parseInt(localStorage.getItem(PV_KEY) || '0', 10) || 0) + 1;
      localStorage.setItem(PV_KEY, String(pv));
      showCount(pv);
      finish();
    };

    const track = () => {
      if (location.protocol !== 'http:' && location.protocol !== 'https:') {
        bumpLocal();
        return;
      }

      const ctrl = new AbortController();
      const timer = setTimeout(() => ctrl.abort(), 8000);
      fetch(VERCOUNT_API, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: location.href, isNewUv: false }),
        signal: ctrl.signal,
      })
        .then((res) => (res.ok ? res.json() : Promise.reject()))
        .then((json) => {
          const pv = Number(json?.data?.site_pv ?? json?.site_pv);
          if (!Number.isFinite(pv)) throw new Error('no data');
          showCount(pv);
          finish();
        })
        .catch(bumpLocal)
        .finally(() => clearTimeout(timer));
    };

    track();
  }

})();
