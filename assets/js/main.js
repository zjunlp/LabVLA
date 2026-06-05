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
      'nav.features': 'Robots',
      'nav.dataset': 'Tasks',
      'nav.dual_tasks': 'Dual-arm',
      'nav.method': 'Randomization',
      'nav.assets': 'Assets',
      'nav.results': 'Scenes',
      'nav.sim2real': 'Sim-to-Real',
      'nav.bibtex': 'Citation',
      'nav.team': 'Team',
      'link.paper': 'Paper',
      'link.code': 'Code',
      'link.dataset': 'Dataset',
      'link.demo': 'Demo Video',
      'features.title': 'Multi-Embodiment Robot Platform',
      'features.subtitle': 'Eleven heterogeneous manipulator embodiments — eight single-arm and three dual-arm — share a unified action interface across diverse manipulation tasks.',
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
      'robots.single.count': '7 platforms',
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
      'dataset.subtitle': 'Ten atomic lab tasks can be demonstrated independently or composed into longer workflows.',
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
      'tasks.atomic.09': 'Atomic Task 09',
      'tasks.atomic.10': 'Atomic Task 10',
      'workflow.kicker': 'Composable Workflow',
      'workflow.title': 'Atomic tasks can be chained into a complete lab workflow.',
      'workflow.desc': 'A complete rollout composes multiple atomic skills into a longer executable lab workflow.',
      'workflow.step1': 'Task 01',
      'workflow.step2': 'Task 04',
      'workflow.step3': 'Task 07',
      'workflow.step4': 'Task 10',
      'dual_tasks.title': 'Dual-arm Task Showcase',
      'dual_tasks.subtitle': 'Reserve four slots for bimanual manipulation tasks and coordinated dual-arm demonstrations.',
      'dual_tasks.tag': 'Dual-arm Task',
      'dual_tasks.slot1': 'Dual-arm task slot 01',
      'dual_tasks.slot2': 'Dual-arm task slot 02',
      'dual_tasks.slot3': 'Dual-arm task slot 03',
      'dual_tasks.slot4': 'Dual-arm task slot 04',
      'dual_tasks.card1': 'Dual-arm Task 01',
      'dual_tasks.card2': 'Dual-arm Task 02',
      'dual_tasks.card3': 'Dual-arm Task 03',
      'dual_tasks.card4': 'Dual-arm Task 04',
      'placeholder.figure_16_7': 'Figure · 16:7',
      'dataset.chart': 'Dataset statistics figure (task/object/embodiment distributions)',
      'dataset.caption': 'Figure: add caption here.',
      'method.title': 'Randomization',
      'method.subtitle': 'We randomize scene appearance, camera viewpoint, object layout, obstacles, and tabletop conditions to improve robustness.',
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
      'random.lighting': 'Lighting',
      'random.camera': 'Camera View',
      'random.position': 'Position',
      'random.objects': 'Objects',
      'random.obstacles': 'Obstacles',
      'random.tabletop_a': 'Tabletop A',
      'random.tabletop_b': 'Tabletop B',
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
      'results.subtitle': 'Generate diverse lab scenes from task requirements, objects, and layout constraints.',
      'placeholder.video': 'Video',
      'label.task': 'Task',
      'scene.kicker': 'Procedural Generation',
      'scene.title': 'From task specification to executable lab scenes.',
      'scene.desc': 'Generated scenes vary workspace layout, object placement, container positions, and environmental context.',
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
    },
    zh: {
      'meta.description': 'LabVLA 视觉-语言-动作研究项目页面。',
      'meta.og_title': 'LabVLA',
      'meta.og_description': 'LabVLA 视觉-语言-动作研究项目页面。',
      'lang.toggle': 'English',
      'lang.aria': '切换到英文',
      'nav.features': '亮点',
      'nav.dataset': '数据集',
      'nav.dual_tasks': '双臂任务',
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
      'dataset.chart': '数据集统计图（任务 / 物体 / 机器人分布等）',
      'dataset.caption': '图：此处填写图注。',
      'dual_tasks.title': '双臂任务展示',
      'dual_tasks.subtitle': '预留 4 个双臂协同操作任务展示位。',
      'dual_tasks.tag': '双臂任务',
      'dual_tasks.slot1': '双臂任务预留位 01',
      'dual_tasks.slot2': '双臂任务预留位 02',
      'dual_tasks.slot3': '双臂任务预留位 03',
      'dual_tasks.slot4': '双臂任务预留位 04',
      'dual_tasks.card1': '双臂任务 01',
      'dual_tasks.card2': '双臂任务 02',
      'dual_tasks.card3': '双臂任务 03',
      'dual_tasks.card4': '双臂任务 04',
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

  Object.assign(I18N.en, {
    'lang.toggle': '中文',
    'dataset.subtitle': 'A suite of atomic manipulation tasks can be demonstrated independently or composed into complete lab workflows.',
    'nav.dual_tasks': 'Dual-arm',
    'dual_tasks.title': 'Dual-arm Task Showcase',
    'dual_tasks.subtitle': 'Reserve four slots for bimanual manipulation tasks and coordinated dual-arm demonstrations.',
    'dual_tasks.tag': 'Dual-arm Task',
    'dual_tasks.slot1': 'Dual-arm task slot 01',
    'dual_tasks.slot2': 'Dual-arm task slot 02',
    'dual_tasks.slot3': 'Dual-arm task slot 03',
    'dual_tasks.slot4': 'Dual-arm task slot 04',
    'dual_tasks.card1': 'Dual-arm Task 01',
    'dual_tasks.card2': 'Dual-arm Task 02',
    'dual_tasks.card3': 'Dual-arm Task 03',
    'dual_tasks.card4': 'Dual-arm Task 04',
    'tasks.robot.01': 'UR5e',
    'tasks.robot.02': 'UR16e',
    'tasks.robot.03': 'Rizon 4',
    'tasks.robot.04': 'Franka',
    'tasks.robot.05': 'Festo',
    'tasks.robot.06': 'Franka',
    'tasks.robot.07': 'UR5e',
    'tasks.robot.08': 'FR3',
  });

  Object.assign(I18N.zh, {
    'meta.description': 'LabVLA 视觉-语言-动作研究项目页面。',
    'meta.og_title': 'LabVLA',
    'meta.og_description': 'LabVLA 视觉-语言-动作研究项目页面。',
    'lang.toggle': 'English',
    'lang.aria': '切换到英文',
    'nav.features': '机器臂',
    'nav.dataset': '多任务',
    'nav.dual_tasks': '双臂任务',
    'nav.method': '随机化',
    'nav.assets': '资产生成',
    'nav.results': '场景生成',
    'nav.sim2real': '仿真到真实',
    'nav.bibtex': '引用',
    'nav.team': '团队',
    'link.paper': '论文',
    'link.code': '代码',
    'link.dataset': '数据集',
    'link.demo': '演示视频',

    'features.title': '多种机器人平台',
    'features.subtitle': '集成 11 种不同的机械臂（8 台单臂、3 台双臂），共用统一动作接口，覆盖多样化操作任务。',
    'robots.single.title': '单臂机器人',
    'robots.single.count': '7 个平台',
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
    'tasks.atomic.09': '原子任务 09',
    'tasks.atomic.10': '原子任务 10',
    'tasks.robot.01': 'UR5e',
    'tasks.robot.02': 'UR16e',
    'tasks.robot.03': 'Rizon 4',
    'tasks.robot.04': 'Franka',
    'tasks.robot.05': 'Festo',
    'tasks.robot.06': 'Franka',
    'tasks.robot.07': 'UR5e',
    'tasks.robot.08': 'FR3',
    'workflow.kicker': '可组合工作流',
    'workflow.title': '原子任务可以串联成完整的实验工作流。',
    'workflow.desc': '完整 rollout 将多个原子技能组合成更长的可执行实验工作流。',
    'workflow.step1': '任务 01',
    'workflow.step2': '任务 04',
    'workflow.step3': '任务 07',
    'workflow.step4': '任务 10',
    'dual_tasks.title': '双臂任务展示',
    'dual_tasks.subtitle': '预留 4 个双臂协同操作任务展示位。',
    'dual_tasks.tag': '双臂任务',
    'dual_tasks.slot1': '双臂任务预留位 01',
    'dual_tasks.slot2': '双臂任务预留位 02',
    'dual_tasks.slot3': '双臂任务预留位 03',
    'dual_tasks.slot4': '双臂任务预留位 04',
    'dual_tasks.card1': '双臂任务 01',
    'dual_tasks.card2': '双臂任务 02',
    'dual_tasks.card3': '双臂任务 03',
    'dual_tasks.card4': '双臂任务 04',

    'method.title': '随机化',
    'method.subtitle': '通过随机化场景外观、相机视角、物体布局、障碍物与桌面条件，提升策略的鲁棒性。',
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
    'random.lighting': '光照',
    'random.position': '位置',
    'random.objects': '物体',
    'random.obstacles': '障碍物',
    'random.tabletop_b': '桌面变化',

    'assets.title': '资产生成',
    'assets.subtitle': '生成可复用的物体、容器、工具和场景道具，用于组合实验环境。',
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
    'results.subtitle': '根据任务需求、物体和布局约束生成多样化实验场景。',
    'scene.kicker': '程序化生成',
    'scene.title': '从任务规格生成可执行实验场景。',
    'scene.desc': '生成场景覆盖工作区布局、物体摆放、容器位置和环境上下文变化。',
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

  let currentLang = 'en';
  const EDIT_STORAGE_KEY = 'labvlaTextEdits';
  const loadTextEdits = () => {
    try {
      return JSON.parse(localStorage.getItem(EDIT_STORAGE_KEY) || '{}');
    } catch {
      return {};
    }
  };
  const saveTextEdits = (edits) => {
    localStorage.setItem(EDIT_STORAGE_KEY, JSON.stringify(edits));
  };
  let textEdits = loadTextEdits();
  const applyTextEdits = () => {
    $$('[data-i18n]').forEach((el) => {
      const key = `${currentLang}:${el.dataset.i18n}`;
      if (typeof textEdits[key] === 'string') el.textContent = textEdits[key];
    });
  };
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
      const text = dict[el.dataset.i18n];
      if (typeof text === 'string') el.textContent = text;
    });
    $$('[data-i18n-aria]').forEach((el) => {
      const text = dict[el.dataset.i18nAria];
      if (typeof text === 'string') el.setAttribute('aria-label', text);
    });
    applyTextEdits();
    if (typeof refreshSceneLabel === 'function') refreshSceneLabel();
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
  const editToggle = $('#editToggle');
  if (editToggle) {
    editToggle.addEventListener('click', () => {
      const isActive = document.body.classList.toggle('edit-mode');
      editToggle.classList.toggle('is-active', isActive);
      editToggle.textContent = isActive ? 'Done' : 'Edit';
    });
  }
  document.addEventListener('click', (event) => {
    if (!document.body.classList.contains('edit-mode')) return;
    const target = event.target.closest('[data-i18n]');
    if (!target || target.closest('#editToggle')) return;
    event.preventDefault();
    event.stopPropagation();
    const key = `${currentLang}:${target.dataset.i18n}`;
    const next = prompt('Edit text', target.textContent.trim());
    if (next === null) return;
    textEdits[key] = next;
    saveTextEdits(textEdits);
    target.textContent = next;
  }, true);
  document.addEventListener('keydown', (event) => {
    if (!document.body.classList.contains('edit-mode')) return;
    if (!event.altKey || event.key.toLowerCase() !== 'r') return;
    if (!confirm(`Reset edited ${currentLang.toUpperCase()} text?`)) return;
    Object.keys(textEdits).forEach((key) => {
      if (key.startsWith(`${currentLang}:`)) delete textEdits[key];
    });
    saveTextEdits(textEdits);
    applyLanguage(currentLang);
  });

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
      const COLUMN_COUNT = 6;     // 列数
      const PER_COLUMN = 4;       // 每列基础图片数（实际 DOM 复制 2 份以实现无缝循环）
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
  setTimeout(startDeferredVideos, 3200);

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

  /* ---------- BibTeX 复制 ---------- */
  /* ---------- Scene preview switcher ---------- */
  const scenePreview = $('[data-scene-preview]');
  const scenePreviewLabel = $('[data-scene-preview-label]');
  const sceneItems = [
    { src: 'assets/media/scene-generation/scene-01.jpg', labelKey: 'scene.caption1', label: 'Scene 01' },
    { src: 'assets/media/scene-generation/scene-02.jpg', labelKey: 'scene.caption2', label: 'Scene 02' },
    { src: 'assets/media/scene-generation/scene-03.jpg', labelKey: 'scene.caption3', label: 'Scene 03' },
    { src: 'assets/media/scene-generation/scene-04.jpg', labelKey: 'scene.caption4', label: 'Scene 04' },
  ];
  let activeSceneIndex = 0;
  const setScene = (index) => {
    if (!scenePreview) return;
    activeSceneIndex = (index + sceneItems.length) % sceneItems.length;
    const item = sceneItems[activeSceneIndex];
    const dict = I18N[currentLang] || I18N.en;
    const label = dict[item.labelKey] || item.label;
    scenePreview.src = item.src;
    scenePreview.alt = label;
    if (scenePreviewLabel) {
      scenePreviewLabel.dataset.i18n = item.labelKey;
      scenePreviewLabel.textContent = label;
    }
  };
  refreshSceneLabel = () => {
    if (!scenePreviewLabel) return;
    const item = sceneItems[activeSceneIndex];
    const dict = I18N[currentLang] || I18N.en;
    scenePreviewLabel.dataset.i18n = item.labelKey;
    scenePreviewLabel.textContent = dict[item.labelKey] || item.label;
  };
  $$('[data-scene-step]').forEach(btn => {
    btn.addEventListener('click', () => {
      const step = Number(btn.dataset.sceneStep) || 1;
      setScene(activeSceneIndex + step);
    });
  });

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
