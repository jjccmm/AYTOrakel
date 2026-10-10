import { latestSeason, prepareSeason, evaluatePuzzle, appearance, pairKey, attemptAssumption, orderNights, resetNight } from './knowledge.js';

const $ = id => document.getElementById(id);
const reducedMotion = () => matchMedia('(prefers-reduced-motion: reduce)').matches;
let model, result, view, assumptions = new Map(), difficulty = 'medium', isSolved = false, winShown = false;
let conflictingNights = new Set();

function element(tag, attributes = {}, children = []) {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(attributes)) {
    if (value === null || value === undefined || value === false) continue;
    if (name.startsWith('on') && typeof value === 'function') node.addEventListener(name.slice(2), value);
    else if (name === 'class') node.className = value;
    else node.setAttribute(name, value === true ? '' : String(value));
  }
  for (const child of Array.isArray(children) ? children : [children]) {
    if (child !== null && child !== undefined) node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function face(name, className = '', decorative = false) {
  const directory = model.data.style?.villa_faces || `insta_styles/${model.key}_faces`;
  const url = new URL(`${directory.replace(/\/$/, '')}/${encodeURIComponent(name)}.png`, document.baseURI);
  const img = element('img', { src: url.href, alt: decorative ? '' : name, class: className, draggable: 'false', loading: 'lazy' });
  img.addEventListener('error', () => {
    img.replaceWith(element('span', { class: `initials-face ${className}`, 'aria-label': decorative ? null : name, 'aria-hidden': decorative ? 'true' : null }, name.slice(0, 2)));
  }, { once: true });
  return img;
}

const symbol = state => !state ? '·' : state.value === 'yes' ? '+' : '×';
const stateLabel = state => !state ? 'offen' : state.value === 'yes' ? 'Perfect Match' : 'No Match';

function pairButton(a, b, nightId) {
  const key = pairKey(a, b), state = view.states.get(key), fixed = model.baseline.has(key);
  const explicit = assumptions.has(key);
  const editable = !fixed && (!state || explicit);
  const certainty = fixed ? 'best\u00e4tigt' : state ? explicit ? 'deine Annahme' : 'aus deinen Annahmen abgeleitet' : 'noch offen';
  const label = `${a} & ${b}: ${stateLabel(state)} (${certainty}).${editable ? explicit ? ' Tippen: Annahme entfernen.' : ' Tippen: Perfect Match annehmen.' : ''}`;
  return element('button', {
    type: 'button', class: `pair-button ${appearance(model, state)}${explicit ? ' is-explicit' : ''}`,
    'data-pair-key': key, 'data-night-id': nightId, 'aria-label': label,
    'aria-pressed': explicit ? 'true' : 'false', title: label, disabled: !editable,
    onclick: () => cyclePair(key, nightId),
  }, [face(b, '', true), state ? element('span', { class: 'state-mark', 'aria-hidden': 'true' }, symbol(state)) : null]);
}

function disc(label, extraClass = '', large = false) {
  return element('span', { class: `disc ${large ? 'total' : 'week'}` },
    label === null ? [] : extraClass ? element('span', { class: extraClass }, label) : label);
}

function headerCell(name, withPartners = false) {
  const partners = withPartners ? model.bitNames.filter(partner => view.states.get(pairKey(name, partner))?.value === 'yes') : [];
  return element('th', { scope: 'col', 'data-person': name }, [
    element('div', { class: `header-faces${partners.length > 1 ? ' multi' : ''}` }, [face(name, 'column-face', true), ...partners.map(partner => face(partner, 'column-face', true))]),
    element('span', { class: 'column-name' }, name),
    partners.length ? element('span', { class: 'column-name partner-name' }, partners.join(' / ')) : null,
  ]);
}

function renderNights() {
  const table = $('mn-table');
  table.style.setProperty('--cast-count', model.rowNames.length);
  table.style.setProperty('--table-units', model.rowNames.length * 74 + 258);
  const heading = element('tr', { class: 'column-header' }, [
    element('th', { class: 'week-cell', scope: 'col', 'aria-label': 'Woche' }, [
      element('div', { class: 'header-faces' }, disc('W')),
      element('span', { class: 'column-name' }, 'Woche'),
    ]),
    ...model.rowNames.map(name => headerCell(name, true)),
    element('th', { class: 'lights-cell', scope: 'col', 'aria-label': 'Gemeldete Lichter' }, [
      element('div', { class: 'header-faces' }, element('span', { class: 'disc letter' }, element('span', { class: 'outlined yellow' }, 'L'))),
      element('span', { class: 'column-name' }, 'Lichter'),
    ]),
    element('th', { class: 'remaining-cell', scope: 'col', 'aria-label': 'Ungelöste Lichter / offene Paare' }, [
      element('div', { class: 'header-faces' }, element('span', { class: 'disc letter' }, element('span', { class: 'outlined red' }, '?'))),
      element('span', { class: 'column-name' }, 'offen'),
    ]),
    element('th', { class: 'reset-cell', scope: 'col' }, element('button', { type: 'button', class: 'row-reset reset-all', id: 'reset', title: 'Alle Annahmen zur\u00fccksetzen', 'aria-label': 'Alle Annahmen zur\u00fccksetzen', disabled: !assumptions.size, onclick: resetAll }, [
      element('div', { class: 'header-faces' }, '\u21ba'),
      element('span', { class: 'column-name' }, 'Reset'),
    ])),
  ]);
  const body = element('tbody');
  const ordered = orderNights(view.nights);
  for (const night of ordered) {
    const conflicting = difficulty !== 'hard' && conflictingNights.has(night.id);
    const row = element('tr', { class: `mn-row${conflicting ? ' conflicting' : ''}`, id: `row-${night.id}`, 'data-night-id': night.id });
    const week = element('td', { class: 'week-cell', title: `Matching Night ${night.number} \u00b7 Woche ${night.week}` }, disc(night.week));
    row.append(week);
    for (const person of model.rowNames) {
      const partners = night.seatings.get(person) || [];
      row.append(element('td', {}, partners.length ? element('div', { class: `seat-group${partners.length > 1 ? ' double' : ''}` },
        partners.map(partner => pairButton(person, partner, night.id))) : []));
    }
    const sold = night.lights === null;
    const hidden = difficulty === 'hard';
    const solved = !hidden && !sold && !conflicting && night.remaining === 0 && night.openCount === 0;
    row.append(element('td', { class: 'lights-cell' }, disc(sold ? '$' : String(night.lights), `outlined ${sold ? 'sold' : 'yellow'}`, true)));
    const remaining = night.remaining === null ? (night.openCount ? `$/${night.openCount}` : '$') : night.openCount ? `${night.remaining}/${night.openCount}` : String(night.remaining);
    const label = hidden ? 'Restzahlen im Modus Schwer ausgeblendet' : solved ? 'Nacht gel\u00f6st' : sold ? `Verkauft, ${night.openCount} offene Paare` : `${night.remaining} ungel\u00f6ste Lichter, ${night.openCount} offene Paare`;
    row.append(element('td', { class: 'remaining-cell', 'data-remaining': hidden ? '' : `${night.remaining}/${night.openCount}`, 'data-solved': String(solved), 'aria-label': label },
      element('span', { class: `capsule${solved ? ' solved' : ''}` }, hidden || solved ? [] : element('span', { class: `outlined ${sold ? 'sold' : 'red'}` }, remaining))));
    const ownCount = [...assumptions.values()].filter(assumption => assumption.nightId === night.id).length;
    row.append(element('td', { class: 'reset-cell' }, element('button', {
      type: 'button', class: 'row-reset', 'data-reset-night': night.id,
      title: `Nur die ${ownCount} Annahmen aus MN ${night.number} zurücksetzen`,
      'aria-label': `MN ${night.number} zurücksetzen: nur Annahmen aus dieser Zeile`, disabled: !ownCount,
      onclick: () => resetRow(night.id),
    }, '↺')));
    body.append(row);
  }
  const found = isSolved ? result.found : view.found;
  const progress = element('tr', { class: 'progress-row' }, element('th', { colspan: model.rowNames.length + 4 },
    element('div', { class: 'progress-track', role: 'progressbar', 'aria-label': 'Perfect Matches', 'aria-valuemin': 0, 'aria-valuemax': result.target, 'aria-valuenow': found },
      element('div', { id: 'progress-fill', style: `width: ${Math.min(100, found / result.target * 100)}%` }))));
  table.replaceChildren(element('caption', { class: 'sr-only' }, 'Matching Nights mit Lichtern und offenen Paaren'), element('thead', {}, [heading, progress]), body);
  $('night-count').textContent = `${model.nights.length} Nächte`;
}

function renderBoxes() {
  $('matchboxes').replaceChildren(...model.boxes.map(box => {
    const label = box.result === 'yes' ? 'Perfect Match' : box.result === 'no' ? 'No Match' : 'Verkauft';
    return element('article', { class: `mb-card ${box.result}` }, [
      element('span', { class: 'mb-week' }, `W${box.week}: ${label}`),
      element('div', { class: 'mb-faces' }, [...box.people.map(name => face(name)), element('span', { class: 'state-mark', 'aria-hidden': 'true' }, box.result === 'yes' ? '+' : box.result === 'no' ? '\u00d7' : '$')]),
    ]);
  }));
}

function renderGrid() {
  const table = $('match-grid');
  table.style.setProperty('--cast-count', model.rowNames.length);
  table.style.setProperty('--table-units', model.rowNames.length * 46 + 46);
  const heading = element('tr', {}, [element('th', { scope: 'col', 'aria-label': 'Frauen' }), ...model.rowNames.map(name => headerCell(name))]);
  const body = element('tbody', {}, model.bitNames.map(partner => element('tr', {}, [
    element('th', { scope: 'row' }, [face(partner, '', true), element('span', { class: 'grid-name' }, partner)]),
    ...model.rowNames.map(person => {
      const key = pairKey(person, partner), state = view.states.get(key);
      const status = state?.value === 'yes' ? 'match' : state?.value === 'no' ? 'impossible' : 'possible';
      return element('td', {}, element('div', {
        class: `grid-cell ${status}`, 'data-grid-pair': key,
        role: 'img', 'aria-label': `${person} & ${partner}: ${stateLabel(state)}`, title: `${person} & ${partner}: ${stateLabel(state)}`,
      }));
    }),
  ])));
  table.replaceChildren(element('caption', { class: 'sr-only' }, 'Alle Paare: nicht m\u00f6glich, m\u00f6glich oder Perfect Match'), element('thead', {}, heading), body);
}

function render(focus = null) {
  ({ result, view, solved: isSolved } = evaluatePuzzle(model, assumptions, difficulty));
  conflictingNights = new Set(result.conflicts.flatMap(issue => [...issue.nightIds]));
  $('game').dataset.difficulty = difficulty;
  renderNights();
  renderGrid();
  if (focus) {
    const region = $('mn-table');
    const button = [...region.querySelectorAll('[data-pair-key]')].find(node => node.dataset.pairKey === focus.key && node.dataset.nightId === (focus.nightId || ''));
    button?.focus({ preventScroll: true });
  }
  if (isSolved && !winShown) {
    winShown = true;
    showSolution();
  } else if (!isSolved) winShown = false;
}

function cyclePair(key, nightId) {
  if (model.baseline.has(key) || (view.states.has(key) && !assumptions.has(key))) return;
  const attempt = attemptAssumption(model, assumptions, key, nightId, difficulty);
  if (!attempt.accepted) {
    showConflict(key, attempt.result.conflicts);
    return;
  }
  assumptions = attempt.assumptions;
  render({ key, nightId });
  const [a, b] = JSON.parse(key);
  $('announcement').textContent = `${a} & ${b}: ${attempt.value === 'yes' ? 'Perfect Match angenommen' : 'Annahme entfernt'}.${difficulty === 'medium' && result.conflicts.length ? ' Es gibt einen Konflikt in den markierten N\u00e4chten.' : ''}`;
}

function resetRow(nightId) {
  assumptions = resetNight(assumptions, nightId);
  render();
  $('announcement').textContent = `Annahmen aus ${nightId.toUpperCase().replace('-', ' ')} zurückgesetzt.`;
}

function resetAll() {
  if (!assumptions.size) return;
  assumptions = new Map();
  $('solution-dialog').close();
  $('conflict-dialog').close();
  render();
  $('announcement').textContent = 'Alle Annahmen zur\u00fcckgesetzt.';
}

$('difficulty-controls').addEventListener('change', event => {
  if (event.target.name !== 'difficulty') return;
  difficulty = event.target.value;
  $('conflict-dialog').close();
  if (model) render();
  $('announcement').textContent = `Schwierigkeit: ${difficulty === 'easy' ? 'Leicht' : difficulty === 'medium' ? 'Mittel' : 'Schwer'}.`;
});

function showConflict(key, conflicts) {
  const [a, b] = JSON.parse(key);
  $('conflict-intro').textContent = `Deine Annahme zu ${a} & ${b} widerspricht den bisherigen Informationen:`;
  const messages = [...new Set(conflicts.map(issue => issue.message))];
  $('conflict-reasons').replaceChildren(...messages.slice(0, 4).map(message => element('li', {}, message)));
  $('conflict-dialog').showModal();
}
$('close-conflict').addEventListener('click', () => $('conflict-dialog').close());
$('dismiss-conflict').addEventListener('click', () => $('conflict-dialog').close());
$('conflict-dialog').addEventListener('click', event => {
  const box = event.currentTarget.getBoundingClientRect();
  if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) event.currentTarget.close();
});

function showSolution() {
  $('solution-list').replaceChildren(...result.components.map(group => {
    const people = group.rows.length > 1 ? [group.rows[0], group.bits[0], ...group.rows.slice(1)] : group.bits.length > 1 ? [group.bits[0], group.rows[0], ...group.bits.slice(1)] : [...group.rows, ...group.bits];
    return element('div', { class: 'solution-group' }, [
      element('div', { class: 'mb-faces' }, people.map(name => face(name))),
      element('span', { class: 'solution-names' }, people.join(' + ')),
    ]);
  }));
  $('solution-dialog').showModal();
  celebrate();
}
$('close-solution').addEventListener('click', () => $('solution-dialog').close());
$('back-to-game').addEventListener('click', () => $('solution-dialog').close());
$('solution-dialog').addEventListener('click', event => {
  const box = event.currentTarget.getBoundingClientRect();
  if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) event.currentTarget.close();
});

let confettiFrame;
function stopCelebration() {
  cancelAnimationFrame(confettiFrame);
  const canvas = $('confetti');
  if (canvas.matches(':popover-open')) canvas.hidePopover();
}
$('solution-dialog').addEventListener('close', stopCelebration);

function celebrate() {
  stopCelebration();
  if (reducedMotion()) return;
  const canvas = $('confetti'), ctx = canvas.getContext('2d');
  if (!ctx) return;
  const width = innerWidth, height = innerHeight, scale = Math.min(devicePixelRatio || 1, 2);
  canvas.width = width * scale; canvas.height = height * scale;
  ctx.scale(scale, scale);
  // Open after the modal so confetti is above both the dialog and its backdrop.
  canvas.showPopover();
  const colors = ['#008000', '#ffe54b', '#ff0000', '#8aafff', '#f5f3ee'];
  const particles = Array.from({ length: 100 }, () => ({ x: Math.random() * width, y: -Math.random() * height * .5, velocity: 2 + Math.random() * 4, angle: Math.random() * 6, drift: Math.random() * 2 - 1, size: 4 + Math.random() * 5, color: colors[Math.floor(Math.random() * colors.length)] }));
  const start = performance.now();
  let last = start;
  function frame(time) {
    const step = Math.min(3, (time - last) / 16.67); last = time;
    ctx.clearRect(0, 0, width, height);
    ctx.globalAlpha = Math.min(1, (3500 - (time - start)) / 800);
    for (const particle of particles) {
      particle.y += particle.velocity * step; particle.x += particle.drift * step; particle.angle += .025 * step;
      ctx.save(); ctx.translate(particle.x, particle.y); ctx.rotate(particle.angle); ctx.fillStyle = particle.color;
      ctx.fillRect(-particle.size / 2, -particle.size / 2, particle.size, particle.size * .55); ctx.restore();
    }
    if (time - start < 3500) confettiFrame = requestAnimationFrame(frame);
    else stopCelebration();
  }
  confettiFrame = requestAnimationFrame(frame);
}

async function load() {
  try {
    const response = await fetch(new URL('./ayto_data.json', document.baseURI), { cache: 'no-store' });
    if (!response.ok) throw new Error('Die Staffeldaten konnten nicht geladen werden.');
    const { key, season } = latestSeason(await response.json());
    model = prepareSeason(season, key);
    document.title = `AYTO Staffel ${model.label}`;
    $('page-title').textContent = `AYTO Staffel ${model.label}`;
    $('load-state').hidden = true;
    $('game').hidden = false;
    renderBoxes();
    render();
  } catch (error) {
    $('load-state').replaceChildren(
      element('p', {}, 'Die Staffel konnte gerade nicht geladen werden.'),
      element('p', {}, error.message),
      element('button', { type: 'button', class: 'button', onclick: () => location.reload() }, 'Erneut laden'),
    );
    console.error(error);
  }
}
load();
