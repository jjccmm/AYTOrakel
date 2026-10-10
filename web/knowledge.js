// Public facts and reversible hypotheses. No solver checkpoint or solution is read.
export const pairKey = (a, b) => JSON.stringify([a, b]);
const compareNames = (a, b) => a.localeCompare(b, 'de', { sensitivity: 'base' });
const union = (...sets) => new Set(sets.flatMap(set => [...set]));

export function latestSeason(data) {
  const entries = Object.entries(data);
  if (!entries.length) throw new Error('Es gibt noch keine Staffel.');
  const [key, season] = entries.at(-1);
  return { key, season };
}

export function seasonLabel(key, data) {
  if (data.name || data.title) return data.name || data.title;
  const match = /^s(\d+)(vip)?$/i.exec(key);
  return match ? `${match[1]}${match[2] ? ' VIP' : ''}` : key;
}

function flatten(data) {
  return [...data.weeks].sort((a, b) => Number(a.number) - Number(b.number))
    .flatMap(week => week.events.map(event => ({ week: Number(week.number), event })));
}

function compile(data, history) {
  const rowGroup = data.matrix.rows;
  const bitGroup = data.matrix.bits;
  const groups = Object.fromEntries(Object.entries(data.groups).map(([group, names]) => [group, [...names]]));
  const active = Object.fromEntries(Object.entries(groups).map(([group, names]) => [
    group, new Set(data.initial_active === 'all' ? names : data.initial_active[group]),
  ]));
  let rules = (data.initial_match_model?.multi_matches || []).map(rule => ({ ...rule }));
  let arrivalExclusions = [];
  const limits = new Map(Object.values(groups).flat().map(name => [name, 1]));
  const applyRules = additions => {
    for (const rule of additions) {
      const partnerGroup = rule.group === 'bits' ? rowGroup : bitGroup;
      for (const name of groups[partnerGroup]) limits.set(name, Math.max(limits.get(name) || 1, Number(rule.size)));
    }
  };
  applyRules(rules);
  for (const { event } of history) {
    if (event.type !== 'cast_change') continue;
    for (const { group, person } of event.add || []) {
      if (!groups[group].includes(person)) groups[group].push(person);
      active[group].add(person);
      limits.set(person, 1);
    }
    for (const person of event.remove || []) for (const names of Object.values(active)) names.delete(person);
    const update = event.match_update || {};
    if (update.mode === 'rebuild') {
      rules = (update.multi_matches || []).map(rule => ({ ...rule }));
      for (const name of Object.values(groups).flat()) limits.set(name, 1);
      arrivalExclusions = [];
      applyRules(rules);
    } else if (update.mode === 'extend') {
      const addition = event.add[0];
      const other = addition.group === rowGroup ? bitGroup : rowGroup;
      const eligible = new Set(update.eligible_partners || active[other]);
      rules.push({ ...update.new_multi_match });
      for (const person of eligible) limits.set(person, Math.max(limits.get(person) || 1, Number(update.new_multi_match.size)));
      for (const person of groups[other]) if (!eligible.has(person)) arrivalExclusions.push([addition.person, person]);
    }
  }
  const rowNames = [...groups[rowGroup]].sort(compareNames);
  const bitNames = [...groups[bitGroup]].sort(compareNames);
  const rowSet = new Set(rowNames), bitSet = new Set(bitNames);
  const canonical = pair => {
    const [a, b] = pair;
    if (rowSet.has(a) && bitSet.has(b)) return [a, b];
    if (rowSet.has(b) && bitSet.has(a)) return [b, a];
    throw new Error(`Ungültiges Paar: ${a} / ${b}`);
  };
  const seeds = [], closed = new Map(), nights = [], boxes = [];
  const seed = (rawPair, value, source) => {
    const [a, b] = canonical(rawPair);
    seeds.push({ key: pairKey(a, b), a, b, value, source });
  };
  const revealGroup = group => {
    for (const member of group.members) seed([group.partner, member], 'yes', 'reveal');
    if (group.complete !== false) closed.set(group.partner, new Set(group.members));
  };
  for (const [index, { week, event }] of history.entries()) {
    if (event.type === 'box') {
      const [a, b] = canonical(event.pair);
      if (event.result !== 'sold') {
        seed([a, b], event.result, 'box');
        if (event.result === 'yes' && (event.complete_pair ?? !event.revealed_group)) {
          closed.set(a, new Set([b]));
          closed.set(b, new Set([a]));
        }
      }
      if (event.revealed_group) revealGroup(event.revealed_group);
      const group = event.revealed_group;
      const people = group ? [group.members[0], group.partner, ...group.members.slice(1)] : [a, b];
      for (const person of event.pair) if (!people.includes(person)) people.push(person);
      boxes.push({ number: boxes.length + 1, week, people, result: event.result });
    } else if (['reveal', 'solution'].includes(event.type)) {
      const partners = new Map();
      for (const rawPair of event.pairs || []) {
        seed(rawPair, 'yes', event.type);
        const [a, b] = canonical(rawPair);
        for (const [person, partner] of [[a, b], [b, a]]) {
          if (!partners.has(person)) partners.set(person, new Set());
          partners.get(person).add(partner);
        }
      }
      if (event.type === 'solution') for (const [person, names] of partners) closed.set(person, names);
      for (const group of event.complete_groups || []) revealGroup(group);
    } else if (event.type === 'matching_night') {
      const pairs = event.pairs.map(canonical);
      nights.push({
        id: `mn-${nights.length + 1}`, number: nights.length + 1, week, eventIndex: index,
        pairs, keys: pairs.map(([a, b]) => pairKey(a, b)),
        lights: event.result === 'sold' ? null : Number(event.lights),
        automaticLights: Number(event.automatic_lights || 0),
      });
    }
  }
  for (const pair of arrivalExclusions) seed(pair, 'no', 'partners');
  return { rowNames, bitNames, rowSet, bitSet, canonical, limits, rules, seeds, closed, nights, boxes,
    targetLights: Number(data.target_lights || 10), data };
}

function positiveComponents(model, states) {
  const neighbors = new Map(), edgeKeys = new Map();
  for (const state of states.values()) {
    if (state.value !== 'yes') continue;
    for (const [person, partner] of [[state.a, state.b], [state.b, state.a]]) {
      if (!neighbors.has(person)) neighbors.set(person, new Set());
      neighbors.get(person).add(partner);
      if (!edgeKeys.has(person)) edgeKeys.set(person, new Set());
      edgeKeys.get(person).add(state.key);
    }
  }
  const seen = new Set(), components = [];
  for (const person of neighbors.keys()) {
    if (seen.has(person)) continue;
    const stack = [person], people = [], keys = new Set();
    while (stack.length) {
      const name = stack.pop();
      if (seen.has(name)) continue;
      seen.add(name);
      people.push(name);
      for (const key of edgeKeys.get(name)) keys.add(key);
      for (const partner of neighbors.get(name)) if (!seen.has(partner)) stack.push(partner);
    }
    components.push({
      rows: people.filter(name => model.rowSet.has(name)).sort(compareNames),
      bits: people.filter(name => model.bitSet.has(name)).sort(compareNames), keys,
    });
  }
  return { neighbors, components };
}

function assignMultiRules(model, components, complete = false) {
  const multi = components.filter(group => group.rows.length > 1 || group.bits.length > 1);
  const eligible = group => model.rules.flatMap((rule, index) => {
    const members = rule.group === 'rows' ? group.rows : group.bits;
    const partners = rule.group === 'rows' ? group.bits : group.rows;
    const known = rule.known_members || [];
    const sizeFits = complete
      ? members.length === Number(rule.size) && known.every(name => members.includes(name))
      : members.length <= Number(rule.size) && new Set([...members, ...known]).size <= Number(rule.size);
    return partners.length === 1 && sizeFits ? [index] : [];
  });
  const choices = multi.map(eligible);
  const result = new Map();
  function assign(index) {
    if (index === multi.length) return true;
    for (const rule of choices[index]) {
      if (result.has(rule)) continue;
      result.set(rule, multi[index]);
      if (assign(index + 1)) return true;
      result.delete(rule);
    }
    return false;
  }
  return assign(0) ? result : null;
}

function derive(model, assumptions = new Map(), baseline = null, options = {}) {
  const states = new Map(), conflicts = new Map();
  const proof = keys => union(...[...keys].map(key => states.get(key)?.dependencies || new Set()));
  const conflict = (type, message, keys = [], dependencies = new Set(), nightId = null) => {
    const signature = type === 'night' ? `night:${nightId}` : `${type}:${message}`;
    const previous = conflicts.get(signature);
    if (previous) {
      previous.message = message;
      previous.keys = union(previous.keys, new Set(keys));
      previous.dependencies = union(previous.dependencies, dependencies);
      if (nightId) previous.nightIds.add(nightId);
    } else {
      conflicts.set(signature, { type, message, keys: new Set(keys), dependencies: new Set(dependencies), nightIds: new Set(nightId ? [nightId] : []) });
    }
  };
  const mark = (a, b, value, source, dependencies = new Set()) => {
    const key = pairKey(a, b), previous = states.get(key);
    if (previous) {
      if (previous.value !== value) conflict('pair', `${a} & ${b}: Match und No Match widersprechen sich.`, [key], union(previous.dependencies, dependencies));
      return false;
    }
    states.set(key, { key, a, b, value, source, dependencies: new Set(dependencies) });
    return true;
  };
  for (const seed of model.seeds) mark(seed.a, seed.b, seed.value, seed.source);
  if (baseline) for (const state of baseline.values()) mark(state.a, state.b, state.value, state.source);
  for (const [key, assumption] of assumptions) {
    const [a, b] = JSON.parse(key);
    if (!model.rowSet.has(a) || !model.bitSet.has(b) || !['yes', 'no'].includes(assumption.value)) continue;
    mark(a, b, assumption.value, 'assumption', new Set([key]));
  }
  const reduceNights = () => model.nights.map(night => {
    const yes = night.keys.filter(key => states.get(key)?.value === 'yes');
    const openKeys = night.keys.filter(key => !states.has(key));
    const remaining = night.lights === null ? null : night.lights - night.automaticLights - yes.length;
    if (remaining !== null && (remaining < 0 || remaining > openKeys.length)) {
      conflict('night', remaining < 0
        ? `MN ${night.number}: ${Math.abs(remaining)} ${Math.abs(remaining) === 1 ? 'Match zu viel' : 'Matches zu viel'} für die gemeldeten Lichter.`
        : `MN ${night.number}: ${remaining} Lichter brauchen noch ein Match, aber nur ${openKeys.length} Paare sind offen.`,
      night.keys, proof(night.keys), night.id);
    }
    return { ...night, remaining, openKeys, openCount: openKeys.length, confirmedCount: yes.length };
  });
  let changed = true;
  while (changed) {
    changed = false;
    const { neighbors, components } = positiveComponents(model, states);
    const assignment = assignMultiRules(model, components);
    if (!assignment) {
      const keys = union(...components.filter(group => group.rows.length > 1 || group.bits.length > 1).map(group => group.keys));
      conflict('topology', 'Diese Matches passen nicht zur erlaubten Doppelmatch-Konstellation.', keys, proof(keys));
    }
    for (const person of [...model.rowNames, ...model.bitNames]) {
      const partners = neighbors.get(person) || new Set();
      const keys = [...partners].map(partner => pairKey(...model.canonical([person, partner])));
      let capacity = model.limits.get(person) || 1;
      let capacityProof = proof(keys);
      if (assignment && capacity > 1 && partners.size <= 1) {
        const memberSide = model.rowSet.has(person) ? 'bits' : 'rows';
        if (!model.rules.some((rule, index) => rule.group === memberSide && !assignment.has(index))) {
          capacity = 1;
          capacityProof = union(capacityProof, ...[...assignment].filter(([index]) => model.rules[index].group === memberSide).map(([, group]) => proof(group.keys)));
        }
      }
      const closed = model.closed.get(person);
      if (partners.size > (closed?.size ?? capacity) || (closed && [...partners].some(partner => !closed.has(partner)))) {
        conflict('capacity', `${person} kann diese Matches nicht gleichzeitig haben.`, keys, proof(keys));
      }
      if (closed || (partners.size > 0 && partners.size === capacity)) {
        const allowed = closed || partners;
        const otherNames = model.rowSet.has(person) ? model.bitNames : model.rowNames;
        for (const partner of otherNames) {
          if (allowed.has(partner)) continue;
          const [a, b] = model.canonical([person, partner]);
          changed = mark(a, b, 'no', 'partners', capacityProof) || changed;
        }
      }
      const possibleKeys = (model.rowSet.has(person) ? model.bitNames : model.rowNames)
        .map(partner => pairKey(...model.canonical([person, partner])));
      if (possibleKeys.every(key => states.get(key)?.value === 'no')) {
        conflict('unmatched', `Für ${person} bleibt kein möglicher Partner übrig.`, possibleKeys, proof(possibleKeys));
      }
      // An unmatched person with one remaining candidate must match that person,
      // even if the pair never sat together and cannot be selected in a night.
      if (options.inferLastPartner !== false && !possibleKeys.some(key => states.get(key)?.value === 'yes')) {
        const open = possibleKeys.filter(key => !states.has(key));
        if (open.length === 1) {
          const [a, b] = JSON.parse(open[0]);
          changed = mark(a, b, 'yes', 'last_partner', proof(possibleKeys)) || changed;
        }
      }
    }
    const nights = reduceNights();
    if (options.inferNights !== false) for (const night of nights) {
      if (night.remaining === null || !night.openCount || night.remaining < 0 || night.remaining > night.openCount) continue;
      if (night.remaining === 0 || night.remaining === night.openCount) {
        const dependencies = proof(night.keys);
        for (const key of night.openKeys) {
          const [a, b] = JSON.parse(key);
          changed = mark(a, b, night.remaining === 0 ? 'no' : 'yes', 'matching_night', dependencies) || changed;
        }
      }
    }
  }
  const nights = reduceNights();
  for (const issue of conflicts.values()) {
    if (issue.type !== 'night') {
      for (const night of nights) if (night.keys.some(key => issue.keys.has(key))) issue.nightIds.add(night.id);
    }
    for (const key of issue.dependencies) {
      const sourceNight = assumptions.get(key)?.nightId;
      if (!issue.nightIds.size && sourceNight) issue.nightIds.add(sourceNight);
    }
  }
  const { components } = positiveComponents(model, states);
  const covered = new Set(components.flatMap(group => [...group.rows, ...group.bits]));
  const finalAssignment = assignMultiRules(model, components, true);
  const solved = !conflicts.size && covered.size === model.rowNames.length + model.bitNames.length &&
    components.length === model.targetLights && finalAssignment !== null && finalAssignment.size === model.rules.length &&
    nights.every(night => night.remaining === null || night.remaining === 0);
  return { states, nights, conflicts: [...conflicts.values()], components, solved,
    found: Math.min(model.targetLights, components.length), target: model.targetLights };
}

export function prepareSeason(data, key = '') {
  const events = flatten(data);
  const model = compile(data, events);
  model.key = key;
  model.label = seasonLabel(key, data);
  model.week = events.at(-1)?.week || 0;
  const baseline = derive(model);
  if (baseline.conflicts.length) throw new Error(`Die Staffeldaten widersprechen sich: ${baseline.conflicts[0].message}`);
  model.baseline = baseline.states;
  for (const night of model.nights) {
    const before = derive(compile(data, events.slice(0, night.eventIndex)));
    const seatings = new Map();
    const sit = (a, b) => {
      if (!seatings.has(a)) seatings.set(a, []);
      if (!seatings.get(a).includes(b)) seatings.get(a).push(b);
    };
    for (const state of before.states.values()) if (state.value === 'yes') sit(state.a, state.b);
    for (const [a, b] of night.pairs) sit(a, b);
    night.seatings = seatings;
  }
  return model;
}

export function evaluate(model, assumptions = new Map(), options = {}) {
  return derive(model, assumptions, model.baseline, options);
}

// Hard requires the user to account for every reported light before revealing a solution.
export function evaluatePuzzle(model, assumptions = new Map(), difficulty = 'medium') {
  const result = evaluate(model, assumptions);
  let view = difficulty === 'hard' ? evaluate(model, assumptions, { inferNights: false, inferLastPartner: false }) : result;
  const solved = result.solved && (difficulty !== 'hard' || view.nights.every(night => night.remaining === null || night.remaining === 0));
  if (solved) view = result;
  return { result, view, solved };
}

export function appearance(model, state) {
  if (!state) return 'unknown';
  return `${model.baseline.has(state.key) ? 'confirmed' : 'assumed'}-${state.value}`;
}

export function nextAssumption(current) {
  return current === 'yes' ? null : 'yes';
}

// Evaluate on a copy. Easy blocks new contradictions; removing inputs always stays possible.
export function attemptAssumption(model, assumptions, key, nightId, difficulty = 'easy') {
  const candidate = new Map(assumptions);
  const value = nextAssumption(assumptions.get(key)?.value);
  if (value) candidate.set(key, { value, nightId });
  else candidate.delete(key);
  const result = evaluate(model, candidate);
  const accepted = difficulty === 'medium' || difficulty === 'hard' || value === null || result.conflicts.length === 0;
  return { accepted, assumptions: accepted ? candidate : assumptions, result, value };
}

export function orderNights(nights) {
  return [...nights].sort((a, b) => (a.week ?? a.number) - (b.week ?? b.number) || a.number - b.number);
}

export function resetNight(assumptions, nightId) {
  return new Map([...assumptions].filter(([, value]) => value.nightId !== nightId));
}
