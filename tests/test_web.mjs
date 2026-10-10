import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { latestSeason, prepareSeason, evaluate, evaluatePuzzle, appearance, pairKey, nextAssumption, attemptAssumption, orderNights, resetNight } from '../web/knowledge.js';

function toy(size = 3, events = [], multi = []) {
  return {
    initial_active: 'all', matrix: { rows: 'a', bits: 'b' },
    groups: { a: Array.from({ length: size }, (_, i) => `A${i}`), b: Array.from({ length: size }, (_, i) => `B${i}`) },
    initial_match_model: { multi_matches: multi }, style: {}, target_lights: size,
    weeks: [{ number: 1, events }],
  };
}
const diagonal = size => Array.from({ length: size }, (_, i) => [`A${i}`, `B${i}`]);
const night = (size, lights, extras = {}) => ({ type: 'matching_night', pairs: diagonal(size), lights, ...extras });
const box = (a, b, result, extras = {}) => ({ type: 'box', pair: [a, b], result, ...extras });
const assumption = (value, nightId = 'mn-1') => ({ value, nightId });
const set = (...entries) => new Map(entries);
const state = (result, a, b) => result.states.get(pairKey(a, b));
const seasonData = JSON.parse(readFileSync(new URL('../ayto_data.json', import.meta.url), 'utf8'));

test('loads the bottom season in JSON order, without guessing its number', () => {
  assert.equal(latestSeason({ s99: {}, s6vip: { name: 'test' } }).key, 's6vip');
  const latest = latestSeason(seasonData);
  assert.equal(latest.key, 's6vip');
  assert.equal(prepareSeason(latest.season, latest.key).label, '6 VIP');
});

test('real latest-season baseline agrees with the Villa weekly counts', () => {
  const { key, season } = latestSeason(seasonData);
  const model = prepareSeason(season, key), result = evaluate(model);
  assert.equal(model.rowNames.length, 11);
  assert.equal(model.bitNames.length, 11);
  assert.deepEqual(result.nights.map(row => [row.remaining, row.openCount]), [[2, 6], [1, 7], [2, 5], [2, 8], [1, 7], [3, 8], [3, 8], [4, 8]]);
  assert.equal(result.conflicts.length, 0);
  assert.equal(result.found, 2);
  assert.equal(result.solved, false);
  assert.deepEqual(model.nights[3].seatings.get('Johannes'), ['Marta', 'Janice']);
  assert.equal(model.nights[0].seatings.has('Laurenz'), false);
});

test('4/10 updates to 4/9 on No Match and 3/9 on Perfect Match', () => {
  const model = prepareSeason(toy(10, [night(10, 4)]));
  assert.deepEqual(evaluate(model).nights.map(row => [row.remaining, row.openCount]), [[4, 10]]);
  assert.deepEqual(evaluate(model, set([pairKey('A0', 'B0'), assumption('no')])).nights.map(row => [row.remaining, row.openCount]), [[4, 9]]);
  assert.deepEqual(evaluate(model, set([pairKey('A0', 'B0'), assumption('yes')])).nights.map(row => [row.remaining, row.openCount]), [[3, 9]]);
});

test('public Matchbox facts are immutable and retain confirmed colors', () => {
  const model = prepareSeason(toy(3, [box('A0', 'B0', 'no')]));
  const result = evaluate(model, set([pairKey('A0', 'B0'), assumption('yes')]));
  assert.equal(state(result, 'A0', 'B0').value, 'no');
  assert.equal(appearance(model, state(result, 'A0', 'B0')), 'confirmed-no');
  assert.ok(result.conflicts.length);
});

test('a partner lock changes earlier nights even if the new match was never seated', () => {
  const model = prepareSeason(toy(3, [night(3, 1)]));
  const result = evaluate(model, set([pairKey('A0', 'B1'), assumption('yes', null)]));
  assert.equal(state(result, 'A0', 'B0').value, 'no');
  assert.equal(state(result, 'A1', 'B1').value, 'no');
  assert.equal(state(result, 'A2', 'B2').value, 'yes');
  assert.equal(appearance(model, state(result, 'A2', 'B2')), 'assumed-yes');
  assert.equal(result.nights[0].remaining, 0);
  assert.equal(result.nights[0].openCount, 0);
});

test('fully resolved nights propagate through a deduction chain', () => {
  const model = prepareSeason(toy(3, [
    night(3, 1), box('A0', 'B0', 'yes'),
    { type: 'matching_night', pairs: [['A1', 'B2'], ['A2', 'B1']], lights: 2, automatic_lights: 0 },
  ]));
  const result = evaluate(model);
  assert.equal(state(result, 'A1', 'B1').value, 'no');
  assert.equal(state(result, 'A1', 'B2').value, 'yes');
  assert.equal(result.conflicts.length, 0);
  assert.equal(result.solved, true);
  assert.ok(result.nights.every(row => row.remaining === 0 && row.openCount === 0));
});

test('automatic lights are subtracted once; sold events reveal nothing', () => {
  const model = prepareSeason(toy(3, [
    box('A0', 'B0', 'yes'), box('A1', 'B1', 'sold'),
    { type: 'matching_night', pairs: [['A1', 'B1'], ['A2', 'B2']], lights: 2, automatic_lights: 1 },
    { type: 'matching_night', pairs: [['A1', 'B2'], ['A2', 'B1']], result: 'sold' },
  ]));
  const result = evaluate(model);
  assert.equal(result.nights[0].remaining, 1);
  assert.equal(result.nights[0].openCount, 2);
  assert.equal(result.nights[1].remaining, null);
  assert.equal(state(result, 'A1', 'B1'), undefined);
});

test('too many positive assumptions highlight the night and identify their roots', () => {
  const model = prepareSeason(toy(3, [night(3, 1)]));
  const assumptions = set([pairKey('A0', 'B0'), assumption('yes')], [pairKey('A1', 'B1'), assumption('yes')]);
  const result = evaluate(model, assumptions);
  const issue = result.conflicts.find(issue => issue.type === 'night');
  assert.ok(issue);
  assert.ok(issue.nightIds.has('mn-1'));
  assert.equal(issue.dependencies.size, 2);
  assert.equal(result.solved, false);
  assumptions.delete(pairKey('A1', 'B1'));
  assert.equal(evaluate(model, assumptions).conflicts.length, 0);
});

test('too many No Matches are also a conflict', () => {
  const model = prepareSeason(toy(3, [night(3, 2)]));
  const result = evaluate(model, set([pairKey('A0', 'B0'), assumption('no')], [pairKey('A1', 'B1'), assumption('no')]));
  assert.ok(result.conflicts.some(issue => issue.type === 'night'));
  assert.equal(result.nights[0].remaining, 2);
  assert.equal(result.nights[0].openCount, 1);
});

test('removing hypotheses rebuilds everything and does not retain inferred results', () => {
  const model = prepareSeason(toy(3, [night(3, 1)]));
  const inferred = evaluate(model, set([pairKey('A0', 'B0'), assumption('yes')]));
  assert.equal(state(inferred, 'A1', 'B1').value, 'no');
  const cleared = evaluate(model);
  assert.equal(state(cleared, 'A1', 'B1'), undefined);
  assert.deepEqual([cleared.nights[0].remaining, cleared.nights[0].openCount], [1, 3]);
});

test('nights stay chronological regardless of edits or input order', () => {
  const nights = [3, 1, 4, 2].map(number => ({ id: `mn-${number}`, number, week: number }));
  assert.deepEqual(orderNights(nights).map(row => row.number), [1, 2, 3, 4]);
  assert.deepEqual(nights.map(row => row.number), [3, 1, 4, 2]);
  const sameWeek = [{ number: 3, week: 2 }, { number: 2, week: 2 }, { number: 1, week: 1 }];
  assert.deepEqual(orderNights(sameWeek).map(row => row.number), [1, 2, 3]);
});

test('row reset removes only assumptions actually entered in that row', () => {
  const assumptions = set(
    [pairKey('A0', 'B0'), assumption('yes', 'mn-3')],
    [pairKey('A1', 'B1'), assumption('yes', 'mn-1')],
    [pairKey('A2', 'B2'), assumption('yes', 'mn-4')],
  );
  const reset = resetNight(assumptions, 'mn-3');
  assert.equal(reset.has(pairKey('A0', 'B0')), false);
  assert.equal(reset.has(pairKey('A1', 'B1')), true);
  assert.equal(reset.has(pairKey('A2', 'B2')), true);
  assert.equal(assumptions.size, 3);
});

test('cycling is open -> yes -> open', () => {
  assert.equal(nextAssumption(undefined), 'yes');
  assert.equal(nextAssumption('yes'), null);
  assert.equal(nextAssumption(null), 'yes');
});

test('an incomplete confirmed double match leaves room for a second partner', () => {
  const data = toy(2);
  data.groups.b.push('B2');
  data.initial_match_model.multi_matches = [{ group: 'bits', size: 2, known_members: [] }];
  data.weeks[0].events = [box('A0', 'B0', 'yes', { complete_pair: false })];
  const model = prepareSeason(data);
  assert.equal(model.baseline.has(pairKey('A0', 'B1')), false);
  const result = evaluate(model, set([pairKey('A0', 'B1'), assumption('yes')], [pairKey('A1', 'B2'), assumption('yes')]));
  assert.equal(result.conflicts.length, 0);
  assert.equal(result.found, 2);
  assert.equal(result.solved, true);
});

test('a revealed complete triple is one match group, not two', () => {
  const data = toy(3);
  data.groups.b.push('B3');
  data.initial_match_model.multi_matches = [{ group: 'bits', size: 2 }];
  data.weeks[0].events = [box('A0', 'B0', 'yes', { revealed_group: { partner: 'A0', members: ['B0', 'B1'], complete: true } })];
  const model = prepareSeason(data), result = evaluate(model);
  assert.equal(result.found, 1);
  assert.equal(state(result, 'A0', 'B2').value, 'no');
  assert.equal(model.boxes[0].people.length, 3);
});

test('extra doubles and wrong known-members cannot produce a false win', () => {
  const data = toy(3);
  data.initial_match_model.multi_matches = [{ group: 'rows', size: 2, known_members: ['A2'] }];
  const model = prepareSeason(data);
  const result = evaluate(model, set([pairKey('A0', 'B0'), assumption('yes')], [pairKey('A1', 'B0'), assumption('yes')]));
  assert.ok(result.conflicts.some(issue => issue.type === 'topology'));
  assert.equal(result.solved, false);
});

test('both double-match directions count once and a complete valid graph wins', () => {
  const data = toy(3);
  data.target_lights = 2;
  data.initial_match_model.multi_matches = [{ group: 'bits', size: 2 }, { group: 'rows', size: 2, known_members: ['A2'] }];
  const model = prepareSeason(data);
  const result = evaluate(model, set(
    [pairKey('A0', 'B0'), assumption('yes')], [pairKey('A0', 'B1'), assumption('yes')],
    [pairKey('A1', 'B2'), assumption('yes')], [pairKey('A2', 'B2'), assumption('yes')],
  ));
  assert.equal(result.conflicts.length, 0);
  assert.equal(result.found, 2);
  assert.equal(result.solved, true);
});

test('finding the target number of groups without matching everyone is not a win', () => {
  const data = toy(3); data.target_lights = 2;
  const result = evaluate(prepareSeason(data), set([pairKey('A0', 'B0'), assumption('yes')], [pairKey('A1', 'B1'), assumption('yes')]));
  assert.equal(result.found, 2);
  assert.equal(result.solved, false);
});

test('static entry uses relative assets and includes the requested Coffee widget', () => {
  const html = readFileSync(new URL('../index.html', import.meta.url), 'utf8');
  assert.ok(html.includes('Kannst du die Staffel lösen?'));
  assert.ok(html.includes('data-id="aytorakel"'));
  assert.ok(html.includes('https://cdnjs.buymeacoffee.com/1.0.0/widget.prod.min.js'));
  assert.ok(html.includes('Fotos © RTL+'));
  assert.ok(html.includes('./web/app.js'));
  assert.ok(!html.includes('src="/'));
});


test('a valid night sharing a pair is not highlighted just because another night conflicts', () => {
  const model = prepareSeason(toy(3, [night(3, 1), { type: 'matching_night', pairs: [['A0', 'B0'], ['A1', 'B2'], ['A2', 'B1']], lights: 1 }]));
  const result = evaluate(model, set([pairKey('A0', 'B0'), assumption('yes')], [pairKey('A1', 'B1'), assumption('yes')]));
  const issues = result.conflicts.filter(issue => issue.type === 'night');
  assert.ok(issues.length);
  assert.ok(issues.every(issue => !issue.nightIds.has('mn-2')));
});

test('partner exclusions retain the hypotheses that used the available double slot', () => {
  const data = toy(3); data.groups.b.push('B3');
  data.initial_match_model.multi_matches = [{ group: 'bits', size: 2 }];
  const model = prepareSeason(data);
  const keys = [pairKey('A0', 'B0'), pairKey('A0', 'B1'), pairKey('A1', 'B2')];
  const result = evaluate(model, new Map(keys.map(key => [key, assumption('yes')])));
  const excluded = state(result, 'A1', 'B3');
  assert.equal(excluded.value, 'no');
  assert.ok(keys.every(key => excluded.dependencies.has(key)));
});


test('a partial larger multi match can grow before the final topology is checked', () => {
  const data = toy(2);
  data.groups.b.push('B2', 'B3');
  data.initial_match_model.multi_matches = [{ group: 'bits', size: 3, known_members: ['B2'] }];
  const model = prepareSeason(data);
  const assumptions = set([pairKey('A0', 'B0'), assumption('yes')], [pairKey('A0', 'B1'), assumption('yes')]);
  assert.equal(evaluate(model, assumptions).conflicts.length, 0);
  assumptions.set(pairKey('A0', 'B2'), assumption('yes'));
  assumptions.set(pairKey('A1', 'B3'), assumption('yes'));
  const result = evaluate(model, assumptions);
  assert.equal(result.conflicts.length, 0);
  assert.equal(result.solved, true);
  assert.equal(result.found, 2);
});


test('the last unseated pair completes the puzzle and clears when its source is reset', () => {
  const model = prepareSeason(toy(3, [box('A0', 'B0', 'yes')]));
  const key = pairKey('A1', 'B1');
  const own = set([key, assumption('yes')]);
  const result = evaluate(model, own);
  assert.equal(state(result, 'A2', 'B2').value, 'yes');
  assert.ok(state(result, 'A2', 'B2').dependencies.has(key));
  assert.equal(result.solved, true);
  const reset = resetNight(own, 'mn-1');
  assert.equal(state(evaluate(model, reset), 'A2', 'B2'), undefined);
  assert.equal(evaluate(model, reset).solved, false);
});


test('a contradictory assumption is rejected without changing previous inputs or deductions', () => {
  const model = prepareSeason(toy(3, [night(3, 1)]));
  const original = set([pairKey('A0', 'B0'), assumption('yes')]);
  const before = evaluate(model, original);
  const refusedKey = pairKey('A1', 'B1');
  const attempt = attemptAssumption(model, original, refusedKey, 'mn-1');
  assert.equal(attempt.accepted, false);
  assert.equal(attempt.assumptions, original);
  assert.equal(original.size, 1);
  assert.equal(original.has(refusedKey), false);
  assert.ok(attempt.result.conflicts.some(issue => issue.type === 'night'));
  const after = evaluate(model, original);
  assert.deepEqual(after.states, before.states);
  assert.deepEqual(after.nights.map(row => [row.remaining, row.openCount]), [[0, 0]]);
  assert.equal(after.conflicts.length, 0);
});

test('valid assumption attempts toggle on and off using copies of the original inputs', () => {
  const model = prepareSeason(toy(4, [night(4, 1)]));
  const original = new Map(), key = pairKey('A0', 'B0');
  const added = attemptAssumption(model, original, key, 'mn-1');
  assert.equal(added.accepted, true);
  assert.equal(original.size, 0);
  assert.equal(added.assumptions.get(key).nightId, 'mn-1');
  const removed = attemptAssumption(model, added.assumptions, key, 'mn-1');
  assert.equal(removed.accepted, true);
  assert.equal(removed.assumptions.size, 0);
  assert.equal(added.assumptions.size, 1);
  assert.deepEqual(removed.result.nights.map(row => [row.remaining, row.openCount]), [[1, 4]]);
});


test('conflict explanations show the final night counts once after a deduction chain', () => {
  const model = prepareSeason(toy(4, [
    night(4, 1),
    { type: 'matching_night', pairs: [['A0', 'B1'], ['A2', 'B2'], ['A3', 'B3']], lights: 2 },
  ]));
  const result = evaluate(model, set(
    [pairKey('A0', 'B0'), assumption('yes')],
    [pairKey('A1', 'B1'), assumption('yes')],
  ));
  const issues = result.conflicts.filter(issue => issue.type === 'night' && issue.nightIds.has('mn-1'));
  assert.equal(result.nights[0].remaining, -3);
  assert.equal(issues.length, 1);
  assert.match(issues[0].message, /3 Matches zu viel/);
});


test('medium and hard accept a contradictory move while easy keeps previous inputs', () => {
  const model = prepareSeason(toy(3, [night(3, 1)]));
  const own = set([pairKey('A0', 'B0'), assumption('yes')]);
  const key = pairKey('A1', 'B1');
  for (const difficulty of ['medium', 'hard']) {
    const attempt = attemptAssumption(model, own, key, 'mn-1', difficulty);
    assert.equal(attempt.accepted, true);
    assert.equal(attempt.assumptions.has(key), true);
    assert.ok(attempt.result.conflicts.length);
    assert.equal(attempt.result.solved, false);
  }
  assert.equal(attemptAssumption(model, own, key, 'mn-1', 'easy').accepted, false);
  assert.equal(own.size, 1);
});

test('switching back to easy still allows removing one of several conflicting assumptions', () => {
  const model = prepareSeason(toy(4, [night(4, 1)]));
  const keys = diagonal(3).map(pair => pairKey(...pair));
  const own = new Map(keys.map(key => [key, assumption('yes')]));
  const removed = attemptAssumption(model, own, keys[0], 'mn-1', 'easy');
  assert.equal(removed.accepted, true);
  assert.equal(removed.assumptions.size, 2);
  assert.ok(removed.result.conflicts.length);
  assert.equal(own.size, 3);
});

test('hard preserves public facts and partner locks without deductions from assumed night totals', () => {
  const model = prepareSeason(toy(3, [night(3, 1)]));
  const own = set([pairKey('A0', 'B0'), assumption('yes')]);
  const full = evaluate(model, own);
  const hard = evaluate(model, own, { inferNights: false, inferLastPartner: false });
  assert.equal(state(full, 'A1', 'B1').value, 'no');
  assert.equal(state(hard, 'A1', 'B1'), undefined);
  assert.equal(state(hard, 'A0', 'B0').value, 'yes');
  assert.equal(state(hard, 'A0', 'B1').value, 'no');
  const publicModel = prepareSeason(toy(3, [box('A0', 'B0', 'yes'), box('A1', 'B1', 'no')]));
  const publicHard = evaluate(publicModel, new Map(), { inferNights: false, inferLastPartner: false });
  for (const [key, fact] of publicModel.baseline) assert.equal(publicHard.states.get(key).value, fact.value);
});


test('hard waits for manually allocated night lights before revealing the complete solution', () => {
  const own = set([pairKey('A0', 'B0'), assumption('yes')]);
  const data = toy(3, [
    night(3, 1),
    { type: 'matching_night', pairs: [['A0', 'B1'], ['A1', 'B2'], ['A2', 'B1']], lights: 2 },
  ]);
  const puzzle = prepareSeason(data);
  const medium = evaluatePuzzle(puzzle, own, 'medium');
  const hard = evaluatePuzzle(puzzle, own, 'hard');
  assert.equal(medium.solved, true);
  assert.equal(hard.result.solved, true);
  assert.equal(hard.solved, false);
  assert.equal(hard.view.nights[1].remaining, 2);
  own.set(pairKey('A1', 'B2'), assumption('yes', 'mn-2'));
  own.set(pairKey('A2', 'B1'), assumption('yes', 'mn-2'));
  assert.equal(evaluatePuzzle(puzzle, own, 'hard').solved, true);
});
