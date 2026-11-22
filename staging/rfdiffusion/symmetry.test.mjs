import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import { expandSymmetry } from './symmetry.mjs'

const fixture = JSON.parse(readFileSync(new URL('./symmetry-reference.json', import.meta.url)))
const unit = Float32Array.from(fixture.unit)
const tolerance = 1e-6

function near(actual, expected) {
  const error = Math.abs(actual - expected)
  assert.ok(error <= tolerance, `Expected ${expected}, received ${actual}`)
  return error
}

for (const [group, copies] of Object.entries(fixture.expected)) {
  test(`${group} matches published SymGen coordinates and subunit order`, (t) => {
    const result = expandSymmetry(group, unit)
    const expected = copies.flat()
    assert.equal(result.length, expected.length)
    let maxError = 0
    for (let i = 0; i < result.length; i++) {
      maxError = Math.max(maxError, near(result[i], expected[i]))
    }
    assert.deepEqual(Array.from(unit), fixture.unit)
    t.diagnostic(`${copies.length} copies, max absolute coordinate error ${maxError}`)
  })
}

test('each copy preserves distances and the orientation of a coordinate frame', () => {
  const basis = Float32Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1)
  for (const group of ['C2', 'C7', 'D3', 'D8']) {
    const frames = expandSymmetry(group, basis)
    const points = expandSymmetry(group, unit)
    for (let copy = 0; copy < frames.length / 9; copy++) {
      const [a, b, c, d, e, f, g, h, i] = frames.subarray(copy * 9, copy * 9 + 9)
      near(a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g), 1)
      const xyz = points.subarray(copy * unit.length, (copy + 1) * unit.length)
      for (let left = 0; left < unit.length; left += 3) {
        for (let right = left + 3; right < unit.length; right += 3) {
          near(distance(xyz, left, right), distance(unit, left, right))
        }
      }
    }
  }
})

function distance(xyz, left, right) {
  return Math.hypot(xyz[left] - xyz[right], xyz[left + 1] - xyz[right + 1], xyz[left + 2] - xyz[right + 2])
}

test('rejects malformed orders, unsupported groups and incomplete coordinates', () => {
  for (const group of ['c0', 'd-2', 'c1.5', 'tetrahedral', 'D', 'd99999999999999999999']) {
    assert.throws(() => expandSymmetry(group, unit), RangeError)
  }
  for (const xyz of [[], new Float32Array(), Float32Array.of(1, 2)]) {
    assert.throws(() => expandSymmetry('c2', xyz), TypeError)
  }
  assert.throws(() => expandSymmetry('c2', Float32Array.of(1, 2, NaN)), RangeError)
})
