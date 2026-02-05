/**
 * inference.js
 * Node script to run inference with the TFJS model in this repo.
 *
 * Usage:
 *   1) Install deps:
 *        npm install @tensorflow/tfjs-node
 *   2) Run with a CSV of 15 values:
 *        node inference.js 0.12,0.34,0.56,...  (15 comma-separated numbers)
 *   3) Or run with no args to use a random sample.
 */

const fs = require('fs');
const path = require('path');
const tf = require('@tensorflow/tfjs-node');

const MODEL_PATH = path.join(__dirname, 'model.json');
const SCALER_PATH = path.join(__dirname, 'scaler.json');

async function loadModel() {
  // tf.loadLayersModel supports file:// for node
  const model = await tf.loadLayersModel('file://' + MODEL_PATH);
  return model;
}

function loadScaler() {
  const raw = fs.readFileSync(SCALER_PATH, 'utf8');
  const scaler = JSON.parse(raw);
  if (!scaler.mean || !scaler.scale) throw new Error('scaler.json missing mean/scale');
  return scaler;
}

// rawArray should be length 15 (model input length)
function applyScaler(rawArray, scaler) {
  const mean = scaler.mean;
  const scale = scaler.scale;
  if (rawArray.length !== mean.length) {
    throw new Error(`Expected input length ${mean.length} but got ${rawArray.length}`);
  }
  // avoid divide-by-zero: if scale[i] === 0 use 1
  return rawArray.map((v, i) => {
    const s = scale[i] === 0 ? 1 : scale[i];
    return (v - mean[i]) / s;
  });
}

async function predictFromArray(rawArray) {
  const model = await loadModel();
  const scaler = loadScaler();

  const scaled = applyScaler(rawArray, scaler);

  // Model expects shape [batch, 15, 1]
  const inputTensor = tf.tensor(scaled, [1, scaled.length, 1], 'float32');

  const out = model.predict(inputTensor);
  // out may be a tensor or array of tensors
  let result;
  if (Array.isArray(out)) {
    result = await Promise.all(out.map(t => t.array()));
  } else {
    result = await out.array();
  }

  tf.dispose([inputTensor, out]);
  return result;
}

function parseCsvArg(arg) {
  return arg.split(',').map(s => {
    const n = Number(s.trim());
    if (Number.isNaN(n)) throw new Error(`Invalid number in input: "${s}"`);
    return n;
  });
}

async function main() {
  try {
    let raw;
    if (process.argv.length >= 3) {
      raw = parseCsvArg(process.argv[2]);
    } else {
      // If no args, produce a random sample centered near scaler.mean
      const scaler = loadScaler();
      raw = scaler.mean.map((m, i) => m + (Math.random() - 0.5) * (scaler.scale[i] || 1));
      console.log('No input provided — using a random sample near the scaler mean.');
    }

    if (raw.length !== 15) {
      console.error(`Input length is ${raw.length} but model requires 15 features.`);
      process.exit(2);
    }

    const pred = await predictFromArray(raw);
    console.log('Raw input:', raw);
    console.log('Prediction result:', JSON.stringify(pred, null, 2));
  } catch (err) {
    console.error('Error:', err.message);
    process.exit(1);
  }
}

if (require.main === module) main();