const { MLP, Value, calculateMSE } = require('./base');
const drawDot = require('./draw.js');

const xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
],
ys = [
  (() => { let y = new Value(1.0); y.label = 'ys1'; return y })(),
  (() => { let y = new Value(-1.0); y.label = 'ys2'; return y })(),
  (() => { let y = new Value(-1.0); y.label = 'ys3'; return y })(),
  (() => { let y = new Value(1.0); y.label = 'ys4'; return y })(),
] // desired targets

const mlp2 = new MLP(xs[0].length, [4, 4, 1]);
const ypred = []

for (const xi of xs) {
  ypred.push(mlp2.exec(xi));
}
console.log('Predictions', ypred.map((yi) => yi.data));

//Calculate the loss - using MSE
const loss = calculateMSE(ys, ypred);

loss.backwardPass();
console.log('Loss', loss);
drawDot(loss);
