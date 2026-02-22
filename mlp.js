const { MLP, calculateMSE } = require('./base');

const xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
],
ys = [1.0, -1.0, -1.0, 1.0] // desired targets

const mlp2 = new MLP(xs[0].length, [4, 4, 1]);
const ypred = []

for (const xi of xs) {
  ypred.push(mlp2.exec(xi));
}
console.log('Predictions', ypred.map((yi) => yi.data));

//Calculate the loss - using MSE
const loss = calculateMSE(ys, ypred);
console.log('Loss', loss);
