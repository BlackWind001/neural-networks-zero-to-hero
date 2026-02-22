const { MLP, Value, calculateMSE } = require('./base');
const drawDot = require('./draw.js');

const xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
],

// Using IIFEs to attach labels to the expected outputs so I can
// easily spot the values on the graphviz output
ys = [
  (() => { let y = new Value(1.0); y.label = 'ys1'; return y })(),
  (() => { let y = new Value(-1.0); y.label = 'ys2'; return y })(),
  (() => { let y = new Value(-1.0); y.label = 'ys3'; return y })(),
  (() => { let y = new Value(1.0); y.label = 'ys4'; return y })(),
] // desired targets,
mlp = new MLP(xs[0].length, [4, 4, 1]);


let ypred = [], loss;

function forwardPass () {
  ypred = [];
  
  for (const xi of xs) {
    ypred.push(mlp.exec(xi));
  }
  // console.log('Predictions', ypred.map((yi) => yi.data));
  
  //Calculate the loss - using MSE
  loss = calculateMSE(ys, ypred);
}

function descent () {
  const parameters = mlp.parameters();
  parameters.forEach((param) => {
    param.data += (-0.05)*param.grad;
  });
}

function resetGrads () {
  const parameters = mlp.parameters();
  parameters.forEach((param) => {
    param.grad = 0;
  });
}

function train () {
  for (let i = 0; i < 20; i++) {
    forwardPass();
    resetGrads();
    loss.backwardPass(); // hydrates the gradient values.
    descent();
    console.log(`i=${i}`, `${loss}`);
  }
}
