const { Value, MLP, calculateMSE } = require('./base');
const drawDot = require('./draw.js');
const { topologicalSort } = require('./sort.js');

function basicTest () {
  // inputs x1,x2
  const x1 = new Value(2.0); x1.label='x1'
  const x2 = new Value(0.0); x2.label='x2'
  // weights w1,w2
  const w1 = new Value(-3.0); w1.label='w1'
  const w2 = new Value(1.0); w2.label='w2'
  // bias of the neuron
  const b = new Value(6.8813735870195432); b.label='b'
  // x1*w1 + x2*w2 + b
  const x1w1 = x1.mul(w1); x1w1.label = 'x1*w1'
  const x2w2 = x2.mul(w2); x2w2.label = 'x2*w2'
  const x1w1x2w2 = x1w1.add(x2w2); x1w1x2w2.label = 'x1*w1 + x2*w2'
  const n = x1w1x2w2.add(b); n.label = 'n'
  const o = n.tanh(); o.label = 'o'
  
  o.grad = 1;
  o.backwardPass();
  
  drawDot(o)
}

let result, loss;
function mlpTest () {
  const inputs = [2.0, 3.0, -1.0]
  const expectedOutput = 3.0
  const mlp = new MLP(inputs.length, [1]);
  
  function forwardPass () {
    result = mlp.exec(inputs);
    result.label = "Ypred";
    
    loss = calculateMSE([expectedOutput], [result]);
  }
  
  function descent () {
    const params = mlp.parameters();
    for (let param of params) {
      param.data += (-0.01)*param.grad;
    }
  }
  
  function train () {
    forwardPass();
    
    loss.backwardPass();
    // drawDot(loss);
    console.log({ result: result.data, loss: loss.data });
    
    descent();
  }
  
  return train;
}

function execute (f, n) {
  Array.from({ length: n }).forEach(() => {
    f();
  });
}
