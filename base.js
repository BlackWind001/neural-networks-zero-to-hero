const drawDot = require('./draw.js');
const { topologicalSort } = require('./sort.js');

function getRandomValue () {
  return (Math.random() * 2) - 1;
}

function calculateMSE (groundTruths /** i.e. the expected values */, predictions /* i.e. output values */) {
  let loss = new Value(0);
  for (let i = 0; i < groundTruths.length; i++) {
    const {value: groundTruth} = Value.getOther(groundTruths[i]);
    const {value: prediction} = Value.getOther(predictions[i]);
    const currentLoss= groundTruth.sub(prediction).pow(2);
    
    console.log(groundTruth.data, prediction.data, currentLoss.data);
    
    loss = loss.add(currentLoss);
  }
  return loss;
}

class Value {
  static _idCounter = 0;
  
  data = 0
  grad = 0
  label = ''
  op = ''
  parents = []
  backward = () => {}
  
  constructor (data, parents = [], op = '') {
    this.id = Value._idCounter++;
    this.data = data;
    this.op = op;
    this.parents = parents;
  }
  
  static getOther (input) {
    if (typeof input === 'number') {
      return { data: input, value: new Value(input) };
    }
    return { data: input.data, value: input };
  }
  
  mul (input) {
    let { data: otherData, value: other } = Value.getOther(input);
    const output = new Value(this.data * otherData, [this, other], '*')
    
    output.backward = () => {
      this.grad += otherData*output.grad;
      other.grad += this.data*output.grad;
    }
    
    return output;
  }
  
  add (input) {
    let { data: otherData, value: other } = Value.getOther(input);
    const output = new Value(this.data + otherData, [this, other], '+')
    
    output.backward = () => {
      this.grad += 1*output.grad;
      other.grad += 1*output.grad;
    }
    
    return output;
  }
  
  sub (input) {
    let { data: otherData, value: other } = Value.getOther(input);
    const output = new Value(this.data - otherData, [this, other], '-')
    
    output.backward = () => {
      this.grad += 1*output.grad;
      other.grad += 1*output.grad;
    }
    
    return output;
  }
  
  pow (input) {
    let { data: otherData, value: other } = Value.getOther(input);
    const output = new Value(Math.pow(this.data, otherData), [this, other], '**')
    
    output.backward = () => {
      this.grad += otherData * Math.pow(this.data, otherData-1)*output.grad;
    }
    
    return output;
  }
  
  tanh () {
    const { data } = this;
    const common = Math.exp(2*data);
    const t = (common - 1)/(common + 1);
    const output = new Value(t, [this], 'tanh')
    
    output.backward = () => {
      this.grad += (1-Math.pow(t, 2))*output.grad;
    }
    
    return output;
  }
  
  backwardPass () {
    const sorted = topologicalSort(this).reverse();
    
    this.grad = 1;
    for (const entry of sorted) {
      entry.backward();
    }
  }
}

class Neuron {
  static weightCounter = 0;
  static biasCounter = 0;
  weights = [];
  bias;
  constructor (noOfInputs) {
    this.weights = Array.from({ length: noOfInputs }, () => {
      const weight = new Value(getRandomValue());
      weight.label = "w" + Neuron.weightCounter++;
      return weight;
    });
    
    this.bias = new Value(getRandomValue());
    this.bias.label = "b" + Neuron.biasCounter++;
  }
  
  exec (inputs) {
    
    return this.weights.reduce((sum, currentWeight, index) => {
      const currentProd = currentWeight.mul(inputs[index]);
      if (sum) {
        return sum.add(currentProd);
      }
      return currentProd;
    }, undefined).add(this.bias).tanh();
  }
}

class Layer {
  neurons = [];
  constructor (noOfInputs, noOfOutputs) {
    this.neurons = Array.from({ length: noOfOutputs }, () => {
      return new Neuron(noOfInputs);
    });
  }
  
  // Gives an array of outputs from each neuron in the layer
  exec (inputs) {
    const neuronOutput = this.neurons.map((neuron) => neuron.exec(inputs));
    
    return neuronOutput.length === 1 ? neuronOutput[0] : neuronOutput
  }
}

class MLP {
  layers;
  constructor (noOfInputs, layersOutputSizeList) {
    let currentNoOfInputs = noOfInputs
    this.layers = layersOutputSizeList.map((layerOutputSize, index) => {
      const layer = new Layer(currentNoOfInputs, layerOutputSize);
      currentNoOfInputs = layerOutputSize;
      
      return layer;
    });
  }
  
  exec (input) {
    return this.layers.reduce((acc, currentLayer, index) => {
      return currentLayer.exec(acc);
    }, input);
  }
}

// const inputs = [2.0, 3.0, -1.0]
// const mlp = new MLP(inputs.length, [4, 4, 1]);
// const value = mlp.exec(inputs);
// console.log(value)
// drawDot(value.length > 1 ? value[0] : value)

module.exports = { Value, Neuron, Layer, MLP, calculateMSE }
