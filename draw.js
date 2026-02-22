const graphviz = require('graphviz');
const fs = require('fs');

function drawDot(root) {
  const outputDir = 'output';
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir);
  }

  const g = graphviz.digraph('G');
  
  g.set('rankdir', 'LR');

  const nodes = new Set();
  const edges = new Set();

  function build(value) {
    if (nodes.has(value)) return;
    nodes.add(value);

    value.parents.forEach(parent => {
      edges.add([parent, value]);
      build(parent);
    });
  }

  build(root);

  nodes.forEach(value => {
    const nodeId = String(value.id);

    g.addNode(nodeId, {
      label: `{ ${value.label ? value.label + ' | ' : ''}data: ${value.data.toFixed(4)} | grad: ${value.grad.toFixed(4)} }`,
      shape: 'record'
    });

    if (value.op) {
      const opNodeId = nodeId + value.op;
      g.addNode(opNodeId, { label: value.op });
      g.addEdge(opNodeId, nodeId);
    }
  });

  edges.forEach(([parent, child]) => {
    const childOpNodeId = String(child.id) + child.op;
    g.addEdge(String(parent.id), childOpNodeId);
  });

  g.render('png', `${outputDir}/graph.png`);
  console.log(`Graph saved to ${outputDir}/graph.png`);
}

module.exports = drawDot;
