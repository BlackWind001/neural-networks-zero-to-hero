const graphviz = require('graphviz');

module.exports = function drawDot(root) {
  const g = graphviz.digraph('G');
  
  g.set('rankdir', 'LR'); // Left to right layout, same as Karpathy's

  const nodes = new Set();
  const edges = new Set();

  // Topological traversal to collect all nodes and edges
  function build(value) {
    if (nodes.has(value)) return;
    nodes.add(value);

    value.parents.forEach(parent => {
      edges.add([parent, value]);
      build(parent);
    });
  }

  build(root);

  // For each value node, add a record node showing label, data, and grad
  nodes.forEach(value => {
    const nodeId = String(value.id);

    g.addNode(nodeId, {
      label: `{ ${value.label ? value.label + ' | ' : ''}data: ${value.data.toFixed(4)} | grad: ${value.grad.toFixed(4)} }`,
      shape: 'record'
    });

    // If this value was produced by an operation, add an op node
    if (value.op) {
      const opNodeId = nodeId + value.op;
      g.addNode(opNodeId, { label: value.op });
      g.addEdge(opNodeId, nodeId);
    }
  });

  // Add edges, pointing into the op node rather than directly to the value node
  edges.forEach(([parent, child]) => {
    const childOpNodeId = String(child.id) + child.op;
    g.addEdge(String(parent.id), childOpNodeId);
  });

  g.render('png', 'graph.png');
  console.log('Graph saved to graph.png');
}
