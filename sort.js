function topologicalSort (node, visited = new Set(), topo = []) {
  if (visited.has(node)) {
    return;
  }
  
  visited.add(node);
  node.parents.forEach((parent) => {
    topologicalSort(parent, visited, topo);
  })
  topo.push(node);
  return topo;
}

module.exports = { topologicalSort };
