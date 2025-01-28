/**
 * Evaluates the CPPN by propagating inputs through the network.
 * @param {Object} cppn - The initialized CPPN (nodes and connections).
 * @param {number[]} inputValues - Array of input values for the input nodes.
 * @returns {number[]} - Array of output values from the output nodes.
 */
function evaluateCPPN(cppn, inputValues) {
    const { nodes, connections } = cppn;
  
    // Step 1: Assign input values to input nodes
    const inputNodes = nodes.filter(node => node.id.startsWith('input'));
    if (inputNodes.length !== inputValues.length) {
      throw new Error('Number of inputs does not match number of input nodes.');
    }
    inputNodes.forEach((node, index) => {
      node.outputValue = inputValues[index]; // Set input node values directly
    });
  
    // Step 2: Reset all other node states
    nodes.forEach(node => {
      if (!node.id.startsWith('input')) {
        node.inputSum = 0; // Reset input sum for next propagation
        node.outputValue = 0; // Clear any lingering output values
      }
    });
  
    // Step 3: Propagate values through connections
    connections.forEach(connection => {
      connection.propagate();
    });
  
    // Step 4: Activate all non-input nodes
    nodes.forEach(node => {
      if (!node.id.startsWith('input')) {
        node.activate();
      }
    });
  
    // Step 5: Collect output values
    const outputNodes = nodes.filter(node => node.id.startsWith('output'));
    return outputNodes.map(node => node.outputValue);
  }
  
  // Example usage of evaluateCPPN
  export { evaluateCPPN };
  