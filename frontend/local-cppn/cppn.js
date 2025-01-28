// Core structure of the CPPN: Nodes and Connections

// Generate a random weight between -1 and 1
function generateRandomWeight() {
    return Math.random() * 2 - 1;
  }
  

/**
 * Represents a single node in the CPPN.
 * Nodes can act as input, hidden, or output nodes.
 */
class Node {
    constructor(id, activationFunction = null) {
      this.id = id; // Unique identifier for the node
      this.activationFunction = activationFunction; // Activation function for the node
      this.inputSum = 0; // Sum of weighted inputs
      this.outputValue = 0; // Output value after applying activation
    }
  
    /**
     * Computes the output value of the node by applying its activation function
     * to the sum of weighted inputs.
     */
    activate() {
      if (this.activationFunction) {
        this.outputValue = this.activationFunction(this.inputSum);
      } else {
        this.outputValue = this.inputSum; // For input nodes, pass through the value
      }
      // Reset input sum after activation for next computation cycle
      this.inputSum = 0;
    }
  }
  
  /**
   * Represents a connection between two nodes in the CPPN.
   */
  class Connection {
    constructor(fromNode, toNode, weight = 0, enabled = true) {
      this.fromNode = fromNode; // Source node
      this.toNode = toNode; // Destination node
      this.weight = weight; // Weight of the connection
      this.enabled = enabled; // Whether the connection is active
    }
  
    /**
     * Propagates the output of the 'fromNode' to the 'toNode'.
     * The input value is multiplied by the connection weight.
     */
    propagate() {
      if (this.enabled) {
        this.toNode.inputSum += this.fromNode.outputValue * this.weight;
      }
    }
  }
  
  /**
   * Initializes a network of nodes with random connections.
   * @param {number} numInputNodes - Number of input nodes.
   * @param {number} numOutputNodes - Number of output nodes.
   * @param {function[]} activationFunctions - Array of activation functions.
   * @returns {Object} - An object containing the nodes and connections.
   */
  function initializeCPPN(numInputNodes, numOutputNodes, activationFunctions) {
    const nodes = [];
    const connections = [];
  
    // Create input nodes (activation function is null for input nodes)
    for (let i = 0; i < numInputNodes; i++) {
      nodes.push(new Node(`input_${i}`));
    }
  
    // Create output nodes with random activation functions
    for (let i = 0; i < numOutputNodes; i++) {
      const activationFunction =
        activationFunctions[Math.floor(Math.random() * activationFunctions.length)];
      nodes.push(new Node(`output_${i}`, activationFunction));
    }
  
    // Create random connections between nodes
    for (let i = 0; i < nodes.length; i++) {
      for (let j = 0; j < nodes.length; j++) {
        if (i !== j) {
          const weight = generateRandomWeight();
          connections.push(new Connection(nodes[i], nodes[j], weight));
        }
      }
    }

    
  
    return { nodes, connections };
  }
  
  // Export classes and initialization function
  export { Node, Connection, initializeCPPN };
  