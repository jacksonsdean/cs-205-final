// Core structure of the CPPN: Nodes and Connections
import { ACTIVATION_FUNCTIONS } from './activation.js';

// Generate a random weight between -1 and 1 as a tensor
function generateRandomWeight() {
    const w = tf.randomUniform([1], -1, 1);
    return w;
  }

function getRandomActivationFunction() {
    const keys = Object.keys(ACTIVATION_FUNCTIONS);
    const randomKey = keys[Math.floor(Math.random() * keys.length)];
    return ACTIVATION_FUNCTIONS[randomKey];
  }

/**
 * Represents a single node in the CPPN.
 * Nodes can act as input, hidden, or output nodes.
 */

class Node {
  constructor(id, activationFunction = null) {
    this.id = id; // Unique identifier
    this.activationFunction = activationFunction; // Activation function
    this.inputTensor = null; // Tensor of inputs
    this.outputTensor = null; // Tensor of outputs
  }

  // Activates the node by applying the activation function to the input tensor
  activate() {
    if (this.inputTensor) {
      if (this.activationFunction) {
        this.outputTensor = this.activationFunction(this.inputTensor); // Apply activation
      } else {
        this.outputTensor = this.inputTensor; // Pass-through if no activation
      }
    }
    // Reset the input tensor after activation to avoid reuse
    this.inputTensor = null;
  }
}

class Connection {
    constructor(fromNode, toNode, weight = 0, enabled = true) {
      this.fromNode = fromNode; // Source node
      this.toNode = toNode; // Destination node
      this.weight = weight; // Weight of the connection
      this.enabled = enabled; // Whether the connection is active
    }
  
    // Propagate the tensor from 'fromNode' to 'toNode', applying the weight
    propagate() {
      if (this.enabled) {
        const weightedTensor = this.fromNode.outputTensor.mul(this.weight); // Weighted tensor
        if (this.toNode.inputTensor) {
          this.toNode.inputTensor = this.toNode.inputTensor.add(weightedTensor); // Accumulate inputs
        } else {
          this.toNode.inputTensor = weightedTensor; // Initialize input tensor
        }
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
  function initializeCPPN(numInputs, hiddenLayerSizes, numOutputs) {
    const layers = [];
    const connections = [];
  
    // Create input layer
    const inputNodes = Array(numInputs)
      .fill(null)
      .map((_, i) => new Node(`input_${i}`));
    layers.push(inputNodes);
  
    // Create hidden layers
    hiddenLayerSizes.forEach((size, layerIndex) => {
      const hiddenLayer = Array(size)
        .fill(null)
        .map((_, i) => new Node(`hidden_${layerIndex}_${i}`, getRandomActivationFunction()));
      layers.push(hiddenLayer);
    });
  
    // Create output layer
    const outputNodes = Array(numOutputs)
      .fill(null)
      .map((_, i) => new Node(`output_${i}`, tf.sigmoid));
    layers.push(outputNodes);
  
   // Connect layers
for (let i = 0; i < layers.length - 1; i++) {
    const currentLayer = layers[i];
    const nextLayer = layers[i + 1];
  
    currentLayer.forEach((fromNode) => {
      nextLayer.forEach((toNode) => {
        const connection = new Connection(fromNode, toNode, generateRandomWeight());
        connections.push(connection); // Push connection as an object
      });
    });
  }
  
    return { layers, connections };
  }
  
  /**
 * Evaluates the CPPN by propagating inputs through the network.
 * @param {Object} cppn - The initialized CPPN (nodes and connections).
 * @param {number[]} inputValues - Array of input values for the input nodes.
 * @returns {number[]} - Array of output values from the output nodes.
 */

export function evaluate(cppn, inputTensors) {
    const { layers, connections } = cppn;
  
    // Set input tensors in the first layer
    layers[0].forEach((node, i) => {
      node.outputTensor = inputTensors[i] || tf.zerosLike(inputTensors[0]);
    });
  
    // Propagate through the network
    for (let i = 1; i < layers.length; i++) {
      const layer = layers[i];
      layer.forEach((node) => {
        // Collect inputs from incoming connections
        const incomingConnections = connections.filter((conn) => conn.toNode === node);
        incomingConnections.forEach((conn) => conn.propagate());
  
        // Activate the node
        node.activate();
      });
    }
    
    let outputs = layers[layers.length - 1].map((node) => node.outputTensor);
    outputs = outputs.map((output) => tf.sub(tf.onesLike(output), tf.abs(output)));
    // Collect output tensors from the last layer
    return outputs;
  }
  
  
  // Export classes and initialization function
  export { Node, Connection, initializeCPPN };
  