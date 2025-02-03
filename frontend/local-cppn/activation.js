// Activation Functions for CPPN

/**
 * Sigmoid function.
 * Maps inputs to the range (0, 1).
 * @param {number} x - The input value.
 * @returns {number} - The output value after applying the sigmoid function.
 */
function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }
  
  /**
   * Tanh function.
   * Maps inputs to the range (-1, 1).
   * @param {number} x - The input value.
   * @returns {number} - The output value after applying the tanh function.
   */
  function tanh(x) {
    return Math.tanh(x);
  }
  
  /**
   * Sinusoidal function (sine wave).
   * Oscillates between -1 and 1.
   * @param {number} x - The input value.
   * @returns {number} - The output value after applying the sine function.
   */
  function sine(x) {
    return Math.sin(x);
  }
  
  /**
   * Linear function.
   * A passthrough activation with no transformation.
   * @param {number} x - The input value.
   * @returns {number} - The same input value (identity function).
   */
  function linear(x) {
    return x;
  }
  
  /**
   * Gaussian function.
   * Produces a bell curve around zero.
   * @param {number} x - The input value.
   * @returns {number} - The output value after applying the Gaussian function.
   */
  function gaussian(x) {
    return Math.exp(-Math.pow(x, 2));
  }
  
  // Export the activation functions for use in other modules
// activation.js


// using tensorflow.js
export const ACTIVATION_FUNCTIONS = {
    sigmoid: (x) => tf.sigmoid(x),
    tanh: (x) => tf.tanh(x),
    relu: (x) => tf.relu(x),
    round: (x) => tf.round(x),
    gaussian: (x) => tf.exp(tf.neg(tf.square(x))),
    sine: (x) => tf.sin(x),
    linear: (x) => x,
  };
  