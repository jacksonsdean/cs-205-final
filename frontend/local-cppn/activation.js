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

export const ACTIVATION_FUNCTIONS = {
    sigmoid: (x) => 1 / (1 + Math.exp(-x)),
    tanh: (x) => Math.tanh(x),
    relu: (x) => Math.max(0, x),
    gaussian: (x) => Math.exp(-x * x),
    sine: (x) => Math.sin(x),
    linear: (x) => x,
  };
  