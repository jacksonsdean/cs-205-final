// Random Weight Generation for CPPN Connections

/**
 * Generates a random floating-point number within a specified range.
 * @param {number} min - The minimum value (inclusive).
 * @param {number} max - The maximum value (inclusive).
 * @returns {number} - A random number between min and max.
 */
function randomInRange(min, max) {
    return Math.random() * (max - min) + min;
  }
  
  /**
   * Generates a random initial weight for a connection.
   * The range of weights can be adjusted to control the scale of initial patterns.
   * @returns {number} - A random weight between -1 and 1.
   */
  function generateRandomWeight() {
    return randomInRange(-1, 1);
  }
  
  /**
   * Mutates a given weight by applying a small random adjustment.
   * The mutation rate controls how drastically weights can change.
   * @param {number} weight - The current weight.
   * @param {number} mutationRate - The standard deviation of the mutation.
   * @returns {number} - The mutated weight.
   */
  function mutateWeight(weight, mutationRate = 0.1) {
    const mutation = randomInRange(-mutationRate, mutationRate);
    return weight + mutation;
  }
  
  /**
   * Generates an array of random weights for initializing a CPPN network.
   * @param {number} numWeights - The number of weights to generate.
   * @returns {number[]} - An array of random weights.
   */
  function generateRandomWeightArray(numWeights) {
    const weights = [];
    for (let i = 0; i < numWeights; i++) {
      weights.push(generateRandomWeight());
    }
    return weights;
  }
  
  // Export functions for use in the CPPN
  export { randomInRange, generateRandomWeight, mutateWeight, generateRandomWeightArray };
  