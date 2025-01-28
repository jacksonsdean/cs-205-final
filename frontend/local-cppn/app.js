// Initialize settings
const WIDTH = 128;
const HEIGHT = 128;
const NUM_IMAGES = 6; // Number of candidate images per generation

// Container to display images
const container = document.getElementById("container");
const evolveButton = document.getElementById("evolve");

// Helper: Create a canvas for an image
function createCanvas(data) {
  const canvas = document.createElement("canvas");
  canvas.width = WIDTH;
  canvas.height = HEIGHT;
  const ctx = canvas.getContext("2d");
  const imageData = new ImageData(new Uint8ClampedArray(data), WIDTH, HEIGHT);
  ctx.putImageData(imageData, 0, 0);
  canvas.addEventListener("click", () => {
    canvas.classList.toggle("selected");
  });
  container.appendChild(canvas);
}

// Generate random CPPN weights
function randomWeights(size) {
  return tf.randomUniform([size], -1, 1).arraySync();
}

// CPPN-like activation functions
function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}
function tanh(x) {
  return Math.tanh(x);
}

// Generate image using CPPN
function generateImage(weights) {
  const data = [];
  for (let y = 0; y < HEIGHT; y++) {
    for (let x = 0; x < WIDTH; x++) {
      const nx = (x / WIDTH) * 2 - 1;
      const ny = (y / HEIGHT) * 2 - 1;
      const r = Math.sqrt(nx * nx + ny * ny);

      // CPPN formula (can be customized)
      const value = tanh(
        weights[0] * nx +
        weights[1] * ny +
        weights[2] * r +
        weights[3] * sigmoid(nx * weights[4] + ny * weights[5])
      );
      const intensity = Math.floor((value + 1) * 127.5);
      data.push(intensity, intensity, intensity, 255); // RGBA
    }
  }
  return data;
}

// Generate initial population
let population = [];
for (let i = 0; i < NUM_IMAGES; i++) {
  const weights = randomWeights(6);
  const image = generateImage(weights);
  createCanvas(image);
  population.push({ weights, image });
}

// Evolve selected images
evolveButton.addEventListener("click", () => {
  const selected = Array.from(container.querySelectorAll(".selected"));
  if (selected.length === 0) return alert("Select at least one image!");

  const selectedWeights = selected.map((_, i) => population[i].weights);

  // Generate next generation by mutating selected weights
  container.innerHTML = ""; // Clear current images
  const newPopulation = [];
  for (let i = 0; i < NUM_IMAGES; i++) {
    const parentWeights =
      selectedWeights[Math.floor(Math.random() * selectedWeights.length)];
    const mutatedWeights = parentWeights.map((w) =>
      w + Math.random() * 0.2 - 0.1
    );
    const newImage = generateImage(mutatedWeights);
    createCanvas(newImage);
    newPopulation.push({ weights: mutatedWeights, image: newImage });
  }
  population = newPopulation;
});
