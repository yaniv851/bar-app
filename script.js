// Define your dataset
const dataset = [
  {
    input: "I am feeling happy today.",
    output: { happy: 1, sad: 0, angry: 0, suicidal: 0 },
  },
  {
    input: "This situation makes me sad.",
    output: { happy: 0, sad: 1, angry: 0, suicidal: 0 },
  },
  {
    input: "I'm really angry right now.",
    output: { happy: 0, sad: 0, angry: 1, suicidal: 0 },
  },
  {
    input: "I want to die.",
    output: { happy: 0, sad: 1, angry: 0, suicidal: 1 },
  },
  {
    input: "I hate my life",
    output: { happy: 0, sad: 1, angry: 0, suicidal: 1 },
  }
  // Add more examples with different characteristics
];

// Create a neural network
const net = new brain.NeuralNetwork();

// Preprocess text data
const vocabulary = new Set();
const maxVocabularySize = 50;

dataset.forEach((item) => {
  const words = item.input.toLowerCase().split(" ");
  words.forEach((word) => {
    vocabulary.add(word);
  });
});

// Convert vocabulary to an array and truncate if necessary
const vocabularyArray = Array.from(vocabulary).slice(0, maxVocabularySize);

// Create a word-to-index map
const wordToIndex = new Map(
  vocabularyArray.map((word, index) => [word, index])
);

// Convert input text to a numerical input vector
function textToInputVector(text) {
  const words = text.toLowerCase().split(" ");
  const inputVector = new Array(maxVocabularySize).fill(0);
  words.forEach((word) => {
    const index = wordToIndex.get(word);
    if (index !== undefined) {
      inputVector[index] = 1;
    }
  });
  return inputVector;
}

// Convert dataset for training
const processedDataset = dataset.map((item) => ({
  input: textToInputVector(item.input),
  output: item.output,
}));

// Train the network
net.train(processedDataset);

// New text input for prediction
const newText = "i hate everybody";
const inputVector = textToInputVector(newText);

// Make a prediction
const prediction = net.run(inputVector);

console.log("Prediction:", prediction);

// Find the output category with the highest value
let highestCategory = null;
let highestValue = -Infinity;

for (const category in prediction) {
  if (prediction[category] > highestValue) {
    highestValue = prediction[category];
    highestCategory = category;
  }
}

console.log("Highest category:", highestCategory);

const diagram = document.getElementById("diagram");
diagram.innerHTML = brain.utilities.toSVG(net);
