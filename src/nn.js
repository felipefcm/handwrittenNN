
const Matrix = require('./matrix');

class NeuralNetwork {

	/** @type number */
	numInputs;

	/** @type number[] */
	numHidden;

	/** @type Matrix[] */
	weights;

	/** @type Matrix[] */
	biases;

	/**
	 * 
	 * @param {number} numInputs Number of input neurons
	 * @param {number[]} numHidden Number of hidden layers and neurons. E.g. [3, 2, 3] for 3 hidden layers with 3, 2 and 3 neurons each
	 * @param {number} numOutputs Number of output neurons
	 */
	constructor(numInputs, numHidden, numOutputs) {

		this.numInputs = numInputs;
		this.numHidden = numHidden;

		this.weights = [];
		this.biases = [];

		const numNeuronsPerLayer = [ numInputs, ...numHidden, numOutputs ];

		for(let layer = 1; layer < numNeuronsPerLayer.length; ++layer) {

			const numNeurons = numNeuronsPerLayer[layer];
			const numNeuronsInPreviousLayer = numNeuronsPerLayer[layer - 1];

			let layerWeights = Matrix.Random(numNeurons, numNeuronsInPreviousLayer, 0, 1);
			let layerBiases = Matrix.Random(numNeurons, 1, 0, 1);
			
			this.weights.push(layerWeights);
			this.biases.push(layerBiases);
		}
	}

	feedForward(input, layer = 0) {
		
		const layerWeights = this.weights[layer];
		const layerBiases = this.biases[layer];

		const result = Matrix.Multiply(layerWeights, input).add(layerBiases);
		result.apply(x => this.activation(x));

		if(layer < this.numHidden.length)
			return this.feedForward(result, layer + 1);
		else
			return result;
	}

	activation(x) {
		return 1 / (1 + Math.exp(-x)); //sigmoid
	}
}

module.exports = NeuralNetwork;
