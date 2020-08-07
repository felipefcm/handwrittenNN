
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

	learningRate = 0.01;

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

			const layerWeights = Matrix.Random(numNeurons, numNeuronsInPreviousLayer, 0, 1);
			const layerBiases = Matrix.Random(numNeurons, 1, -1, 1);
			
			this.weights.push(layerWeights);
			this.biases.push(layerBiases);
		}
	}

	train(sampleInput, desiredOutput) {

		const output = this.feedForward(sampleInput);
		const error = this.calculateError(output, desiredOutput);

		this.backPropagate(error);
	}

	feedForward(input, layer = 0) {

		if(layer > this.numHidden.length) return input;
		
		const layerWeights = this.weights[layer];
		const layerBiases = this.biases[layer];

		const result = Matrix.Multiply(layerWeights, input).add(layerBiases);
		result.apply(x => this.activation(x));

		return this.feedForward(result, layer + 1);
	}

	activation(x) {
		return 1 / (1 + Math.exp(-x)); //sigmoid
	}

	backPropagate(error) {

		
	}

	calculateError(output, target) {
		
		let error = new Matrix(output.rows, 1);

		for(let r = 0; r < error.rows; ++r)
			error.matrix[r][0] = Math.pow(target.matrix[r][0] - output.matrix[r][0], 2);

		return error;
	}

	calculateSampleError(error) {
		
		let e = 0;

		for(let r = 0; r < error.rows; ++r)
			e += error.matrix[r][0];

		return e;
	}
}

module.exports = NeuralNetwork;
