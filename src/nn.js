
const Matrix = require('./matrix');

class NeuralNetwork {

	/** @type number[] */
	numNeuronsPerLayer;

	/** @type number */
	numLayers;

	/** @type Matrix[] */
	inputs;

	/** @type Matrix[] */
	activations;

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

		this.inputs = [ null ];
		this.activations = [];
		this.weights = [ null ];
		this.biases = [ null ];

		this.numNeuronsPerLayer = [ numInputs, ...numHidden, numOutputs ];
		this.numLayers = this.numNeuronsPerLayer.length;

		for(let layer = 1; layer < this.numLayers; ++layer) {

			const numNeurons = this.numNeuronsPerLayer[layer];
			const numNeuronsInPreviousLayer = this.numNeuronsPerLayer[layer - 1];

			const layerWeights = Matrix.Random(numNeurons, numNeuronsInPreviousLayer, 0, 1);
			const layerBiases = Matrix.Random(numNeurons, 1, -1, 1);
			
			this.weights.push(layerWeights);
			this.biases.push(layerBiases);
		}
	}

	feedForward(activations, layer = 0) {

		if(layer >= this.numLayers) return activations;
		
		let layerActivations;

		if(layer === 0) {
			layerActivations = activations;
		}
		else {
			const layerWeights = this.weights[layer];
			const layerBiases = this.biases[layer];
	
			const layerInput = Matrix.Add(Matrix.Multiply(layerWeights, activations), layerBiases);
			this.inputs[layer] = layerInput;
	
			layerActivations = layerInput.clone().apply(x => this.activation(x));
		}
		
		this.activations[layer] = layerActivations;
		return this.feedForward(layerActivations, layer + 1);
	}

	activation(x) {
		return 1 / (1 + Math.exp(-x));
	}

	activationDerivative(x) {
		const activation = this.activation(x);
		return activation * (1 - activation);
	}

	train(sampleInput, desiredOutput) {

		const output = this.feedForward(sampleInput);
		const error = this.calculateError(output, desiredOutput);

		this.backPropagate(error, this.numLayers - 1);
	}

	calculateError(output, desiredOutput) {
		
		const error = new Matrix(output.rows, 1);
		const inputs = this.inputs[this.numLayers - 1];

		for(let r = 0; r < output.rows; ++r) {
			const diff = output.matrix[r][0] - desiredOutput.matrix[r][0];
			error.matrix[r][0] = 2 * diff * this.activationDerivative(inputs.matrix[r][0]);
		}

		return error;
	}

	backPropagate(error, layer) {

		if(layer <= 0) return;
		
		let layerError;
		
		if(layer >= this.numLayers - 1) {
			layerError = error;
		}
		else {
			
			const inputs = this.inputs[layer];
			const frontWeights = this.weights[layer + 1];
			
			layerError = Matrix.Multiply(Matrix.Transpose(frontWeights), error);
			for(let r = 0; r < layerError.rows; ++r)
				layerError.matrix[r][0] *= this.activationDerivative(inputs.matrix[r][0]);
		}

		this.backPropagate(layerError, layer - 1);

		const weights = this.weights[layer];
		const biases = this.biases[layer];
		const prevActivations = this.activations[layer - 1];
		
		for(let r = 0; r < weights.rows; ++r) {
			
			for(let c = 0; c < weights.cols; ++c)
				weights.matrix[r][c] -= prevActivations.matrix[c][0] * layerError.matrix[r][0] * this.learningRate;
			
			biases.matrix[r][0] -= layerError.matrix[r][0] * this.learningRate;
		}
	}
}

module.exports = NeuralNetwork;
