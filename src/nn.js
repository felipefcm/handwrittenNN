
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

	/** @type Matrix[] */
	errors;

	/** @type number */
	learningRate;

	/**
	 * 
	 * @param {number} numInputs Number of input neurons
	 * @param {number[]} numHidden Number of hidden layers and neurons. E.g. [3, 2, 3] for 3 hidden layers with 3, 2 and 3 neurons each
	 * @param {number} numOutputs Number of output neurons
	 */
	constructor(numInputs, numHidden, numOutputs, learningRate = 0.01) {

		this.inputs = [ null ];
		this.activations = [];
		this.weights = [ null ];
		this.biases = [ null ];
		this.errors = [ null ];

		this.learningRate = learningRate;

		this.numNeuronsPerLayer = [ numInputs, ...numHidden, numOutputs ];
		this.numLayers = this.numNeuronsPerLayer.length;

		for(let layer = 1; layer < this.numLayers; ++layer) {

			const numNeurons = this.numNeuronsPerLayer[layer];
			const numNeuronsInPreviousLayer = this.numNeuronsPerLayer[layer - 1];

			const layerWeights = Matrix.Random(numNeurons, numNeuronsInPreviousLayer, -1, 1);
			// layerWeights.multiplyScalar(1 / Math.sqrt(numNeuronsInPreviousLayer)); //for sigmoid
			layerWeights.multiplyScalar(Math.sqrt(2 / numNeuronsInPreviousLayer)); //for ReLU

			const layerBiases = new Matrix(numNeurons, 1, 1);
			
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
	
			const layerInput = Matrix.Multiply(layerWeights, activations).add(layerBiases);
			this.inputs[layer] = layerInput;
	
			layerActivations = layerInput.clone().apply(x => this.activation(x));
		}
		
		this.activations[layer] = layerActivations;
		return this.feedForward(layerActivations, layer + 1);
	}

	activation(x) {
		// return 1 / (1 + Math.exp(-x)); //sigmoid
		return Math.max(0, x); //ReLU
	}

	activationDerivative(x) {
		
		//sigmoid
		// const activation = this.activation(x);
		// return activation * (1 - activation);

		//ReLU
		return x > 0 ? 1 : 0;
	}

	train(sampleInput, desiredOutput) {

		const output = this.feedForward(sampleInput);
		const error = this.calculateError(output, desiredOutput);

		this.backPropagate(error, this.numLayers - 1);

		for(let layer = 1; layer < this.numLayers; ++layer) {

			const layerError = this.errors[layer];
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

	trainMultiple(sampleInputs, desiredOutputs) {

		const num = sampleInputs.length;

		const accumWeightGradients = [ null ];
		const accumBiasGradients = [ null ];

		for(let layer = 1; layer < this.numLayers; ++layer) {
			const numNeurons = this.numNeuronsPerLayer[layer];
			const numNeuronsInPreviousLayer = this.numNeuronsPerLayer[layer - 1];
			accumWeightGradients.push(new Matrix(numNeurons, numNeuronsInPreviousLayer));
			accumBiasGradients.push(new Matrix(numNeurons, 1));
		}

		for(let [i, sampleInput] of sampleInputs.entries()) {
			
			const output = this.feedForward(sampleInput);
			const error = this.calculateError(output, desiredOutputs[i]);

			this.backPropagate(error, this.numLayers - 1);

			for(let layer = this.numLayers - 1; layer >= 1; --layer) {
				
				const weightGradient = Matrix.Multiply(this.errors[layer], Matrix.Transpose(this.activations[layer - 1]));
				weightGradient.multiplyScalar(this.learningRate / num);
				accumWeightGradients[layer].add(weightGradient);

				const biasGradient = this.errors[layer].clone().multiplyScalar(this.learningRate / num);
				accumBiasGradients[layer].add(biasGradient);
			}
		}

		for(let layer = this.numLayers - 1; layer >= 1; --layer) {
			
			const layerAccWeightGradients = accumWeightGradients[layer];//.multiplyScalar(this.learningRate);
			const layerAccBiasGradients = accumBiasGradients[layer];//.multiplyScalar(this.learningRate);

			this.weights[layer].subtract(layerAccWeightGradients);
			this.biases[layer].subtract(layerAccBiasGradients);
		}
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

		this.errors[layer] = layerError;
		this.backPropagate(layerError, layer - 1);
	}

	normalize(input) {

		let max = 0;
		for(let r = 0; r < input.rows; ++r)
			max = Math.max(max, input.matrix[r][0]);

		for(let r = 0; r < input.rows; ++r)
			input.matrix[r][0] /= max;

		return input;
	}
}

module.exports = NeuralNetwork;
