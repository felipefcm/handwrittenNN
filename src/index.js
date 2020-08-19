
const fs = require('fs-extra');
const imageOutput = require('image-output');

const NeuralNetwork = require('./nn');
const Matrix = require('./matrix');

const MNIST = require('./mnist');

const testMatrixLib = () => {

	console.log(`--- Identity matrix ---`);
	const id = Matrix.Identity(3);
	id.print();
	
	console.log(`--- RandomInt #1 ---`);
	const rand = Matrix.RandomInt(3, 3);
	rand.print();
	
	console.log(`--- RandomInt #2 ---`);
	const rand2 = Matrix.RandomInt(3, 3);
	rand2.print();
	
	console.log(`--- Multiply ---`);
	const mult = Matrix.Multiply(rand, rand2);
	mult.print();
	
	console.log(`--- Multiply speed test ---`);
	const n = 50000;
	const s = 10;
	let start = new Date();
	for(let i = 0; i < n; ++i) {
		let r1 = Matrix.RandomInt(s, s);
		let r2 = Matrix.RandomInt(s, s);
		let m = Matrix.Multiply(r1, r2);
	}
	console.log(`Took ${new Date() - start}ms`);
};

const testNNCode = () => {

	let testNN = new NeuralNetwork(3, [ 2, 2 ], 3);

	const start = new Date();
	let output = testNN.feedForward(Matrix.FromArray([ [0.3], [0.8], [0.1] ]));
	console.log(`Feedforward took ${new Date() - start}ms`);
	console.log(`Initial output:`);
	output.print();

	const trainingStart = new Date();
	for(let i = 0; i < 10000; ++i)
		testNN.train(Matrix.FromArray([ [0.3], [0.8], [0.1] ]), Matrix.FromArray([ [1.0], [0.0], [0.5] ]))

	console.log(`Training took ${new Date() - trainingStart}ms`);

	output = testNN.feedForward(Matrix.FromArray([ [0.3], [0.8], [0.1] ]));
	console.log(`Trained output:`);
	output.print();
};

const testNNCodeBatch = () => {

	let testNN = new NeuralNetwork(3, [ 2, 2 ], 3, 0.05);

	const start = new Date();
	let output = testNN.feedForward(Matrix.FromArray([ [0.3], [0.8], [0.1] ]));
	console.log(`Feedforward took ${new Date() - start}ms`);
	console.log(`Initial output:`);
	output.print();

	const samples = 30000;
	const batchSize = 100;
	
	const trainingStart = new Date();
	for(let batch = 0; batch < samples / batchSize; ++batch) {

		const examples = [];
		const outputs = [];

		for(let i = 0; i < batchSize; ++i) {
			examples.push(Matrix.FromArray([ [0.3], [0.8], [0.1] ]));
			outputs.push(Matrix.FromArray([ [1.0], [0.0], [0.5] ]));
		}

		testNN.trainMultiple(examples, outputs);
	}

	console.log(`Training took ${new Date() - trainingStart}ms`);

	output = testNN.feedForward(Matrix.FromArray([ [0.3], [0.8], [0.1] ]));
	console.log(`Trained output:`);
	output.print();
};

const testMNIST = async () => {
	await MNIST.loadTrainingImages();
	await MNIST.loadTrainingLabels();
};




const labelToMatrix = (label) => {
	const matrix = new Matrix(10, 1);
	matrix.matrix[label][0] = 1;
	return matrix;
};

const matrixToLabel = (matrix) => {
	
	let max = 0;
	let maxLabel;

	for(let r = 0; r < 10; ++r) {
		if(matrix.matrix[r][0] > max) {
			max = matrix.matrix[r][0];
			maxLabel = r;
		}
	}

	return maxLabel;
};

const imageToMatrix = (image) => {

	const matrix = new Matrix(784, 1);

	for(let r = 0; r < 784; ++r)
		matrix.matrix[r][0] = Math.max(image[r] / 255, 0.2);

	return matrix;
};

const matrixToImage = (matrix) => {

	const image = [];
	
	for(let r = 0; r < 28; ++r)
		for(let c = 0; c < 28; ++c)
			image.push(matrix.matrix[r][c] * 255);

	return image;
};

const shuffleArray = (array) => {
	
	for(let i = array.length - 1; i > 0; i--){
		const j = Math.floor(Math.random() * i);
		const temp = array[i];
		array[i] = array[j];
		array[j] = temp;
	}

	return array;
};

const testNN = async () => {

	const imagesBuffer = await fs.readFile('./train-images-idx3-ubyte');
	const images = await MNIST.loadTrainingImages(imagesBuffer);

	const labelsBuffer = await fs.readFile('./train-labels-idx1-ubyte');
	const labels = await MNIST.loadTrainingLabels(labelsBuffer);

	// const idx = Math.trunc(Math.random() * 60000);
	// console.log(`IDX ${idx}: expected ${labels[idx]}`);

	// imageOutput({
	// 	data: images[idx],
	// 	width: 28,
	// 	height: 28
	// }, 'digit.png');
	
	const nn = new NeuralNetwork(784, [16, 16], 10, 0.04);

	const numTrainingRounds = 1;

	const trainingStart = new Date();
	for(let i = 0; i < numTrainingRounds; ++i) {
		for(let [i, image] of images.entries()) {
			
			const imageInput = imageToMatrix(image);
			const expectedLabel = labels[i];
			const expected = labelToMatrix(expectedLabel);
			
			nn.train(imageInput, expected);
		}
	}
	console.log(`Training took ${new Date() - trainingStart}ms - ${numTrainingRounds} rounds`);

	let correct = 0;
	for(let [i, image] of images.entries()) {
		
		const imageInput = imageToMatrix(image);
		const outputMatrix = nn.feedForward(imageInput);

		const outputLabel = matrixToLabel(outputMatrix);

		const expectedLabel = labels[i];
		// const expected = labelToMatrix(expectedLabel);

		if(expectedLabel === outputLabel)
			++correct;
	}

	console.log(`TRAINING SET ERROR RATE (%) ${100 * (images.length - correct) / images.length}`);
};

const testNNBatch = async () => {

	const imagesBuffer = await fs.readFile('./train-images-idx3-ubyte');
	const images = await MNIST.loadTrainingImages(imagesBuffer);

	const labelsBuffer = await fs.readFile('./train-labels-idx1-ubyte');
	const labels = await MNIST.loadTrainingLabels(labelsBuffer);
	
	const nn = new NeuralNetwork(784, [16, 16], 10, 0.02);

	const numTrainingRounds = 1;

	const trainingStart = new Date();
	for(let i = 0; i < numTrainingRounds; ++i) {

		const batchSize = 200;
		for(let batch = 0; batch < images.length / batchSize; ++batch) {

			const batchImages = images.slice(batch * batchSize, (batch + 1) * batchSize).map(i => imageToMatrix(i));
			const batchLabels = labels.slice(batch * batchSize, (batch + 1) * batchSize).map(l => labelToMatrix(l));

			nn.trainMultiple(batchImages, batchLabels);
		}
	}
	console.log(`Training took ${new Date() - trainingStart}ms - ${numTrainingRounds} rounds`);

	let correct = 0;
	for(let [i, image] of images.entries()) {
		
		const imageInput = imageToMatrix(image);
		const outputMatrix = nn.feedForward(imageInput);

		const outputLabel = matrixToLabel(outputMatrix);

		const expectedLabel = labels[i];
		// const expected = labelToMatrix(expectedLabel);

		if(expectedLabel === outputLabel)
			++correct;
	}

	console.log(`TRAINING SET ERROR RATE (%) ${100 * (images.length - correct) / images.length}`);
};

// testMatrixLib();
// testNNCode();
// testNNCodeBatch();
// testMNIST();
testNN();
// testNNBatch();
