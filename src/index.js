
const NeuralNetwork = require('./nn');
const Matrix = require('./matrix');

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

// testMatrixLib();

let nn = new NeuralNetwork(3, [ 2, 2 ], 1);
let output = nn.feedForward(Matrix.FromArray([ [0.3], [0.8], [0.1] ]));
output.print();
