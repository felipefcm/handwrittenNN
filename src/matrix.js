
class Matrix {

	matrix;
	rows;
	cols;

	constructor(rows, cols) {
		
		this.rows = rows;
		this.cols = cols;

		this.matrix = [];
		
		for(let r = 0; r < rows; ++r)
			this.matrix[r] = new Array(cols).fill(0, 0, cols);
	}

	clone() {
		
		const clone = new Matrix(this.rows, this.cols);
		
		for(let r = 0; r < this.rows; ++r)
			for(let c = 0; c < this.cols; ++c)
				clone.matrix[r][c] = this.matrix[r][c];
		
		return clone;
	}

	/**
	 * Execute the function for every element and store the output
	 * @param {(val: number, row: number, col: number)=>Matrix} fn A function receiving the element value and row,col
	 */
	apply(fn) {
		
		for(let r = 0; r < this.rows; ++r)
			for(let c = 0; c < this.cols; ++c)
				this.matrix[r][c] = fn(this.matrix[r][c], r, c);

		return this;
	}

	print() {
		console.table(this.matrix);
	}

	static Identity(n) {
		
		let id = new Matrix(n, n);
		
		for(let i = 0; i < n; ++i)
			id.matrix[i][i] = 1;

		return id;
	}

	static Random(rows, cols, min = 0, max = 50) {
		
		let rand = new Matrix(rows, cols);
		
		for(let r = 0; r < rows; ++r)
			for(let c = 0; c < cols; ++c)
				rand.matrix[r][c] = min + Math.random() * (max - min);
		
		return rand;
	}

	static RandomInt(rows, cols, min = 0, max = 50) {
		
		let rand = Matrix.Random(rows, cols, min, max);
		
		for(let r = 0; r < rows; ++r)
			for(let c = 0; c < cols; ++c)
				rand.matrix[r][c] = Math.trunc(rand.matrix[r][c]);

		return rand;
	}

	static Transpose(a) {
		
		let t = new Matrix(a.cols, a.rows);

		for(let r = 0; r < t.rows; ++r)
			for(let c = 0; c < t.cols; ++c)
				t.matrix[r][c] = a.matrix[c][r];

		return t;
	}

	static Add(a, b) {
		
		if(a.rows !== b.rows || a.cols !== b.cols)
			throw new Error('Add error: dimensions mismatch');

		let sum = new Matrix(a.rows, a.cols);

		for(let r = 0; r < a.rows; ++r)
			for(let c = 0; c < a.cols; ++c)
				sum.matrix[r][c] = a.matrix[r][c] + b.matrix[r][c];

		return sum;
	}

	static Subtract(a, b) {
		
		if(a.rows !== b.rows || a.cols !== b.cols)
			throw new Error('Subtract error: dimensions mismatch');

		let sub = new Matrix(a.rows, a.cols);

		for(let r = 0; r < a.rows; ++r)
			for(let c = 0; c < a.cols; ++c)
				sub.matrix[r][c] = a.matrix[r][c] - b.matrix[r][c];

		return sub;
	}

	static Multiply(a, b) {

		if(a.cols !== b.rows)
			throw new Error('Multiply error: dimensions mismatch');

		let mult = new Matrix(a.rows, b.cols);

		for(let r = 0; r < mult.rows; ++r)
			for(let c = 0; c < mult.cols; ++c)
				for(let ri = 0; ri < a.cols; ++ri)
					mult.matrix[r][c] += a.matrix[r][ri] * b.matrix[ri][c];

		return mult;
	}

	static FromArray(arr) {
		
		let rows = arr.length;
		let cols = arr.length === 0 ? 0 : arr[0].length;

		let mat = new Matrix(rows, cols);
		mat.matrix = arr;

		return mat;
	}
}

module.exports = Matrix;
