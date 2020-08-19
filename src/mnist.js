
class MNIST {

	constructor() {
	}

	static async loadTrainingImages(imagesBuffer) {

		const startTime = new Date();
		
		// const magicNum = imagesBuffer.readInt32BE(0);
		const numImages = imagesBuffer.readInt32BE(4);
		const numRows = imagesBuffer.readInt32BE(8);
		const numCols = imagesBuffer.readInt32BE(12);

		const basePixelOffset = 16;
		
		const images = [];

		for(let imageStart = 0; imageStart < numImages; ++imageStart) {
			
			const offset = basePixelOffset + imageStart * numRows * numCols;
			const image = [];

			for(let row = 0; row < numRows; ++row)
				for(let col = 0; col < numCols; ++col)
					image.push(imagesBuffer.readUIntBE(offset + (row * numCols) + col, 1));

			images.push(image);
		}

		console.log(`Loaded ${numImages} images in ${new Date() - startTime}ms`);
		return images;
	}

	static async loadTrainingLabels(labelsBuffer) {

		const startTime = new Date();

		// const magicNum = labelsBuffer.readInt32BE(0);
		const numLabels = labelsBuffer.readInt32BE(4);

		const baseLabelOffset = 8;

		const labels = [];

		for(let labelStart = 0; labelStart < numLabels; ++labelStart)
			labels.push(labelsBuffer.readUIntBE(baseLabelOffset + labelStart, 1));

		console.log(`Loaded ${numLabels} labels in ${new Date() - startTime}ms`);
		return labels;
	}
};

module.exports = MNIST;
