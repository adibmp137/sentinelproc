//VERSION=3
function setup() {
  return {
    input: ["VV", "VH", "dataMask"],
    output: {
      bands: 3, // VV, VH, dataMask
      sampleType: "FLOAT32"
    }
  };
}

function evaluatePixel(sample) {
  return [sample.VV, sample.VH, sample.dataMask];
}