//VERSION=3
function setup() {
  return {
    input: ["VV", "VH", "dataMask", "localIncidenceAngle"],
    output: {
      bands: 4, // VV, VH, dataMask, localIncidenceAngle
      sampleType: "FLOAT32"
    }
  };
}

function evaluatePixel(sample) {
  return [sample.VV, sample.VH, sample.dataMask, sample.localIncidenceAngle];
}