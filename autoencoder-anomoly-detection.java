public class AnomolyDetection {
    public static void main(String[] args) throws Exception {

        // set up the network
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(12345)
            .iterations(1)
            .weightInit(WeightInit.XAVIER)
            .updater(Updater.NESTEROVS)
            .activation(Activation.RELU)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .learningRate(0.01)
            .regularization(true).l2(1e-4)
            .list()
            .layer(0, new DenseLayer.Builder()
                .nIn(784)
                .nOut(1000)
                .build())

        // Load the model
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork("model.zip");

        // Load the data
        DataSetIterator testIter = new MnistDataSetIterator(1, false, 12345);
        DataSet test = testIter.next();

        // Get the features
        INDArray features = test.getFeatures();

        // Get the reconstruction
        INDArray reconstruction = model.output(features, false);

        // Get the reconstruction error
        double error = reconstruction.distance2(features);

        // Print the error
        System.out.println("Reconstruction error: " + error);
    }
}