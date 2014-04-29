import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.util.Arrays;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.util.TransferFunctionType;

/**
 * This sample shows how to create, train, save and load simple Multi Layer Perceptron
 */
public class MultiLayerPerceptronSample {

	public static void main(String[] args) {

		// create training set (extending XOR sample)
		DataSet trainingSet = new DataSet(2, 1);
		trainingSet.addRow(new DataSetRow(new double[]{0, 0}, new double[]{1}));
		trainingSet.addRow(new DataSetRow(new double[]{0, 1}, new double[]{1}));
		trainingSet.addRow(new DataSetRow(new double[]{1, 0}, new double[]{1}));
		trainingSet.addRow(new DataSetRow(new double[]{1, 1}, new double[]{0}));
		trainingSet.addRow(new DataSetRow(new double[]{2, 2}, new double[]{-1}));
		trainingSet.addRow(new DataSetRow(new double[]{1, 2}, new double[]{-1}));
		trainingSet.addRow(new DataSetRow(new double[]{1, 3}, new double[]{-1}));
		trainingSet.addRow(new DataSetRow(new double[]{2, 2}, new double[]{-1}));
		trainingSet.addRow(new DataSetRow(new double[]{2, 42}, new double[]{-1}));

		// create multi layer perceptron
		MultiLayerPerceptron myMlPerceptron = new MultiLayerPerceptron(TransferFunctionType.TANH, 2, 9, 1);
		// learn the training set

		long start = System.currentTimeMillis();

		myMlPerceptron.learn(trainingSet);

		long time = System.currentTimeMillis()-start;

		System.out.println("It took: "+time+" ms");

		// test perceptron
		System.out.println("Testing trained neural network");
		testNeuralNetwork(myMlPerceptron, trainingSet);

		// save trained neural network
		myMlPerceptron.save("myMlPerceptron.nnet");

		// load saved neural network
		
		FileInputStream stream;
		
		try {
			
			stream = new FileInputStream("myMlPerceptron.nnet");
			
			NeuralNetwork loadedMlPerceptron = NeuralNetwork.load(stream);
		
			// test loaded neural network
			System.out.println("Testing loaded neural network");
			testNeuralNetwork(loadedMlPerceptron, trainingSet);
			
			System.out.println("Testing unknown input");
			testNeuralNetwork(loadedMlPerceptron, new DataSetRow(new double[]{2, 30}));
			
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}

	public static void testNeuralNetwork(NeuralNetwork nnet, DataSetRow dataRow) {

		nnet.setInput(dataRow.getInput());
		nnet.calculate();
		double[ ] networkOutput = nnet.getOutput();
		System.out.print("Input: " + Arrays.toString(dataRow.getInput()) );
		System.out.println(" Output: " + Arrays.toString(networkOutput) );

	}

	public static void testNeuralNetwork(NeuralNetwork nnet, DataSet tset) {

		for(DataSetRow dataRow : tset.getRows()) {

			nnet.setInput(dataRow.getInput());
			nnet.calculate();
			double[ ] networkOutput = nnet.getOutput();
			System.out.print("Input: " + Arrays.toString(dataRow.getInput()) );
			System.out.println(" Output: " + Arrays.toString(networkOutput) );

		}

	}

}