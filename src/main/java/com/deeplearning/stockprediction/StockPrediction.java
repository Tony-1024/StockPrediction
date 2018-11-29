package com.deeplearning.stockprediction;

import com.deeplearning.network.LSTMNetwork;
import com.deeplearning.util.DrawingTool;
import com.deeplearning.util.PropertiesUtil;
import javafx.util.Pair;
import org.apache.log4j.PropertyConfigurator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.File;
import java.io.IOException;
import java.util.List;

/**
 * NY Stock Exchange prediction
 */
public class StockPrediction {

	private static final Logger logger = LoggerFactory.getLogger(StockPrediction.class);

	public static void main(String[] args) throws IOException {
		logger.info("Application is starting!");
		PropertyConfigurator.configure(new File("log4j.properties").getAbsolutePath());
		
		int batchSize = PropertiesUtil.getBatchSize();
        int exampleLength = PropertiesUtil.getExampleLength();
        int firstTestItemNumber = PropertiesUtil.getFirstTestItemNumber();
        int testItems = PropertiesUtil.getTestItems();
        String symbol = PropertiesUtil.getStockName(); // stock name

		String file = new ClassPathResource(PropertiesUtil.getDatasetFilename()).getFile().getAbsolutePath();

		logger.info("Create dataSet iterator...");
		StockDataSetIterator iterator = new StockDataSetIterator(file, symbol, batchSize, exampleLength,
				firstTestItemNumber, testItems);

		logger.info("Load test dataset...");
		List<Pair<INDArray, INDArray>> test = iterator.getTestDataSet();
		
		trainAndTest(iterator, test);
	}

	private static void trainAndTest(StockDataSetIterator iterator, List<Pair<INDArray, INDArray>> test) throws IOException {
		logger.info("Build lstm networks...");
		String fileName = "StockLSTM_" + PropertiesUtil.getWaveletType() + ".zip";
		File locationToSave = new File("savedModels/" + fileName);
		MultiLayerNetwork net = LSTMNetwork.buildLSTMNetwork(iterator.inputColumns(), iterator.totalOutcomes());
		// if not use saved model, train new model
		if(!PropertiesUtil.getUseSavedModel()) {
			logger.info("starting to train LSTM networks with " +PropertiesUtil.getWaveletType()+ " wavelet...");
			for (int i = 0; i < PropertiesUtil.getEpochs(); i++) {
				logger.info("training at epoch "+i);
				DataSet dataSet;
				while (iterator.hasNext()) {
					dataSet = iterator.next();
					net.fit(dataSet);
				}
				iterator.reset(); // reset iterator
				net.rnnClearPreviousState(); // clear previous state
			}
			// save model to file
			logger.info("saving trained network model...");
			ModelSerializer.writeModel(net, locationToSave, true);
		} else {
			logger.info("loading network model...");
			net = ModelSerializer.restoreMultiLayerNetwork(locationToSave);
		}
		// testing
		test(net, test, iterator, PropertiesUtil.getExampleLength(), PropertiesUtil.getEpochs());
		
        logger.info("Both the training and testing are finished, system is exiting...");
        System.exit(0);

	} 

	private static void test(MultiLayerNetwork net, List<Pair<INDArray, INDArray>> test, StockDataSetIterator iterator,
			int exampleLength, int epochNum) {
		logger.info("Testing...");
		INDArray max = Nd4j.create(iterator.getMaxNum());
		INDArray min = Nd4j.create(iterator.getMinNum());
		INDArray[] predicts = new INDArray[test.size()];
		INDArray[] actuals = new INDArray[test.size()];

		double[] mseValue = new double[PropertiesUtil.getVectorSize()];

		for (int i = 0; i < test.size(); i++) {
			predicts[i] = net.rnnTimeStep(test.get(i).getKey()).getRow(exampleLength - 1).mul(max.sub(min)).add(min);
			actuals[i] = test.get(i).getValue();
			// Calculate the MSE of results
			mseValue[0] += Math.pow((actuals[i].getDouble(0, 0) - predicts[i].getDouble(0, 0)), 2);
			mseValue[1] += Math.pow((actuals[i].getDouble(0, 1) - predicts[i].getDouble(0, 1)), 2);
			mseValue[2] += Math.pow((actuals[i].getDouble(0, 2) - predicts[i].getDouble(0, 2)), 2);
			mseValue[3] += Math.pow((actuals[i].getDouble(0, 3) - predicts[i].getDouble(0, 3)), 2);
//			mseValue[4] += Math.pow((actuals[i].getDouble(0, 4) - predicts[i].getDouble(0, 4)), 2);
		}

		double mseOpen = Math.sqrt(mseValue[0] / test.size());
		double mseClose = Math.sqrt(mseValue[1] / test.size());
		double mseLow = Math.sqrt(mseValue[2] / test.size());
		double mseHigh = Math.sqrt(mseValue[3] / test.size());
//		double mseVOLUME = Math.sqrt(mseValue[4] / test.size());
//		logger.info("MSE for [Open,Close,Low,High,VOLUME] is: [" + mseOpen + ", " + mseClose + ", " + mseLow + ", "
//				+ mseHigh + ", " + mseVOLUME);
		logger.info("MSE for [Open,Close,Low,High] is: [" + mseOpen + ", " + mseClose + ", " + mseLow + ", " + mseHigh + "]");

		// plot predicts and actual values
		logger.info("Starting to print out values.");
		for (int i = 0; i < predicts.length; i++)
			logger.info("Prediction="+predicts[i] + ", Actual=" + actuals[i]);
		logger.info("Drawing chart...");
		plotAll(predicts, actuals, epochNum);
		logger.info("Finished drawing...");
	}

	/**
	 * plot all predictions
	 * 
	 * @param predicts
	 * @param actuals
	 * @param epochNum
	 */
	private static void plotAll(INDArray[] predicts, INDArray[] actuals, int epochNum) {
//		String[] titles = { "Open", "Close", "Low", "High", "VOLUME" };
		String[] titles = { "Open", "Close", "Low", "High" };
		for (int n = 0; n < PropertiesUtil.getVectorSize(); n++) {
			double[] pred = new double[predicts.length];
			double[] actu = new double[actuals.length];
			for (int i = 0; i < predicts.length; i++) {
				pred[i] = predicts[i].getDouble(n);
				actu[i] = actuals[i].getDouble(n);
			}
			DrawingTool.drawChart(pred, actu, titles[n], epochNum);
		}
	}
}
