/**
 * Created by Josh on 4/24/2017.
 */

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.rules.PART;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.J48;
import weka.core.FastVector;
import weka.core.Instances;

public class Project5MachineLearning {
    public static BufferedReader readDataFile(String filename) {
        BufferedReader inputReader = null;

        try {
            inputReader = new BufferedReader(new FileReader(filename));
        } catch (FileNotFoundException ex) {
            System.err.println("File not found: " + filename);
        }

        return inputReader;
    }

    public static Evaluation classify(Classifier model, Instances trainingSet, Instances testingSet) throws Exception {
        Evaluation evaluation = new Evaluation(trainingSet);

        model.buildClassifier(trainingSet);
        evaluation.evaluateModel(model, testingSet);

        return evaluation;
    }

    public static double calculateAccuracy(FastVector predictions) {
        double correct = 0;

        for (int i = 0; i < predictions.size(); i++) {
            NominalPrediction np = (NominalPrediction) predictions.elementAt(i);
            if (np.predicted() == np.actual()) {
                correct++;
            }
        }

        return 100 * correct / predictions.size();
    }

    public static Instances[][] crossValidationSplit(Instances data, int numberOfFolds) {
        Instances[][] split = new Instances[2][numberOfFolds];

        for (int i = 0; i < numberOfFolds; i++) {
            split[0][i] = data.trainCV(numberOfFolds, i);
            split[1][i] = data.testCV(numberOfFolds, i);
        }

        return split;
    }

    public static void main(String[] args) throws Exception {
        int x = 1;

        while (x < 6)
        {
            if(x < 5)
            {
                System.out.println("This is the results of classifying for a windy day\n");
            }
            else
                System.out.println("This is the results of classifying for daily outlook\n");


            BufferedReader datafile = readDataFile("C:\\Users\\Josh\\IdeaProjects\\CSC204\\src\\weather.txt"); //You will need to input your own path here

            Instances data = new Instances(datafile);
            data.setClassIndex(data.numAttributes() - x); //-1 will classify windy days while -5 will classify outlook
            // note that outlook doesn't classify as accurate
            // Do 10-split cross validation
            Instances[][] split = crossValidationSplit(data, 10);

            // Separate split into training and testing arrays
            Instances[] trainingSplits = split[0];
            Instances[] testingSplits = split[1];

            // Use a set of classifiers
            Classifier[] models = {
                    new J48(), // a decision tree
                    new PART(),
                    new DecisionTable(),//decision table majority classifier
                    new DecisionStump() //one-level decision tree

            };

            // Runs for each model
            for (int j = 0; j < models.length; j++) {

                // Collect every group of predictions for current model in a FastVector
                FastVector predictions = new FastVector();

                // For each training-testing split pair, train and test the classifier
                for (int i = 0; i < trainingSplits.length; i++) {
                    Evaluation validation = classify(models[j], trainingSplits[i], testingSplits[i]);

                    predictions.appendElements(validation.predictions());
                }

                // Calculate overall accuracy of current classifier on all splits
                double accuracy = calculateAccuracy(predictions);
                //if(x < 5)
                //{
                    //System.out.println("This is the results of classifying for a windy day\n");
                //}
                //else
                    //System.out.println("This is the results of classifying for daily outlook\n");
                System.out.println("Accuracy of " + models[j].getClass().getSimpleName() + ": " +
                        String.format("%.2f%%", accuracy) + "\n---------------------------------");


            }

            Classifier cls = new J48();
            Evaluation eval = new Evaluation(data);
            Random rand = new Random(1);  // using seed = 1
            int folds = 10;
            eval.crossValidateModel(cls, data, folds, rand);
            System.out.println(eval.toSummaryString());
            System.out.println("\n---------------------------------");
            System.out.println(eval.toMatrixString());

            x = x + 4;// this causes the second iteration as well as the feature change from windy to outlook
        }

    }

}