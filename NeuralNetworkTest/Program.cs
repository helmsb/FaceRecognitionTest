using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;

namespace NeuralNetworkTest
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            Console.WriteLine("\n -| Begin Perceptron demo |- \n");

            const int maxEpochs = 1000000;
            const double alpha = 0.001;
            const double targetError = 0.0;
            double bestBias;
            var trainingData = LoadTrainingData().ToList();

            double[] bestWeights = FindBestWeights(trainingData, maxEpochs, alpha, targetError, out bestBias);
            Console.WriteLine("\n Training complete \n");


            var tf1 = new Bitmap(Environment.CurrentDirectory + "\\..\\..\\TestData\\tf1.jpg").ToGrayScaleArray();
            var tf2 = new Bitmap(Environment.CurrentDirectory + "\\..\\..\\TestData\\tf2.jpg").ToGrayScaleArray();
            var tf3 = new Bitmap(Environment.CurrentDirectory + "\\..\\..\\TestData\\tf3.jpg").ToGrayScaleArray();
            var tn1 = new Bitmap(Environment.CurrentDirectory + "\\..\\..\\TestData\\tn1.jpg").ToGrayScaleArray();
            var tn2 = new Bitmap(Environment.CurrentDirectory + "\\..\\..\\TestData\\tn2.jpg").ToGrayScaleArray();
            var tn3 = new Bitmap(Environment.CurrentDirectory + "\\..\\..\\TestData\\tn3.jpg").ToGrayScaleArray();

            var testData = new Dictionary<string, byte[]>
            {
                {"Face 1", tf1},
                {"Face 2", tf2},
                {"Face 3", tf3},
                {"No Face 1", tn1},
                {"No Face 2", tn2},
                {"No Face 3", tn3}
            };


            Console.WriteLine("\n Evaluating Inputs ... \n");

            foreach (var data in testData)
            {
                
                var prediction = Predict(data.Value, bestWeights, bestBias); // perform the classification

                Console.Write(" " + data.Key + ":");

                if (prediction == 0)
                {
                    Console.Write(" No");
                }

                Console.WriteLine(" Face Detected");
            }


            Console.ReadLine();
        }

        private static IEnumerable<TrainingObject> LoadTrainingData()
        {
            var trainingDataPath = Environment.CurrentDirectory + "\\..\\..\\TrainingData\\";
            var trainingDataFiles = Directory.GetFiles(trainingDataPath);
            var trainingData = trainingDataFiles.Select(f => new TrainingObject
            {
                Data = new Bitmap(f).ToGrayScaleArray(),
                ContainsFeature = Path.GetFileName(f).Contains("f")
            }).ToList();

            return trainingData;
        }

        public static int ComputeOutput(byte[] trainVector, double[] weights, double bias)
        {
            var dotP = trainVector.Select((t, j) => (t * weights[j])).Sum();

            dotP += bias;

            return dotP > 0.5 ? 1 : 0;
        }

        public static int Predict(byte[] dataVector, double[] bestWeights, double bestBias)
        {
            var dotP = dataVector.Select((t, j) => (t * bestWeights[j])).Sum();
            dotP += bestBias;

            return dotP > 0.5 ? 1 : 0;
        }

        public static double TotalError(IEnumerable<TrainingObject> trainingData, double[] weights, double bias)
        {
            var sum = 0.0d;

            var trainingImages = trainingData as IList<TrainingObject> ?? trainingData.ToList();
            foreach (var td in trainingImages)
            {
                var desired = td.ContainsFeature ? 1 : 0;
                var output = ComputeOutput(td.Data, weights, bias);
                sum += (desired - output) * (desired - output); // equivalent to Abs(desired - output) in this case
            }

            return 0.5*sum;
        }

        public static double[] FindBestWeights(IEnumerable<TrainingObject> trainingData, int maxEpochs, double alpha, double targetError, out double bestBias)
        {
            var trainingDataList = trainingData.ToList();
            var dim = trainingDataList.First().Data.Length;
            var weights = new double[dim]; // implicitly all 0.0
            var bias = 0.01;
            var totalError = double.MaxValue;
            var epoch = 0;

            while (epoch < maxEpochs && totalError > targetError)
            {
                foreach (var t in trainingDataList)
                {
                    var desired = t.ContainsFeature ? 1 : 0;
                    var output = ComputeOutput(t.Data, weights, bias);
                    var delta = desired - output; // -1 (if output too large), 0 (output correct), or +1 (output too small)

                    for (var j = 0; j < weights.Length; ++j)
                    {
                        weights[j] = weights[j] + (alpha * delta * t.Data[j]); // corrects weight
                    }

                    bias = bias + (alpha*delta);
                }

                totalError = TotalError(trainingDataList, weights, bias); // rescans; could do in for loop
                ++epoch;
            }

            bestBias = bias;

            return weights;
        }
    } 
}
