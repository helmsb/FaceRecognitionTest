using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkTest
{
    public class Perceptron
    {
        private readonly double[] _weights;

        public double Threshold { get; set; }
        public double Bias { get; set; }

        public Perceptron()
        {
            _weights = new double[20];
        }

        public double Activate(double[] input)
        {
            double dp = 0;

            if (input.Count() != _weights.Count()) {
                throw new Exception("The number if inputs does not match the number of weights");
            }

            for (var i = 0; i < input.Count(); i++)
            {
                dp += input[i]*_weights[i] ;
            }

            return dp + Bias > Threshold ? 1 : 0;
        }
    }
}
