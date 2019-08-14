using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using Microsoft.ML.OnnxRuntime;
using System.Numerics.Tensors;

namespace OnnxPredictionSample
{
    class Program
    {
        const string DirPath = @".\Assets";

        static readonly string ImagePath = Path.Combine(DirPath, "sampleImage.png");

        static readonly string ModelPath = Path.Combine(DirPath, "sampleModel.onnx");

        static void Main(string[] args)
        {
            using (var session = new InferenceSession(ModelPath))
            {
            }
        }

        //static void Main(string[] args)
        //{
        //    using (var session = new InferenceSession(ModelPath))
        //    {
        //        var inputMeta = session.InputMetadata;
        //        var container = new List<NamedOnnxValue>();

        //        float[] inputData = LoadTensorFromFile(ImagePath); // this is the data for only one input tensor for this model

        //        foreach (var name in inputMeta.Keys)
        //        {
        //            var tensor = new DenseTensor<float>(inputData, inputMeta[name].Dimensions);
        //            container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
        //        }

        //        // Run the inference
        //        var results = session.Run(container);  // results is an IReadOnlyList<NamedOnnxValue> container

        //        // dump the results
        //        foreach (var r in results)
        //        {
        //            Console.WriteLine("Output for {0}", r.Name);
        //            Console.WriteLine(r.AsTensor<float>().GetArrayString());
        //        }
        //    }
        //}

        static float[] LoadTensorFromFile(string filename)
        {
            var tensorData = new List<float>();

            // read data from file
            using (var inputFile = new StreamReader(filename))
            {
                inputFile.ReadLine(); //skip the input name
                string[] dataStr = inputFile.ReadLine().Split(new char[] { ',', '[', ']' }, StringSplitOptions.RemoveEmptyEntries);
                for (int i = 0; i < dataStr.Length; i++)
                {
                    tensorData.Add(Single.Parse(dataStr[i]));
                }
            }

            return tensorData.ToArray();
        }
    }
}
