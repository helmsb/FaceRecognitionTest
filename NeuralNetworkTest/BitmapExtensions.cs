using System.Drawing;

namespace NeuralNetworkTest
{
    public static class BitmapExtensions
    {
        public static Image Crop(this Image img, Rectangle cropArea)
        {
            var bmpImage = new Bitmap(img);
            return bmpImage.Clone(cropArea, bmpImage.PixelFormat);
        }

        public static byte[] ToGrayScaleArray(this Bitmap image)
        {
            var bytes = new byte[6400];
            var currentByte = 0;

            for (var r = 0; r < 80; r++)
            {
                for (var c = 0; c < 80; c++)
                {
                    var color = image.GetPixel(r, c);
                    bytes[currentByte++] = color.G;
                }
            }

            return bytes;
        }
    }
}